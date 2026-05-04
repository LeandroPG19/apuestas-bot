"""FastAPI entry point — health, metrics, routers."""

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from apuestas import __version__
from apuestas.config import get_settings
from apuestas.db import dispose_engine, session_scope
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)
settings = get_settings()

_LAN_HOSTS = [
    h.strip() for h in os.environ.get("APUESTAS_ALLOWED_HOSTS", "").split(",") if h.strip()
]
if not _LAN_HOSTS:
    # Starlette TrustedHost NO acepta wildcards IP parciales (e.g. "192.168.*").
    # Para LAN: setear APUESTAS_ALLOWED_HOSTS con hosts concretos
    # (ej. "apuestas.local,192.168.1.100"). Default permisivo para dev.
    _LAN_HOSTS = (
        ["localhost", "127.0.0.1", "apuestas-api", "*"]
        if not settings.is_prod
        else [
            "localhost",
            "127.0.0.1",
            "apuestas-api",
        ]
    )

_CORS_ORIGINS = [
    o.strip() for o in os.environ.get("APUESTAS_CORS_ORIGINS", "").split(",") if o.strip()
]
if not _CORS_ORIGINS:
    _CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3301"]


# Rate limiter con storage Valkey si VALKEY_URL está disponible (consistente
# entre workers granian); fallback a memoria para dev/tests.
_limiter_storage = os.environ.get("VALKEY_URL") or ""
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    storage_uri=_limiter_storage if _limiter_storage else "memory://",
    strategy="fixed-window",
    headers_enabled=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    if settings.is_prod:
        pwd = settings.database.postgres_password.get_secret_value()
        if pwd.startswith(("change-me", "your-", "password", "changeme")) or len(pwd) < 16:
            msg = "Refusing to start in prod with default/weak POSTGRES_PASSWORD"
            raise RuntimeError(msg)
    logger.info("app.startup", env=settings.apuestas_env.value, version=__version__)
    yield
    logger.info("app.shutdown")
    await dispose_engine()


app = FastAPI(
    title="Apuestas Bot API",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_prod else None,
    redoc_url="/redoc" if not settings.is_prod else None,
    openapi_url="/openapi.json" if not settings.is_prod else None,
)

app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    resp = JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMITED",
                "message": f"Rate limit exceeded: {exc.detail}",
            }
        },
    )
    # Retry-After según el window del límite violado
    retry_after = getattr(exc, "retry_after", None) or 60
    resp.headers["Retry-After"] = str(int(retry_after))
    return resp


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_LAN_HOSTS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)


@app.middleware("http")
async def security_headers(request: Request, call_next) -> Response:
    resp: Response = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-site"
    if settings.is_prod:
        resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    path = request.url.path
    if path not in ("/docs", "/redoc", "/openapi.json"):
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"
        )
    return resp


_metrics_token = os.environ.get("APUESTAS_METRICS_TOKEN", "")
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
    should_gzip=True,
)


@app.middleware("http")
async def metrics_guard(request: Request, call_next):
    if request.url.path == "/metrics" and _metrics_token:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {_metrics_token}":
            return JSONResponse(
                status_code=401,
                content={"error": {"code": "UNAUTHORIZED", "message": "Metrics token required"}},
            )
    return await call_next(request)


class HealthCheck(BaseModel):
    status: str = Field(default="ok")
    version: str
    env: str
    checks: dict[str, str]


@app.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
@limiter.limit("120/minute")
async def health(request: Request, response: Response) -> HealthCheck:
    """Healthcheck ligero — verifica DB y dependencias críticas.

    Rate limit alto (120/min) porque Docker healthcheck hace polling
    cada 30s = 2/min; dashboards externos pueden sumar más.
    """
    checks: dict[str, str] = {}

    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc.__class__.__name__}"

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.llm.llama_server_url}/health")
            checks["llm"] = "ok" if resp.status_code == 200 else f"status:{resp.status_code}"
    except Exception:
        checks["llm"] = "unreachable"

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.llm.tei_url}/health")
            checks["embed"] = "ok" if resp.status_code == 200 else f"status:{resp.status_code}"
    except Exception:
        checks["embed"] = "unreachable"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return HealthCheck(
        status=overall, version=__version__, env=settings.apuestas_env.value, checks=checks
    )


@app.get("/version")
@limiter.limit("30/minute")
async def version(request: Request, response: Response) -> dict[str, str]:
    return {"version": __version__, "env": settings.apuestas_env.value}


@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request, response: Response) -> dict[str, str]:
    return {"app": "apuestas", "status": "running", "docs": "/docs"}


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Exporta métricas en formato Prometheus (Deuda 2).

    Restringido a IPs internas por Caddy/nginx en producción (per §9 del
    plan). Si prometheus_client no instalado, devuelve 503.
    """
    from apuestas.obs.metrics import render_metrics

    body = render_metrics()
    if not body:
        return Response(content=b"# prometheus_client not installed\n", status_code=503)
    return Response(content=body, media_type="text/plain; version=0.0.4")
