"""FastAPI entry point — health, metrics, routers."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, status
from fastapi.responses import ORJSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sqlalchemy import text

from apuestas import __version__
from apuestas.config import get_settings
from apuestas.db import dispose_engine, session_scope
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("app.startup", env=settings.apuestas_env.value, version=__version__)
    yield
    logger.info("app.shutdown")
    await dispose_engine()


app = FastAPI(
    title="Apuestas Bot API",
    version=__version__,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_prod else None,
    redoc_url="/redoc" if not settings.is_prod else None,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class HealthCheck(BaseModel):
    status: str = Field(default="ok")
    version: str
    env: str
    checks: dict[str, str]


@app.get("/health", response_model=HealthCheck, status_code=status.HTTP_200_OK)
async def health() -> HealthCheck:
    """Healthcheck ligero — verifica DB y dependencias críticas."""
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
async def version() -> dict[str, str]:
    return {"version": __version__, "env": settings.apuestas_env.value}


@app.get("/")
async def root() -> dict[str, str]:
    return {"app": "apuestas", "status": "running", "docs": "/docs"}
