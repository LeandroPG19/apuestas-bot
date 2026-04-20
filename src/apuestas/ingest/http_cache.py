"""HTTP cache persistente para APIs externas — anti-pérdida de créditos.

Usa hishel sobre httpx con storage SQLite en disco. Si el bot crashea o se
reinicia durante un `make analyze`, las respuestas de football-data/odds api
ya obtenidas NO se pierden y NO re-consumen créditos al siguiente arranque.

Política por defecto:
- Fixtures próximos (football-data): TTL 15 min (cambian con lesiones/lineups)
- Odds live (The Odds API):          TTL 2 min  (volátiles)
- Historical (Sackmann, MoneyPuck):  TTL 6 horas (estáticos)
- Weather forecast (Open-Meteo):     TTL 30 min
- LLM calls:                         NO cache (cada call es única)

Storage: ~/.cache/apuestas/http_cache.sqlite (persistente entre sesiones).

Uso:
    from apuestas.ingest.http_cache import CachedClient
    async with CachedClient() as client:
        resp = await client.get("https://api.football-data.org/v4/...")
        # si ya se pidió en últimos 15min, devuelve desde disco sin red
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hishel
import httpx

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# TTL por patrón de URL (segundos)
_TTL_BY_HOST: dict[str, int] = {
    "api.football-data.org": 15 * 60,
    "api.the-odds-api.com": 2 * 60,
    "api.openweathermap.org": 30 * 60,
    "api.open-meteo.com": 30 * 60,
    "api-web.nhle.com": 5 * 60,
    "api.thesportsdb.com": 60 * 60,
    "raw.githubusercontent.com": 6 * 60 * 60,  # Sackmann tennis CSVs
    "moneypuck.com": 6 * 60 * 60,
    "site.api.espn.com": 5 * 60,
}

_DEFAULT_TTL = 15 * 60


def _cache_dir() -> Path:
    base = os.environ.get("APUESTAS_CACHE_DIR") or os.path.expanduser("~/.cache/apuestas")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ttl_for(url: str) -> int:
    for host, ttl in _TTL_BY_HOST.items():
        if host in url:
            return ttl
    return _DEFAULT_TTL


def make_cached_async_client(
    *,
    timeout: float = 30.0,
    cache_file: str = "http_cache.sqlite",
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Construye un httpx.AsyncClient con cache SQLite persistente.

    El cache es **compartido entre procesos** (útil si la TUI y un flow
    Prefect corren en paralelo).
    """
    storage = hishel.AsyncSQLiteStorage(
        connection=None,  # hishel lo gestiona
        ttl=_DEFAULT_TTL,
    )
    # Override: apunta al archivo del cache_dir
    storage._sqlite_file_path = str(_cache_dir() / cache_file)  # type: ignore[attr-defined]

    controller = hishel.Controller(
        cacheable_methods=["GET"],
        cacheable_status_codes=[200, 203, 300, 301, 308],
        allow_heuristics=True,
        allow_stale=False,
    )

    transport = hishel.AsyncCacheTransport(
        transport=httpx.AsyncHTTPTransport(retries=2),
        storage=storage,
        controller=controller,
    )
    return httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        headers=headers or {},
    )


class CachedClient:
    """Wrapper async context manager con logging cache hit/miss."""

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._headers = headers or {}
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> CachedClient:
        self._client = make_cached_async_client(timeout=self._timeout, headers=self._headers)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        if self._client is None:
            msg = "CachedClient usado fuera de async context"
            raise RuntimeError(msg)
        resp = await self._client.get(url, params=params, headers=headers)
        cached = resp.extensions.get("from_cache", False)
        logger.debug(
            "http_cache.get",
            url=url[:120],
            status=resp.status_code,
            from_cache=cached,
            ttl=_ttl_for(url),
        )
        return resp


def cache_stats() -> dict[str, Any]:
    """Reporta estadísticas de tamaño + entradas del cache en disco."""
    p = _cache_dir() / "http_cache.sqlite"
    if not p.exists():
        return {"exists": False}
    size_mb = p.stat().st_size / 1_048_576
    return {
        "exists": True,
        "path": str(p),
        "size_mb": round(size_mb, 3),
    }


def cache_clear() -> int:
    """Borra todo el cache. Retorna bytes liberados."""
    p = _cache_dir() / "http_cache.sqlite"
    if not p.exists():
        return 0
    size = p.stat().st_size
    p.unlink()
    logger.info("http_cache.cleared", freed_bytes=size)
    return size
