"""Cliente HTTP base con rate limiting, retry exponential y circuit breaker.

Todos los clientes de APIs externas (The Odds API, API-Football, nba_api,
Reddit, ESPN, OpenWeatherMap) heredan de BaseAPIClient.

Features:
- httpx AsyncClient con HTTP/2
- aiolimiter para rate limit global coordinado vía Valkey
- stamina retry (exponential + jitter)
- pybreaker circuit breaker (fail-fast cuando API down)
- correlation_id propagation
- OpenTelemetry auto-instrumentation
- Tracking de créditos API restantes en Valkey (para TUI SystemStatus)
"""

from __future__ import annotations

import os
import uuid
from abc import ABC
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
import pybreaker
import stamina
from aiolimiter import AsyncLimiter

from apuestas.obs.logging import get_logger


async def _cache_api_credits(source: str, remaining: int) -> None:
    """Persiste créditos restantes en Valkey para dashboards live.

    Clave: `api_credits:<source>` (ej. api_credits:odds_api).
    Legacy alias: `odds_api_credits_remaining` (usado por TUI).
    """
    url = os.environ.get("VALKEY_URL", "") or "redis://localhost:6379/0"
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(url, socket_timeout=1, decode_responses=True)
        await r.set(f"api_credits:{source}", str(remaining), ex=86400)
        if source == "odds_api":
            await r.set("odds_api_credits_remaining", str(remaining), ex=86400)
        await r.aclose()
    except Exception:
        pass


async def _is_quota_exhausted(source: str) -> bool:
    """True si la fuente está en cooldown por quota agotada."""
    url = os.environ.get("VALKEY_URL", "") or "redis://localhost:6379/0"
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(url, socket_timeout=1, decode_responses=True)
        val = await r.get(f"quota_exhausted:{source}")
        await r.aclose()
        return val == "1"
    except Exception:
        return False


async def _mark_quota_exhausted(source: str, ttl_seconds: int = 86400) -> None:
    """Marca fuente como agotada por TTL (default 24h)."""
    url = os.environ.get("VALKEY_URL", "") or "redis://localhost:6379/0"
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(url, socket_timeout=1, decode_responses=True)
        await r.set(f"quota_exhausted:{source}", "1", ex=ttl_seconds)
        await r.aclose()
    except Exception:
        pass


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = get_logger(__name__)


class APIRateLimitError(Exception):
    """429 or 503 sostenido."""


class APICircuitOpenError(Exception):
    """Circuit breaker open; falla rápido."""


class APIQuotaExhaustedError(Exception):
    """Cuota mensual/diaria agotada; cooldown 24h para no spammear retries."""


class BaseAPIClient(ABC):
    """Cliente base para APIs externas.

    Subclasses deben definir:
    - base_url: str
    - source_name: str (para logs y rate limiters)
    - rate_limit: tuple[int, float] -> (max_calls, per_seconds)
    """

    base_url: str = ""
    source_name: str = "unknown"
    rate_limit: tuple[int, float] = (60, 60.0)  # default: 60 req/min

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        http2: bool = True,
    ) -> None:
        if not self.base_url:
            msg = f"{type(self).__name__} must define base_url"
            raise ValueError(msg)
        self._api_key = api_key
        self._timeout = timeout
        self._http2 = http2
        self._client: httpx.AsyncClient | None = None
        max_calls, per_seconds = self.rate_limit
        self._limiter = AsyncLimiter(max_calls, per_seconds)
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=10,
            reset_timeout=60,
            name=f"circuit_{self.source_name}",
        )

    def _default_headers(self) -> dict[str, str]:
        headers = {"User-Agent": f"apuestas-bot/0.1 (+{self.source_name})"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    @asynccontextmanager
    async def session(self) -> AsyncIterator[BaseAPIClient]:
        """Context manager que abre/cierra httpx client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            http2=self._http2,
            headers=self._default_headers(),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True,
        )
        try:
            yield self
        finally:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = f"{type(self).__name__} usado fuera de session()"
            raise RuntimeError(msg)
        return self._client

    @stamina.retry(
        on=(httpx.HTTPError, httpx.ReadTimeout, APIRateLimitError),
        attempts=4,
        wait_initial=1.0,
        wait_max=30.0,
        wait_jitter=2.0,
        wait_exp_base=2.0,
    )
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> httpx.Response:
        """HTTP request con rate limit + circuit breaker + retry."""
        correlation_id = correlation_id or uuid.uuid4().hex[:8]

        if await _is_quota_exhausted(self.source_name):
            msg = f"Quota exhausted for {self.source_name} (24h cooldown active)"
            raise APIQuotaExhaustedError(msg)

        try:
            self._breaker._state_storage.state
        except pybreaker.CircuitBreakerError as exc:
            msg = f"Circuit open for {self.source_name}"
            raise APICircuitOpenError(msg) from exc

        async with self._limiter:
            logger.debug(
                "http.request",
                source=self.source_name,
                method=method,
                path=path,
                cid=correlation_id,
            )
            resp = await self.client.request(method, path, params=params, json=json_body)

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "5"))
            logger.warning(
                "http.rate_limited",
                source=self.source_name,
                retry_after=retry_after,
                cid=correlation_id,
            )
            msg = f"Rate limited by {self.source_name}, retry after {retry_after}s"
            raise APIRateLimitError(msg)

        if resp.status_code in (401, 403):
            body_lower = resp.text[:500].lower()
            quota_markers = (
                "usage quota",
                "quota exceeded",
                "quota has been reached",
                "rate limit",
            )
            if any(m in body_lower for m in quota_markers):
                logger.error(
                    "http.quota_exhausted",
                    source=self.source_name,
                    status=resp.status_code,
                    cid=correlation_id,
                )
                await _mark_quota_exhausted(self.source_name, ttl_seconds=86400)
                msg = f"Quota exhausted for {self.source_name}; cooldown 24h"
                raise APIQuotaExhaustedError(msg)

        if resp.status_code >= 500:
            logger.warning(
                "http.server_error",
                source=self.source_name,
                status=resp.status_code,
                cid=correlation_id,
            )
            resp.raise_for_status()

        if resp.status_code >= 400:
            logger.error(
                "http.client_error",
                source=self.source_name,
                status=resp.status_code,
                body=resp.text[:500],
                cid=correlation_id,
            )
            resp.raise_for_status()

        return resp

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> Any:
        resp = await self._request("GET", path, params=params, correlation_id=correlation_id)
        # Rastrear créditos restantes de APIs que los exponen en headers
        # (para que SystemStatus de la TUI muestre valores live).
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            try:
                await _cache_api_credits(self.source_name, int(remaining))
            except Exception:
                pass
        return resp.json()

    async def post(
        self,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> Any:
        resp = await self._request("POST", path, json_body=json_body, correlation_id=correlation_id)
        return resp.json()
