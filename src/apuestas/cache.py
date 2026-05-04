"""Cache genérico en Valkey (Gap 8 / A3).

Expone get/set con TTL + client singleton. Uso:

    from apuestas.cache import cache_get, cache_set

    cached = await cache_get("best_odds:123:h2h:home")
    if cached is None:
        cached = await fetch_from_db(...)
        await cache_set("best_odds:123:h2h:home", cached, ttl_seconds=600)

Fail-safe: si Valkey no está disponible, todas las ops retornan None/False
y el caller cae al path sin cache.
"""

from __future__ import annotations

import json
from typing import Any

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_client: Any = None


async def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    try:
        import redis.asyncio as aioredis

        from apuestas.config import get_settings

        url = str(get_settings().valkey.url)
        _client = aioredis.from_url(url, decode_responses=True)
        return _client
    except Exception as exc:
        logger.debug("cache.client_unavailable", error=str(exc)[:80])
        return None


async def cache_get(key: str) -> Any | None:
    client = await _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug("cache.get_fail", key=key, error=str(exc)[:80])
        return None


async def cache_set(key: str, value: Any, *, ttl_seconds: int = 600) -> bool:
    client = await _get_client()
    if client is None:
        return False
    try:
        await client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
        return True
    except Exception as exc:
        logger.debug("cache.set_fail", key=key, error=str(exc)[:80])
        return False


async def cache_delete(key: str) -> bool:
    client = await _get_client()
    if client is None:
        return False
    try:
        await client.delete(key)
        return True
    except Exception:
        return False


__all__ = ["cache_delete", "cache_get", "cache_set"]
