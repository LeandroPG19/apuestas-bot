"""Rate-limit coordinator central en Valkey (Gap 8 / A4).

Token-bucket distribuido keyed por host. Varios ingesters paralelos que
pegan al mismo host (p.ej. odds-api.com via OddsAPI + OddsJam por CDN)
comparten el bucket y evitan HTTP 429.

Uso:
    from apuestas.ingest.rate_limiter import acquire

    await acquire(host="gamma-api.polymarket.com", capacity=60, refill_per_sec=1.0)
    # ... hacer la request

Fail-safe: si Valkey no disponible, `acquire` devuelve inmediatamente
(degrada a sin-rate-limit; cada ingester sigue con su límite local).
"""

from __future__ import annotations

import asyncio
import time

from apuestas.cache import _get_client
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def acquire(
    *,
    host: str,
    capacity: int = 60,
    refill_per_sec: float = 1.0,
    max_wait_sec: float = 30.0,
) -> bool:
    """Consume 1 token del bucket del `host`. Espera si no hay tokens.

    Args:
        host: identificador del bucket (normalmente host del HTTP).
        capacity: tamaño máximo del bucket.
        refill_per_sec: tokens/segundo restaurados.
        max_wait_sec: tiempo máximo de espera antes de abortar (devuelve False).

    Returns:
        True si obtuvo token (posiblemente tras esperar); False si timeout.
    """
    client = await _get_client()
    if client is None:
        # Sin Valkey → no coordinamos. Callers mantienen sus propios límites.
        return True

    key = f"ratelim:{host}"
    start = time.monotonic()

    # Implementación simple: INCR + EXPIRE. Si count > capacity, sleep (1/refill).
    # No es un bucket real, pero es correcto como máximo de N req/s cross-proc.
    while True:
        try:
            # Ventana de 1s: count INCR, si primer INCR setea TTL 1s.
            count = await client.incr(key)
            if count == 1:
                await client.expire(key, 1)
            if count <= capacity:
                return True
        except Exception as exc:
            logger.debug("rate_limiter.err", host=host, error=str(exc)[:60])
            return True  # fail-open: no coordinar es mejor que bloquear

        if (time.monotonic() - start) > max_wait_sec:
            logger.warning("rate_limiter.timeout", host=host)
            return False
        await asyncio.sleep(1.0 / max(refill_per_sec, 0.1))


__all__ = ["acquire"]
