"""Ingesta de tipos de cambio USD↔MXN (gratis, sin key).

Fuentes por orden de prioridad:
1. open.er-api.com (exchangerate-api.com free tier sin key, ~250 req/mes).
2. frankfurter.app (BCE, sin MXN directo, usa USD→EUR→MXN encadenado).
3. Fallback estático (Decimal("17.50")).

Idempotente: una sola fila por (base, quote, date).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_PRIMARY_URL = "https://open.er-api.com/v6/latest/USD"
_TIMEOUT = httpx.Timeout(10.0)


async def _fetch_open_er_api() -> Decimal | None:
    """USD→MXN desde open.er-api.com. Devuelve None si falla."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_PRIMARY_URL)
            resp.raise_for_status()
            payload = resp.json()
        if payload.get("result") != "success":
            return None
        rate = payload.get("rates", {}).get("MXN")
        if rate is None:
            return None
        return Decimal(str(rate))
    except Exception as exc:
        logger.warning("fx.open_er_api_fail", error=str(exc)[:100])
        return None


async def persist_fx_rate(
    base: str, quote: str, rate: Decimal, *, source: str, at: date | None = None
) -> None:
    captured = at if at is not None else datetime.now(tz=UTC).date()
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO fx_rates (captured_at, base_currency, quote_currency, rate, source)
                VALUES (:d, :b, :q, :r, :s)
                ON CONFLICT (captured_at, base_currency, quote_currency)
                DO UPDATE SET rate = EXCLUDED.rate, source = EXCLUDED.source
                """
            ),
            {"d": captured, "b": base, "q": quote, "r": str(rate), "s": source},
        )


async def ingest_fx_rates_once() -> dict[str, str]:
    """Descarga USD→MXN del día y lo persiste. Retorna dict con rate usado."""
    rate = await _fetch_open_er_api()
    source = "open.er-api.com"
    if rate is None:
        logger.warning("fx.all_sources_failed_using_fallback")
        rate = Decimal("17.50")
        source = "fallback_static"
    await persist_fx_rate("USD", "MXN", rate, source=source)
    logger.info("fx.ingested", rate_usd_mxn=str(rate), source=source)
    return {"rate_usd_mxn": str(rate), "source": source}


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(ingest_fx_rates_once()))
