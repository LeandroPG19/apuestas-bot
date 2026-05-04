"""Kalshi sports contracts ingester (Sprint 6b + Sprint B fix abr-2026).

Kalshi es un CLOB regulado por CFTC (USA). Ofrece contratos sobre eventos
deportivos NBA/NHL/NCAAB. Precios ∈ [0.00, 1.00] = prob implícita directa.

Free tier: read-only sin auth en el host nuevo (validado HTTP 200 desde MX
2026-04-25). El host viejo `trading-api.kalshi.com` devuelve 401.

API pública: https://api.elections.kalshi.com/trade-api/v2
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi usa `ticker` prefix por dominio: `KXNBA`, `KXNHL`, `KXNCAAB`, ...
SPORT_PREFIX = {
    "nba": "KXNBA",
    "nhl": "KXNHL",
    "ncaab": "KXNCAAB",
    "nfl": "KXNFL",
    "mlb": "KXMLB",
}


async def fetch_markets(
    client: httpx.AsyncClient, *, series_ticker: str, limit: int = 200
) -> list[dict[str, Any]]:
    """Trae markets abiertos para un series_ticker."""
    params = {"series_ticker": series_ticker, "status": "open", "limit": limit}
    try:
        r = await client.get(f"{BASE_URL}/markets", params=params, timeout=15.0)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("kalshi.fetch_fail", series=series_ticker, error=str(exc)[:120])
        return []
    data = r.json() or {}
    return list(data.get("markets", []) or [])


async def _persist(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    async with session_scope() as session:
        for row in rows:
            try:
                await session.execute(
                    text(
                        """
                        INSERT INTO kalshi_prices
                          (ticker, title, sport, yes_midpoint, no_midpoint,
                           volume, close_ts, captured_at)
                        VALUES
                          (:tk, :ti, :sp, :yes, :no, :vol, :close, :ts)
                        ON CONFLICT (ticker, captured_at) DO NOTHING
                        """
                    ),
                    row,
                )
            except Exception as exc:
                # La tabla puede no existir aún; log y continúa.
                logger.debug("kalshi.persist_fail", error=str(exc)[:120])
                return 0
        await session.commit()
    return len(rows)


async def ingest_kalshi_sport(sport: str) -> int:
    series = SPORT_PREFIX.get(sport.lower())
    if series is None:
        return 0
    now = datetime.now(tz=UTC)
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        markets = await fetch_markets(client, series_ticker=series)
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            yes_bid = m.get("yes_bid")
            yes_ask = m.get("yes_ask")
            if yes_bid is None or yes_ask is None:
                continue
            # Midpoint de yes = (bid+ask)/2 / 100 (Kalshi reporta en centavos).
            try:
                yes_mid = (Decimal(str(yes_bid)) + Decimal(str(yes_ask))) / Decimal("200")
                no_mid = Decimal("1") - yes_mid
            except Exception:
                continue
            rows.append(
                {
                    "tk": str(ticker)[:80],
                    "ti": str(m.get("title", ""))[:400],
                    "sp": sport.lower(),
                    "yes": yes_mid,
                    "no": no_mid,
                    "vol": Decimal(str(m.get("volume", 0) or 0)),
                    "close": m.get("close_time"),
                    "ts": now,
                }
            )
    n = await _persist(rows)
    logger.info("kalshi.ingested", sport=sport, rows=n)
    return n


async def run_kalshi_ingest() -> dict[str, int]:
    """Orquestador: itera SPORT_PREFIX y agrega counts."""
    out: dict[str, int] = {}
    for sport in SPORT_PREFIX:
        try:
            out[sport] = await ingest_kalshi_sport(sport)
        except Exception as exc:
            logger.warning("kalshi.sport_fail", sport=sport, error=str(exc)[:100])
            out[sport] = 0
    return out


__all__ = [
    "fetch_markets",
    "ingest_kalshi_sport",
    "run_kalshi_ingest",
]
