"""Polymarket ingest — mercados de predicción sobre eventos deportivos.

Polymarket tiene mercados futures: NBA MVP, Ballon d'Or, Super Bowl winner,
Premier League champion, etc. Los precios en Polymarket son "market-driven"
(no bookmaker margin), útiles como **benchmark de fair value** para
comparar con sportsbooks.

API gratis: https://gamma-api.polymarket.com/markets
- filter por tag "sports"
- precios en USDC (0-1 representa probabilidad implícita)
- volume_24h valida liquidez
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


BASE_URL = "https://gamma-api.polymarket.com"


SPORT_TAG_MAP: dict[str, list[str]] = {
    "nba": ["nba", "basketball"],
    "nfl": ["nfl", "football", "super-bowl"],
    "mlb": ["mlb", "baseball", "world-series"],
    "soccer": ["soccer", "world-cup", "champions-league", "premier-league"],
    "boxing": ["boxing"],
    "tennis": ["tennis", "grand-slam"],
    "nhl": ["nhl", "hockey"],
}


async def fetch_polymarket_active(sport_code: str | None = None) -> list[dict[str, Any]]:
    """Trae mercados activos. Si sport_code, filtra por tags sport."""
    tags = SPORT_TAG_MAP.get(sport_code or "", [])
    params: dict[str, Any] = {"active": "true", "closed": "false", "limit": "200"}
    if tags:
        params["tag_slug"] = tags[0]

    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "apuestas-bot/0.1", "Accept": "application/json"},
    ) as c:
        try:
            r = await c.get(f"{BASE_URL}/markets", params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning("polymarket.fetch_fail", error=str(exc))
            return []

    markets = data if isinstance(data, list) else data.get("data", [])
    logger.info("polymarket.fetched", sport=sport_code, count=len(markets))
    return markets


async def persist_markets(markets: list[dict[str, Any]], sport_code: str) -> int:
    """Persiste markets con outcomes + current_prices."""
    if not markets:
        return 0
    inserted = 0
    async with session_scope() as s:
        for m in markets:
            import json as _json

            cond_id = m.get("conditionId") or m.get("id")
            if not cond_id:
                continue
            question = m.get("question", "")[:500]
            event_type = _infer_event_type(question)
            end_str = m.get("endDate") or m.get("end_date_iso")
            end_dt = _parse_ts(end_str)

            outcomes_list = m.get("outcomes", [])
            prices_list = m.get("outcomePrices", m.get("clobTokenIds", []))
            outcomes_json = _json.dumps(outcomes_list if isinstance(outcomes_list, list) else [])
            current_json = _json.dumps(
                dict(zip(outcomes_list, prices_list, strict=False))
                if isinstance(outcomes_list, list) and isinstance(prices_list, list)
                else {}
            )
            volume = float(m.get("volume24hr", m.get("volume", 0)) or 0)

            try:
                await s.execute(
                    text(
                        """
                        INSERT INTO polymarket_markets
                            (condition_id, question, sport_code, event_type,
                             end_date, outcomes, current_prices, volume_24h_usd,
                             last_updated)
                        VALUES (:c, :q, :s, :et, :ed, CAST(:o AS jsonb),
                                CAST(:p AS jsonb), :v, NOW())
                        ON CONFLICT (condition_id) DO UPDATE SET
                            current_prices = EXCLUDED.current_prices,
                            volume_24h_usd = EXCLUDED.volume_24h_usd,
                            last_updated = NOW()
                        """
                    ),
                    {
                        "c": str(cond_id)[:100],
                        "q": question,
                        "s": sport_code,
                        "et": event_type,
                        "ed": end_dt,
                        "o": outcomes_json,
                        "p": current_json,
                        "v": volume,
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug("polymarket.persist_fail", error=str(exc))
    return inserted


def _infer_event_type(question: str) -> str:
    low = question.lower()
    if "mvp" in low:
        return "mvp"
    if "ballon" in low:
        return "ballon_dor"
    if "champion" in low or "win the" in low:
        return "champion"
    if "cy young" in low:
        return "cy_young"
    if "rookie" in low:
        return "roy"
    return "other"


def _parse_ts(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


async def run_ingest() -> dict[str, int]:
    """Orquestador: trae mercados para todos los sports soportados."""
    results: dict[str, int] = {}
    for sport in ("nba", "nfl", "mlb", "soccer", "boxing", "tennis", "nhl"):
        markets = await fetch_polymarket_active(sport)
        results[sport] = await persist_markets(markets, sport)
    return results
