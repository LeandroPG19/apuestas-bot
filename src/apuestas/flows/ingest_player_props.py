"""Flow ingest_player_props — explotación paid tier The Odds API.

Estrategia de costo (paid $30/mes = 20k créditos, 667/día):
- T-30min del commence: 1 snap con 3 markets base × 1 region = 3 créditos/evento.
- Solo eventos en ventana [T-2h, T+5min] (pre-game).
- Filtro sharp books: pinnacle + circasports + betonlineag + draftkings + fanduel.
- Persistencia en JSONB stage (`odds_api_event_snapshots`). ETL downstream
  resuelve player_id async y normaliza a `player_prop_lines`.

Presupuesto diario estimado:
- NBA ~10 partidos × 3 mercados × 1 snap = 30 cr
- NFL ~1 partido (Sunday/Thu/Mon) × 4 mercados × 1 snap = 4 cr
- MLB ~15 partidos × 3 mercados = 45 cr
- NHL ~6 partidos × 3 mercados = 18 cr
- TOTAL: ~100 cr/día → 3,000 cr/mes (~15% del budget).
"""

from __future__ import annotations

import asyncio
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.odds_api import OddsAPIClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Mapping sport interno → (odds api sport key, player prop markets csv).
# Markets elegidos: mayor liquidez + mejor calibrado.
_PROP_MARKETS: dict[str, tuple[str, str]] = {
    "nba": (
        "basketball_nba",
        "player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists",
    ),
    "nfl": (
        "americanfootball_nfl",
        "player_pass_yds,player_rush_yds,player_receptions,player_anytime_td",
    ),
    "mlb": (
        "baseball_mlb",
        "batter_home_runs,batter_hits,batter_total_bases,pitcher_strikeouts",
    ),
    "nhl": (
        "icehockey_nhl",
        "player_points,player_goals,player_shots_on_goal",
    ),
}


async def _pending_prop_fetch() -> list[dict[str, Any]]:
    """Matches en ventana [T-2h, T+5min] sin snapshot reciente en esta ventana.

    Filtra por sport_focus en runtime: nhl/nfl/etc. quedan fuera si están off,
    evitando ~50-200 créditos/día en player props de sports desactivados.
    """
    from apuestas.betting.sport_focus import is_emit_enabled

    enabled_sports = [s for s in ("nba", "nfl", "mlb", "nhl") if is_emit_enabled(s)]
    if not enabled_sports:
        return []

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.sport_code, m.external_id_odds_api, m.start_time
                FROM matches m
                WHERE m.status != 'finished'
                  AND m.start_time BETWEEN NOW() + INTERVAL '5 minutes' AND NOW() + INTERVAL '2 hours'
                  AND m.sport_code = ANY(:sports)
                  AND m.external_id_odds_api IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM odds_api_event_snapshots s
                      WHERE s.event_id = m.external_id_odds_api
                        AND s.captured_at > NOW() - INTERVAL '30 minutes'
                  )
                ORDER BY m.start_time ASC
                LIMIT 50
                """
            ),
            {"sports": enabled_sports},
        )
        return [dict(r._mapping) for r in result.all()]


async def _persist_snapshot(
    *,
    sport_key: str,
    event_id: str,
    internal_match_id: int,
    markets: str,
    payload: dict[str, Any],
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO odds_api_event_snapshots
                    (sport_key, event_id, internal_match_id, markets, payload)
                VALUES (:sk, :eid, :mid, :mk, CAST(:payload AS JSONB))
                """
            ),
            {
                "sk": sport_key,
                "eid": event_id,
                "mid": internal_match_id,
                "mk": markets,
                "payload": _jsonable(payload),
            },
        )


def _jsonable(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)


@task(retries=2, retry_delay_seconds=30)
async def fetch_props_for_match(
    *,
    match_id: int,
    sport_code: str,
    event_id: str,
) -> int:
    """Fetch event-level player props. Retorna créditos consumidos."""
    if sport_code not in _PROP_MARKETS:
        return 0
    sport_key, markets = _PROP_MARKETS[sport_code]

    try:
        client = OddsAPIClient()
    except ValueError:
        logger.info("ingest_props.no_key")
        return 0

    async with client.session():
        try:
            data = await client.fetch_event_odds(
                sport_key,
                event_id,
                markets=markets,
                regions="us",
                bookmakers="pinnacle,betonlineag,circasports,draftkings,fanduel",
                include_sids=True,
            )
        except Exception as exc:
            logger.warning(
                "ingest_props.fetch_fail",
                match_id=match_id,
                event_id=event_id,
                error=str(exc)[:140],
            )
            return 0

        credits_before = (client._last_remaining or 0) + (client._last_request_cost or 0)
        cost = client._last_request_cost or 0
        _ = credits_before  # side-channel para logging

    try:
        await _persist_snapshot(
            sport_key=sport_key,
            event_id=event_id,
            internal_match_id=match_id,
            markets=markets,
            payload=data,
        )
    except Exception as exc:
        logger.warning("ingest_props.persist_fail", match_id=match_id, error=str(exc)[:140])
        return cost

    logger.info(
        "ingest_props.captured",
        match_id=match_id,
        event_id=event_id,
        sport=sport_code,
        n_books=len(data.get("bookmakers", [])),
        cost=cost,
    )
    return cost


@flow(name="apuestas-ingest-player-props", log_prints=True)
async def ingest_player_props_flow(*, max_credits: int = 150) -> dict[str, Any]:
    """Fetch player props para matches en ventana T-2h → T+5min.

    Budget guard: detiene si el costo acumulado supera `max_credits`.
    Default 150 créditos/run permite ~50 events con 3-4 markets.
    """
    matches = await _pending_prop_fetch()
    logger.info("ingest_props.start", candidates=len(matches))
    if not matches:
        return {"candidates": 0, "captured": 0, "credits_spent": 0}

    captured = 0
    spent = 0
    for m in matches:
        if spent >= max_credits:
            logger.warning("ingest_props.budget_exhausted", spent=spent, max=max_credits)
            break
        cost = await fetch_props_for_match.fn(
            match_id=int(m["id"]),
            sport_code=str(m["sport_code"]),
            event_id=str(m["external_id_odds_api"]),
        )
        spent += cost
        if cost > 0:
            captured += 1

    logger.info(
        "ingest_props.done", candidates=len(matches), captured=captured, credits_spent=spent
    )
    return {"candidates": len(matches), "captured": captured, "credits_spent": spent}


if __name__ == "__main__":
    asyncio.run(ingest_player_props_flow())
