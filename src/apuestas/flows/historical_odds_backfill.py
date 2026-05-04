"""Flow historical_odds_backfill — paid-tier bulk histórico para backtest + CLV real.

Estrategia:
- Historical odds endpoint: 10× markets × regions por snapshot.
- Snapshot T-5min ANTES del commence = closing line → CLV real contra Pinnacle.
- Backfill controlado por budget (default 500 créditos/run = ~50 snapshots).

Uso:
    # Manual: backfill una fecha específica
    python -m apuestas.flows.historical_odds_backfill --date 2026-04-22 --sport basketball_nba

    # Via Prefect schedule: backfill previous day closing lines cada noche
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.odds_api import OddsAPIClient
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


async def _persist_historical(
    *,
    sport_key: str,
    snapshot_ts: datetime,
    markets: str,
    regions: str,
    payload: dict[str, Any],
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO odds_api_historical_snapshots
                    (sport_key, snapshot_ts, markets, regions, payload)
                VALUES (:sk, :ts, :mk, :rg, CAST(:payload AS JSONB))
                ON CONFLICT (sport_key, snapshot_ts, markets, regions) DO NOTHING
                """
            ),
            {
                "sk": sport_key,
                "ts": snapshot_ts,
                "mk": markets,
                "rg": regions,
                "payload": json.dumps(payload, default=str),
            },
        )


@task(retries=2, retry_delay_seconds=30)
async def fetch_historical_snapshot(
    sport_key: str,
    timestamp: datetime,
    *,
    markets: str = "h2h,spreads,totals",
    regions: str = "us",
    bookmakers: str = "pinnacle",
) -> int:
    """Fetch y persiste historical odds snapshot. Retorna créditos gastados."""
    try:
        client = OddsAPIClient()
    except ValueError:
        return 0

    async with client.session():
        try:
            data = await client.fetch_historical_odds(
                sport_key,
                timestamp=timestamp,
                regions=regions,
                markets=markets,
                bookmakers=bookmakers,
            )
        except Exception as exc:
            logger.warning(
                "historical_backfill.fetch_fail",
                sport=sport_key,
                ts=timestamp.isoformat(),
                error=str(exc)[:140],
            )
            return 0
        cost = client._last_request_cost or 0
        api_ts_raw = data.get("timestamp")

    # Parse API timestamp (ISO 8601)
    try:
        api_ts = datetime.fromisoformat(str(api_ts_raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        api_ts = timestamp

    try:
        await _persist_historical(
            sport_key=sport_key,
            snapshot_ts=api_ts,
            markets=markets,
            regions=regions,
            payload=data,
        )
    except Exception as exc:
        logger.warning(
            "historical_backfill.persist_fail",
            sport=sport_key,
            error=str(exc)[:140],
        )

    logger.info(
        "historical_backfill.captured",
        sport=sport_key,
        snapshot_ts=api_ts.isoformat(),
        cost=cost,
        n_events=len(data.get("data", [])),
    )
    return cost


async def _closing_line_targets(hours_back: int = 30) -> list[dict[str, Any]]:
    """Matches finalizados en últimas N horas → necesitan closing line snapshot.

    No requerimos `external_id_odds_api` porque el historical endpoint devuelve
    TODOS los eventos del sport a ese timestamp — el matching con matches
    internos se hace downstream en el parser JSONB.
    """
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT sport_code, external_id_odds_api, start_time
                FROM matches
                WHERE status = 'finished'
                  AND start_time > NOW() - INTERVAL ':hours hours'
                  AND start_time < NOW() - INTERVAL '15 minutes'
                  AND sport_code IN ('nba','nfl','mlb','nhl','soccer','epl','laliga','bundesliga','seriea','ligue1','liga_mx')
                ORDER BY start_time DESC
                LIMIT 200
                """.replace(":hours", str(hours_back))
            )
        )
        return [dict(row._mapping) for row in r.all()]


@flow(name="apuestas-historical-backfill", log_prints=True)
async def historical_backfill_flow(
    *,
    hours_back: int = 30,
    markets: str = "h2h,spreads,totals",
    max_credits: int = 500,
) -> dict[str, Any]:
    """Backfill de closing lines (T-5min antes del commence) para CLV real.

    Un snapshot cubre TODOS los eventos en el sport a ese timestamp,
    así que agrupamos por (sport_key, rounded_ts) para deduplicar requests.
    """
    targets = await _closing_line_targets(hours_back=hours_back)
    if not targets:
        return {"targets": 0, "snapshots": 0, "credits_spent": 0}

    # Agrupar por (sport_key, rounded_ts_to_5min) — un snapshot cubre el sport
    sport_map = {
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "mlb": "baseball_mlb",
        "nhl": "icehockey_nhl",
        "soccer": "soccer_epl",  # top-5 EU default, Liga MX/MLS requieren run dedicado
        "epl": "soccer_epl",
        "laliga": "soccer_spain_la_liga",
        "bundesliga": "soccer_germany_bundesliga",
        "seriea": "soccer_italy_serie_a",
        "ligue1": "soccer_france_ligue_one",
        "liga_mx": "soccer_mexico_ligamx",
    }
    snapshots_to_fetch: set[tuple[str, datetime]] = set()
    for t in targets:
        sport_key = sport_map.get(str(t["sport_code"]))
        if not sport_key:
            continue
        # Closing line ≈ T-5min antes de start_time
        closing_ts = t["start_time"] - timedelta(minutes=5)
        # Round a 5min bucket
        minute = (closing_ts.minute // 5) * 5
        closing_ts = closing_ts.replace(minute=minute, second=0, microsecond=0)
        snapshots_to_fetch.add((sport_key, closing_ts))

    spent = 0
    captured = 0
    for sport_key, ts in sorted(snapshots_to_fetch, key=lambda x: x[1]):
        if spent >= max_credits:
            logger.warning("historical_backfill.budget_hit", spent=spent, max=max_credits)
            break
        cost = await fetch_historical_snapshot.fn(
            sport_key, ts, markets=markets, regions="us", bookmakers="pinnacle"
        )
        spent += cost
        if cost > 0:
            captured += 1

    logger.info(
        "historical_backfill.done",
        targets=len(targets),
        snapshots=captured,
        credits_spent=spent,
    )
    return {
        "targets": len(targets),
        "snapshots": captured,
        "credits_spent": spent,
    }


def _cli() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="Historical odds backfill")
    p.add_argument("--hours-back", type=int, default=30)
    p.add_argument("--markets", type=str, default="h2h,spreads,totals")
    p.add_argument("--max-credits", type=int, default=500)
    args = p.parse_args()
    res = asyncio.run(
        historical_backfill_flow(
            hours_back=args.hours_back,
            markets=args.markets,
            max_credits=args.max_credits,
        )
    )
    print(res)


if __name__ == "__main__":
    _cli()
