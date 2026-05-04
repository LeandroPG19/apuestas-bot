"""Bulk ingest clubelo.com — Elo ratings pre-calculados soccer.

Fuente: http://clubelo.com/API
Licencia: Free use with attribution.

Descarga Elo diario por equipo (Big-5 + más). Inserta en `team_elo_daily`
con source='clubelo'.

API:
    GET http://api.clubelo.com/{YYYY-MM-DD}     → CSV con ratings de ese día
    GET http://api.clubelo.com/{Team}           → histórico completo del equipo

Uso:
    uv run python scripts/ingest_clubelo.py --since 2018-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _fetch_day_ratings(client: httpx.AsyncClient, target_date: date) -> pd.DataFrame | None:
    url = f"http://api.clubelo.com/{target_date.isoformat()}"
    try:
        r = await client.get(url, timeout=20.0)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception as exc:
        logger.debug("clubelo.fetch_fail", date=str(target_date), error=str(exc)[:100])
        return None


async def _resolve_team_id(session, club_name: str, country: str) -> int | None:  # type: ignore[no-untyped-def]
    """Match clubelo team name → teams.id. Fuzzy por substring."""
    from sqlalchemy import text as _text

    try:
        r = await session.execute(
            _text(
                """
                SELECT t.id FROM teams t
                JOIN matches m ON (m.home_team_id=t.id OR m.away_team_id=t.id)
                WHERE m.sport_code='soccer'
                  AND LOWER(t.name) LIKE :pattern
                LIMIT 1
                """
            ),
            {"pattern": f"%{club_name.lower()[:10]}%"},
        )
        row = r.first()
        return int(row[0]) if row else None
    except Exception:
        return None


async def ingest_clubelo_range(since: date, until: date, *, step_days: int = 7) -> int:
    """Descarga ratings cada step_days entre since y until.

    step_days=7 es suficiente (Elo cambia slow). Reduce requests 7x.
    """
    from apuestas.db import session_scope

    total = 0
    current = since
    async with httpx.AsyncClient() as client:
        while current <= until:
            df = await _fetch_day_ratings(client, current)
            if df is None or len(df) == 0:
                current += timedelta(days=step_days)
                continue

            async with session_scope() as session:
                from sqlalchemy import text as _text

                for _, row in df.iterrows():
                    club = str(row.get("Club") or "").strip()
                    country = str(row.get("Country") or "").strip()
                    elo = row.get("Elo")
                    if not club or elo is None or pd.isna(elo):
                        continue
                    team_id = await _resolve_team_id(session, club, country)
                    if team_id is None:
                        continue
                    try:
                        await session.execute(
                            _text(
                                """
                                INSERT INTO team_elo_daily (
                                    team_id, sport_code, rating_date, source, elo_rating
                                ) VALUES (:tid, 'soccer', :d, 'clubelo', :elo)
                                ON CONFLICT (team_id, source, rating_date)
                                DO UPDATE SET elo_rating = EXCLUDED.elo_rating
                                """
                            ),
                            {"tid": team_id, "d": current, "elo": float(elo)},
                        )
                        total += 1
                    except Exception as exc:
                        logger.debug("clubelo.insert_fail", error=str(exc)[:80])
                await session.commit()
            logger.info("clubelo.day_ingested", date=str(current), n=len(df))
            current += timedelta(days=step_days)
            await asyncio.sleep(0.2)  # cortesía
    logger.info("clubelo.done", total=total)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=str, default="2018-01-01")
    parser.add_argument("--until", type=str, default=None)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()

    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC).date()
    until = (
        datetime.strptime(args.until, "%Y-%m-%d").replace(tzinfo=UTC).date()
        if args.until
        else datetime.now(tz=UTC).date()
    )
    n = asyncio.run(ingest_clubelo_range(since, until, step_days=args.step_days))
    print(f"✓ Inserted {n} clubelo ratings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
