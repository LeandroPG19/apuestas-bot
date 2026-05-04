"""Bulk ingest MoneyPuck NHL — shots + xG + Corsi/Fenwick.

Fuente: https://moneypuck.com/data.htm
Licencia: Free with attribution.

Endpoints CSV:
- /moneypuck/playerData/careers/gameByGame/all_teams.csv (games team-level stats)
- /moneypuck/playerData/seasonSummary/{YEAR}/regular/teams.csv (season summary)
- /moneypuck/playerData/seasonSummary/{YEAR}/regular/skaters.csv (players)

Volumen: ~18k partidos × team + shot-level stats.

Uso:
    uv run python scripts/ingest_moneypuck_nhl.py
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

GAME_BY_GAME_URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"


async def _ensure_table() -> None:
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    async with session_scope() as s:
        await s.execute(
            _text(
                """
                CREATE TABLE IF NOT EXISTS moneypuck_team_games (
                    id bigserial PRIMARY KEY,
                    team text NOT NULL,
                    game_id text NOT NULL,
                    game_date date,
                    situation text,  -- 'all', '5on5', 'other'
                    opposing_team text,
                    home_or_away text,
                    shots_for integer,
                    shots_against integer,
                    xg_for numeric(6,3),
                    xg_against numeric(6,3),
                    corsi_for integer,
                    corsi_against integer,
                    fenwick_for integer,
                    fenwick_against integer,
                    goals_for integer,
                    goals_against integer,
                    ingested_at timestamptz DEFAULT now(),
                    UNIQUE (team, game_id, situation)
                )
                """
            )
        )
        await s.execute(
            _text(
                "CREATE INDEX IF NOT EXISTS idx_mp_nhl_team_date "
                "ON moneypuck_team_games (team, game_date DESC)"
            )
        )
        await s.commit()


async def ingest_all() -> int:
    """Descarga el CSV gigante y hace COPY bulk."""
    await _ensure_table()
    async with httpx.AsyncClient() as client:
        r = await client.get(GAME_BY_GAME_URL, timeout=180.0, follow_redirects=True)
        if r.status_code != 200:
            logger.error("moneypuck.fetch_fail", status=r.status_code)
            return 0
        df = pd.read_csv(io.BytesIO(r.content), on_bad_lines="skip", low_memory=False)

    logger.info("moneypuck.fetched", rows=len(df), cols=len(df.columns))

    # Columnas relevantes
    cols_want = [
        "team",
        "gameId",
        "gameDate",
        "situation",
        "opposingTeam",
        "home_or_away",
        "shotsOnGoalFor",
        "shotsOnGoalAgainst",
        "xGoalsFor",
        "xGoalsAgainst",
        "corsiForAfterShifts",
        "corsiAgainstAfterShifts",
        "fenwickForAfterShifts",
        "fenwickAgainstAfterShifts",
        "goalsFor",
        "goalsAgainst",
    ]
    existing = [c for c in cols_want if c in df.columns]
    df = df[existing].copy()

    if "gameDate" in df.columns:
        df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce").dt.date

    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    inserted = 0
    chunk_size = 3000
    async with session_scope() as session:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            rows = []
            for _, r in chunk.iterrows():
                rows.append(
                    {
                        "team": str(r.get("team") or ""),
                        "gid": str(r.get("gameId") or ""),
                        "gd": r.get("gameDate"),
                        "sit": str(r.get("situation") or "all")[:16],
                        "opp": str(r.get("opposingTeam") or ""),
                        "hoa": str(r.get("home_or_away") or ""),
                        "sf": _to_int(r.get("shotsOnGoalFor")),
                        "sa": _to_int(r.get("shotsOnGoalAgainst")),
                        "xgf": _to_float(r.get("xGoalsFor")),
                        "xga": _to_float(r.get("xGoalsAgainst")),
                        "cf": _to_int(r.get("corsiForAfterShifts")),
                        "ca": _to_int(r.get("corsiAgainstAfterShifts")),
                        "ff": _to_int(r.get("fenwickForAfterShifts")),
                        "fa": _to_int(r.get("fenwickAgainstAfterShifts")),
                        "gf": _to_int(r.get("goalsFor")),
                        "ga": _to_int(r.get("goalsAgainst")),
                    }
                )
            if not rows:
                continue
            try:
                await session.execute(
                    _text(
                        """
                        INSERT INTO moneypuck_team_games (
                            team, game_id, game_date, situation, opposing_team,
                            home_or_away, shots_for, shots_against,
                            xg_for, xg_against, corsi_for, corsi_against,
                            fenwick_for, fenwick_against, goals_for, goals_against
                        ) VALUES (
                            :team, :gid, :gd, :sit, :opp, :hoa,
                            :sf, :sa, :xgf, :xga, :cf, :ca, :ff, :fa, :gf, :ga
                        )
                        ON CONFLICT (team, game_id, situation) DO NOTHING
                        """
                    ),
                    rows,
                )
                inserted += len(rows)
            except Exception as exc:
                logger.warning("moneypuck.chunk_fail", start=start, error=str(exc)[:100])
        await session.commit()
    logger.info("moneypuck.done", inserted=inserted)
    return inserted


def _to_int(v):  # type: ignore[no-untyped-def]
    try:
        import math

        f = float(v)
        if math.isnan(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def _to_float(v):  # type: ignore[no-untyped-def]
    try:
        import math

        f = float(v)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def main() -> int:
    n = asyncio.run(ingest_all())
    print(f"✓ Inserted {n} MoneyPuck rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
