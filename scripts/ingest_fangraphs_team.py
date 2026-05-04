"""Bulk ingest FanGraphs team stats via pybaseball.

Fuente: FanGraphs vía pybaseball.team_batting + team_pitching.
Licencia: free (pybaseball MIT).

Guarda en `fangraphs_team_stats_daily` con wRC+, FIP, xFIP, WAR rolling
por team × season.

Uso:
    uv run python scripts/ingest_fangraphs_team.py --since 2018
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def _fetch_season(year: int):  # type: ignore[no-untyped-def]
    """Blocking wrapper para pybaseball."""
    import pybaseball as pb

    batting = pb.team_batting(year)
    pitching = pb.team_pitching(year)
    return batting, pitching


async def _resolve_team_id(session, team_abbr: str) -> int | None:  # type: ignore[no-untyped-def]
    """Resolve FanGraphs abbr → teams.id en nuestra DB."""
    from sqlalchemy import text as _text

    # FanGraphs usa: BOS, NYY, LAD, etc. Similar a short_name de teams.
    try:
        r = await session.execute(
            _text(
                "SELECT t.id FROM teams t "
                "JOIN matches m ON (m.home_team_id=t.id OR m.away_team_id=t.id) "
                "WHERE m.sport_code='mlb' AND UPPER(t.short_name)=:a LIMIT 1"
            ),
            {"a": team_abbr.upper()},
        )
        row = r.first()
        if row:
            return int(row[0])
        # Fallback: match fuzzy por nombre
        r2 = await session.execute(
            _text(
                "SELECT t.id FROM teams t "
                "JOIN matches m ON (m.home_team_id=t.id OR m.away_team_id=t.id) "
                "WHERE m.sport_code='mlb' AND UPPER(t.name) LIKE :p LIMIT 1"
            ),
            {"p": f"%{team_abbr.upper()}%"},
        )
        row2 = r2.first()
        return int(row2[0]) if row2 else None
    except Exception:
        return None


async def ingest_season(year: int) -> int:
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    loop = asyncio.get_event_loop()
    try:
        batting, pitching = await loop.run_in_executor(None, _fetch_season, year)
    except Exception as exc:
        logger.warning("fangraphs.fetch_fail", year=year, error=str(exc)[:100])
        return 0

    # Merge batting + pitching por Team
    batting_cols = {c for c in ("Team", "wRC+", "BsR", "WAR") if c in batting.columns}
    pitching_cols = {c for c in ("Team", "FIP", "xFIP", "WAR") if c in pitching.columns}
    if "Team" not in batting_cols or "Team" not in pitching_cols:
        return 0

    stat_date = date(year, 10, 30)  # último día aproximado temporada
    total = 0
    async with session_scope() as session:
        for _, row in batting.iterrows():
            team_abbr = str(row.get("Team") or "")
            if not team_abbr or team_abbr.lower() == "nan":
                continue
            team_id = await _resolve_team_id(session, team_abbr)
            if team_id is None:
                continue
            wrc_plus = row.get("wRC+")
            bsr = row.get("BsR")
            war_bat = row.get("WAR")
            # Buscar pitching row matching
            pit_row = pitching[pitching["Team"] == team_abbr]
            fip = (
                float(pit_row["FIP"].iloc[0])
                if len(pit_row) > 0 and "FIP" in pit_row.columns
                else None
            )
            xfip = (
                float(pit_row["xFIP"].iloc[0])
                if len(pit_row) > 0 and "xFIP" in pit_row.columns
                else None
            )
            war_pit = (
                float(pit_row["WAR"].iloc[0])
                if len(pit_row) > 0 and "WAR" in pit_row.columns
                else None
            )
            war_total = (float(war_bat) if war_bat else 0) + (war_pit or 0)

            try:
                await session.execute(
                    _text(
                        """
                        INSERT INTO fangraphs_team_stats_daily (
                            team_id, stat_date, wrc_plus, fip, xfip,
                            war_rolling_30, bsr_rolling_30
                        ) VALUES (:tid, :d, :wrc, :fip, :xfip, :war, :bsr)
                        ON CONFLICT (team_id, stat_date) DO UPDATE SET
                            wrc_plus = EXCLUDED.wrc_plus,
                            fip = EXCLUDED.fip,
                            xfip = EXCLUDED.xfip,
                            war_rolling_30 = EXCLUDED.war_rolling_30,
                            bsr_rolling_30 = EXCLUDED.bsr_rolling_30
                        """
                    ),
                    {
                        "tid": team_id,
                        "d": stat_date,
                        "wrc": float(wrc_plus) if wrc_plus else None,
                        "fip": fip,
                        "xfip": xfip,
                        "war": float(war_total),
                        "bsr": float(bsr) if bsr else None,
                    },
                )
                total += 1
            except Exception as exc:
                logger.debug("fangraphs.insert_fail", error=str(exc)[:80])
        await session.commit()
    logger.info("fangraphs.season_done", year=year, teams=total)
    return total


async def main_async(since: int) -> int:
    total = 0
    end = datetime.now(tz=UTC).year
    for year in range(since, end + 1):
        try:
            n = await ingest_season(year)
            total += n
        except Exception as exc:
            logger.warning("fangraphs.year_fail", year=year, error=str(exc)[:100])
    logger.info("fangraphs.done", total=total)
    print(f"✓ Inserted {total} FanGraphs team-season rows")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2018)
    args = parser.parse_args()
    return asyncio.run(main_async(args.since))


if __name__ == "__main__":
    raise SystemExit(main())
