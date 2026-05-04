"""Seed NFL player_game_logs desde nflreadpy.

Descarga weekly player stats (passing/rushing/receiving) y los guarda como
`stats` JSONB en player_game_logs. Desbloquea detect_all_player_props_for_match
para NFL props (passing_yards, rushing_yards, receiving_yards, receptions,
touchdowns).

Uso:
    apuestas seed-player-logs-nfl --seasons 2022,2023,2024

Idempotente via UNIQUE (player_id, match_id).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _resolve_player_id(
    session: Any, *, full_name: str, gsis_id: str | None = None
) -> int | None:
    """Busca NFL player. Crea con gsis_id como external_id si no existe."""
    if gsis_id:
        r = await session.execute(
            text("SELECT id FROM players WHERE external_id = :eid LIMIT 1"),
            {"eid": f"nfl:gsis:{gsis_id}"},
        )
        row = r.first()
        if row:
            return int(row.id)

    r = await session.execute(
        text("SELECT id FROM players WHERE sport_code = 'nfl' AND full_name ILIKE :n LIMIT 1"),
        {"n": full_name.strip()},
    )
    row = r.first()
    if row:
        return int(row.id)

    ext = f"nfl:gsis:{gsis_id}" if gsis_id else f"nfl:name:{full_name.lower()}"
    r2 = await session.execute(
        text(
            """
            INSERT INTO players (external_id, sport_code, full_name, created_at)
            VALUES (:ext, 'nfl', :name, NOW())
            ON CONFLICT (external_id) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING id
            """
        ),
        {"ext": ext, "name": full_name},
    )
    pid = r2.first()
    return int(pid.id) if pid else None


async def _resolve_match_id(session: Any, *, season: str, week: int, team_abbr: str) -> int | None:
    """Match lookup por season + week + equipo (nflreadpy usa abbrev)."""
    r = await session.execute(
        text(
            """
            SELECT m.id FROM matches m
            JOIN teams t ON (t.id = m.home_team_id OR t.id = m.away_team_id)
            WHERE m.sport_code = 'nfl'
              AND m.season = :ss
              AND t.external_id = :team_ext
            ORDER BY m.start_time ASC
            OFFSET :wk LIMIT 1
            """
        ),
        {"ss": season, "team_ext": f"nfl:{team_abbr.lower()}", "wk": max(0, week - 1)},
    )
    row = r.first()
    return int(row.id) if row else None


async def seed_nfl_season(season: int) -> dict[str, int]:
    """Descarga nflreadpy player stats y persiste."""
    import nflreadpy as nfl  # type: ignore[import-untyped]

    def _fetch() -> pl.DataFrame:
        df = nfl.load_player_stats(seasons=[season])
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
        return df

    df = await asyncio.to_thread(_fetch)
    logger.info("seed_player_logs.nfl.fetched", season=season, rows=df.height)

    season_str = f"{season}-{str(season + 1)[-2:]}"

    inserted = 0
    skipped = 0
    async with session_scope() as session:
        for row in df.iter_rows(named=True):
            try:
                full_name = str(row.get("player_display_name") or row.get("player_name") or "")
                gsis = row.get("player_id")  # nflreadpy gsis format
                if not full_name:
                    skipped += 1
                    continue
                player_id = await _resolve_player_id(session, full_name=full_name, gsis_id=gsis)
                if player_id is None:
                    skipped += 1
                    continue

                team_abbr = str(row.get("team") or row.get("recent_team") or "")
                week = int(row.get("week") or 0)
                if not team_abbr or week <= 0:
                    skipped += 1
                    continue

                match_id = await _resolve_match_id(
                    session, season=season_str, week=week, team_abbr=team_abbr
                )
                if match_id is None:
                    skipped += 1
                    continue

                stats = {
                    "passing_yards": int(row.get("passing_yards") or 0),
                    "passing_tds": int(row.get("passing_tds") or 0),
                    "completions": int(row.get("completions") or 0),
                    "attempts": int(row.get("attempts") or 0),
                    "interceptions": int(row.get("interceptions") or 0),
                    "rushing_yards": int(row.get("rushing_yards") or 0),
                    "rushing_tds": int(row.get("rushing_tds") or 0),
                    "carries": int(row.get("carries") or 0),
                    "receiving_yards": int(row.get("receiving_yards") or 0),
                    "receptions": int(row.get("receptions") or 0),
                    "receiving_tds": int(row.get("receiving_tds") or 0),
                    "targets": int(row.get("targets") or 0),
                }

                await session.execute(
                    text(
                        """
                        INSERT INTO player_game_logs
                            (player_id, match_id, sport_code, stats)
                        VALUES (:pid, :mid, 'nfl', CAST(:stats AS jsonb))
                        ON CONFLICT (player_id, match_id) DO UPDATE SET
                            stats = EXCLUDED.stats
                        """
                    ),
                    {
                        "pid": player_id,
                        "mid": match_id,
                        "stats": json.dumps(stats),
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug(
                    "seed_player_logs.nfl_row_fail",
                    row=str(row)[:100],
                    error=str(exc)[:80],
                )
                skipped += 1
    return {"inserted": inserted, "skipped": skipped, "total": df.height}


async def main(args: argparse.Namespace) -> None:
    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    for season in seasons:
        result = await seed_nfl_season(season)
        print(f"✅ NFL {season}: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2023,2024", help="CSV seasons (int)")
    asyncio.run(main(parser.parse_args()))
