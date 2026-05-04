"""Seed NBA player_game_logs desde nba_api.

Descarga logs por jugador × juego usando `playergamelogs` endpoint.
Poblar esta tabla desbloquea:
  - detect_all_player_props_for_match (historial real por jugador/stat)
  - compute_player_stat_std (std real vs fallback hardcoded)
  - FASE M.2 zero-hardcoded en sport_props_models.

Uso:
    apuestas seed-player-logs --sport nba --seasons 2023-24,2024-25

Idempotente via UNIQUE (player_id, match_id).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _resolve_player_id(
    session: Any, *, full_name: str, sport_code: str = "nba"
) -> int | None:
    """Busca player por nombre con fuzzy ILIKE; crea si no existe."""
    r = await session.execute(
        text("SELECT id FROM players WHERE sport_code = :sc AND full_name ILIKE :n LIMIT 1"),
        {"sc": sport_code, "n": full_name.strip()},
    )
    row = r.first()
    if row:
        return int(row.id)
    r2 = await session.execute(
        text(
            """
            INSERT INTO players (external_id, sport_code, full_name, created_at)
            VALUES (:ext, :sc, :name, NOW())
            ON CONFLICT (external_id) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING id
            """
        ),
        {
            "ext": f"nba_api:nba:{full_name.lower()}",
            "sc": sport_code,
            "name": full_name,
        },
    )
    pid = r2.first()
    return int(pid.id) if pid else None


async def _resolve_match_id(session: Any, *, game_date: str, team_id_nba_api: int) -> int | None:
    """Resuelve match_id por fecha + team_id (matches externos formato nba_api:nba:X-Y:date)."""
    r = await session.execute(
        text(
            """
            SELECT id FROM matches
            WHERE sport_code = 'nba'
              AND (home_team_id = :tid OR away_team_id = :tid)
              AND DATE(start_time) = DATE(:gd)
            LIMIT 1
            """
        ),
        {"tid": team_id_nba_api, "gd": game_date},
    )
    row = r.first()
    return int(row.id) if row else None


async def seed_nba_season(season: str) -> dict[str, int]:
    """Descarga playergamelogs para una season completa e INSERT a player_game_logs."""
    # nba_api import lazy (CPU-expensive)
    from nba_api.stats.endpoints import playergamelogs

    def _fetch() -> list[dict[str, Any]]:
        logs = playergamelogs.PlayerGameLogs(season_nullable=season)
        return logs.get_normalized_dict().get("PlayerGameLogs", [])

    raw = await asyncio.to_thread(_fetch)
    logger.info("seed_player_logs.nba.fetched", season=season, rows=len(raw))

    inserted = 0
    skipped = 0
    async with session_scope() as session:
        for row in raw:
            try:
                player_id = await _resolve_player_id(session, full_name=row.get("PLAYER_NAME", ""))
                if player_id is None:
                    skipped += 1
                    continue
                match_id = await _resolve_match_id(
                    session,
                    game_date=row.get("GAME_DATE", ""),
                    team_id_nba_api=int(row.get("TEAM_ID", 0)),
                )
                if match_id is None:
                    skipped += 1
                    continue
                stats = {
                    "points": int(row.get("PTS") or 0),
                    "rebounds": int(row.get("REB") or 0),
                    "assists": int(row.get("AST") or 0),
                    "steals": int(row.get("STL") or 0),
                    "blocks": int(row.get("BLK") or 0),
                    "three_pointers_made": int(row.get("FG3M") or 0),
                    "turnovers": int(row.get("TOV") or 0),
                    "field_goals_made": int(row.get("FGM") or 0),
                    "field_goals_attempted": int(row.get("FGA") or 0),
                    "plus_minus": int(row.get("PLUS_MINUS") or 0),
                }
                minutes = None
                min_str = row.get("MIN")
                if isinstance(min_str, (int, float)):
                    minutes = float(min_str)
                await session.execute(
                    text(
                        """
                        INSERT INTO player_game_logs
                            (player_id, match_id, sport_code, stats, minutes_played)
                        VALUES (:pid, :mid, 'nba', CAST(:stats AS jsonb), :min)
                        ON CONFLICT (player_id, match_id) DO UPDATE SET
                            stats = EXCLUDED.stats,
                            minutes_played = EXCLUDED.minutes_played
                        """
                    ),
                    {
                        "pid": player_id,
                        "mid": match_id,
                        "stats": json.dumps(stats),
                        "min": minutes,
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug(
                    "seed_player_logs.row_fail",
                    player=row.get("PLAYER_NAME"),
                    error=str(exc)[:80],
                )
                skipped += 1
    return {"inserted": inserted, "skipped": skipped, "total": len(raw)}


async def main(args: argparse.Namespace) -> None:
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if args.sport == "nba":
        for season in seasons:
            result = await seed_nba_season(season)
            print(f"✅ NBA {season}: {result}")
    else:
        print(f"⚠  Sport {args.sport} pendiente de implementación en este script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba", choices=["nba"])
    parser.add_argument("--seasons", default="2023-24,2024-25", help="CSV seasons")
    asyncio.run(main(parser.parse_args()))
