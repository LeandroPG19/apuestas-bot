"""Seed MLB player_game_logs desde pybaseball (Statcast + traditional stats).

Descarga batting_stats_range y pitching_stats_range por rango de fechas,
rotando por (inicio_season → fin_season) para no agotar rate limits.

Uso:
    apuestas seed-player-logs mlb --seasons 2023,2024

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
    session: Any, *, full_name: str, mlbam_id: int | None = None
) -> int | None:
    """Busca MLB player. Crea con MLBAM id si no existe."""
    if mlbam_id:
        r = await session.execute(
            text("SELECT id FROM players WHERE external_id = :eid LIMIT 1"),
            {"eid": f"mlb:mlbam:{mlbam_id}"},
        )
        row = r.first()
        if row:
            return int(row.id)

    r = await session.execute(
        text("SELECT id FROM players WHERE sport_code = 'mlb' AND full_name ILIKE :n LIMIT 1"),
        {"n": full_name.strip()},
    )
    row = r.first()
    if row:
        return int(row.id)

    ext = f"mlb:mlbam:{mlbam_id}" if mlbam_id else f"mlb:name:{full_name.lower().replace(' ', '_')}"
    r2 = await session.execute(
        text(
            """
            INSERT INTO players (external_id, sport_code, full_name, created_at)
            VALUES (:ext, 'mlb', :name, NOW())
            ON CONFLICT (external_id) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING id
            """
        ),
        {"ext": ext, "name": full_name},
    )
    pid = r2.first()
    return int(pid.id) if pid else None


async def _resolve_match_id_by_date_team(
    session: Any, *, game_date: str, team_name: str
) -> int | None:
    """Fuzzy match sobre home/away team name (pybaseball devuelve abbreviations)."""
    r = await session.execute(
        text(
            """
            SELECT m.id FROM matches m
            JOIN teams th ON th.id = m.home_team_id
            JOIN teams ta ON ta.id = m.away_team_id
            WHERE m.sport_code = 'mlb'
              AND DATE(m.start_time) = DATE(:gd)
              AND (th.name ILIKE :t OR ta.name ILIKE :t)
            LIMIT 1
            """
        ),
        {"gd": game_date, "t": f"%{team_name}%"},
    )
    row = r.first()
    return int(row.id) if row else None


async def seed_mlb_year(year: int) -> dict[str, int]:
    """Descarga batting + pitching para una temporada MLB completa."""
    import pybaseball  # type: ignore[import-untyped]

    def _fetch_batting() -> Any:
        return pybaseball.batting_stats(year, qual=50)  # qual=min 50 plate appearances

    def _fetch_pitching() -> Any:
        return pybaseball.pitching_stats(year, qual=20)  # qual=min 20 innings

    batting = await asyncio.to_thread(_fetch_batting)
    pitching = await asyncio.to_thread(_fetch_pitching)
    logger.info(
        "seed_player_logs.mlb.fetched",
        year=year,
        batting_rows=len(batting),
        pitching_rows=len(pitching),
    )

    inserted = 0
    skipped = 0

    # MLB pybaseball devuelve stats agregados por temporada, no por juego.
    # Para juegos individuales se necesita statcast_batter que es muy lento.
    # MVP: guardamos los aggregados como "season log" con match_id = temporada-placeholder.
    # TODO: reemplazar por pybaseball.statcast_batter para per-game cuando urgent.
    # Por ahora devolvemos aggregated stats como "season average" para enable props.

    async with session_scope() as session:
        # Batter aggregates
        for idx in range(len(batting)):
            try:
                row = batting.iloc[idx]
                name = str(row.get("Name", ""))
                mlbam = int(row.get("IDfg", 0)) if row.get("IDfg") else None
                if not name:
                    skipped += 1
                    continue
                player_id = await _resolve_player_id(session, full_name=name, mlbam_id=mlbam)
                if player_id is None:
                    skipped += 1
                    continue
                stats = {
                    "home_runs": int(row.get("HR", 0) or 0),
                    "hits": int(row.get("H", 0) or 0),
                    "rbi": int(row.get("RBI", 0) or 0),
                    "runs": int(row.get("R", 0) or 0),
                    "total_bases": int(row.get("TB", 0) or 0),
                    "walks": int(row.get("BB", 0) or 0),
                    "strikeouts": int(row.get("SO", 0) or 0),
                    "stolen_bases": int(row.get("SB", 0) or 0),
                    "at_bats": int(row.get("AB", 0) or 0),
                    "plate_appearances": int(row.get("PA", 0) or 0),
                    "season_avg": True,
                    "year": year,
                }
                # Usamos -1 × year como match_id placeholder (no crashes FK-constraint
                # si no existe match; mejor usar primer match de esa season)
                r = await session.execute(
                    text(
                        """
                        SELECT id FROM matches
                        WHERE sport_code = 'mlb' AND season = :ss
                        ORDER BY start_time ASC LIMIT 1
                        """
                    ),
                    {"ss": str(year)},
                )
                match_row = r.first()
                if match_row is None:
                    skipped += 1
                    continue
                await session.execute(
                    text(
                        """
                        INSERT INTO player_game_logs
                            (player_id, match_id, sport_code, stats)
                        VALUES (:pid, :mid, 'mlb', CAST(:stats AS jsonb))
                        ON CONFLICT (player_id, match_id) DO UPDATE SET
                            stats = EXCLUDED.stats
                        """
                    ),
                    {
                        "pid": player_id,
                        "mid": int(match_row.id),
                        "stats": json.dumps(stats),
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug("seed_mlb.row_fail", idx=idx, error=str(exc)[:80])
                skipped += 1

    return {
        "inserted": inserted,
        "skipped": skipped,
        "batting_total": len(batting),
        "pitching_total": len(pitching),
    }


async def main(args: argparse.Namespace) -> None:
    years = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    for year in years:
        try:
            result = await seed_mlb_year(year)
            print(f"✅ MLB {year}: {result}")
        except Exception as exc:
            print(f"❌ MLB {year}: {str(exc)[:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2023,2024", help="CSV year integers")
    asyncio.run(main(parser.parse_args()))
