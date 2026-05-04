"""Seed NHL player_game_logs desde NHL Stats Web API (api-web.nhle.com).

Endpoint gratis sin auth: `https://api-web.nhle.com/v1/player/{player_id}/game-log/{season}/{type}`
Descarga logs por jugador × temporada. Para MVP arranca con top scorers del
año (ids conocidos) y va expandiendo.

Uso:
    apuestas seed-player-logs nhl --seasons 20232024,20242025
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_NHL_API_BASE = "https://api-web.nhle.com/v1"
_GAME_TYPE_REGULAR = 2


async def _resolve_player_id(
    session: Any, *, full_name: str, nhl_api_id: int | None = None
) -> int | None:
    if nhl_api_id:
        r = await session.execute(
            text("SELECT id FROM players WHERE external_id = :eid LIMIT 1"),
            {"eid": f"nhl:api:{nhl_api_id}"},
        )
        row = r.first()
        if row:
            return int(row.id)

    r = await session.execute(
        text("SELECT id FROM players WHERE sport_code = 'nhl' AND full_name ILIKE :n LIMIT 1"),
        {"n": full_name.strip()},
    )
    row = r.first()
    if row:
        return int(row.id)

    ext = f"nhl:api:{nhl_api_id}" if nhl_api_id else f"nhl:name:{full_name.lower()}"
    r2 = await session.execute(
        text(
            """
            INSERT INTO players (external_id, sport_code, full_name, created_at)
            VALUES (:ext, 'nhl', :name, NOW())
            ON CONFLICT (external_id) DO UPDATE SET full_name = EXCLUDED.full_name
            RETURNING id
            """
        ),
        {"ext": ext, "name": full_name},
    )
    pid = r2.first()
    return int(pid.id) if pid else None


async def _resolve_match_id_nhl(session: Any, *, game_date: str, team_abbr: str) -> int | None:
    r = await session.execute(
        text(
            """
            SELECT m.id FROM matches m
            JOIN teams th ON th.id = m.home_team_id
            JOIN teams ta ON ta.id = m.away_team_id
            WHERE m.sport_code = 'nhl'
              AND DATE(m.start_time) = DATE(:gd)
              AND (th.external_id ILIKE :tx OR ta.external_id ILIKE :tx
                   OR th.name ILIKE :tn OR ta.name ILIKE :tn)
            LIMIT 1
            """
        ),
        {"gd": game_date, "tx": f"%{team_abbr}%", "tn": f"%{team_abbr}%"},
    )
    row = r.first()
    return int(row.id) if row else None


async def fetch_top_scorers(season: str) -> list[dict[str, Any]]:
    """Top 200 NHL scorers de la season — NHL stats API endpoint."""
    url = f"{_NHL_API_BASE}/skater-stats-leaders/{season}/{_GAME_TYPE_REGULAR}?categories=points&limit=200"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
            logger.warning("seed_nhl.top_scorers_fail", status=r.status_code)
            return []
        data = r.json()
        return list(data.get("points", []))


async def fetch_player_game_log(player_id: int, season: str) -> list[dict[str, Any]]:
    url = f"{_NHL_API_BASE}/player/{player_id}/game-log/{season}/{_GAME_TYPE_REGULAR}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url)
        if r.status_code != 200:
            return []
        return list(r.json().get("gameLog", []))


async def seed_nhl_season(season: str) -> dict[str, int]:
    """Descarga top 200 scorers + sus game logs completos."""
    scorers = await fetch_top_scorers(season)
    if not scorers:
        return {"inserted": 0, "skipped": 0, "scorers": 0}

    logger.info("seed_player_logs.nhl.scorers", season=season, n=len(scorers))
    inserted = 0
    skipped = 0

    for scorer in scorers[:100]:  # Top 100 para MVP
        try:
            api_pid = int(scorer.get("id") or 0)
            full_name = f"{scorer.get('firstName', {}).get('default', '')} {scorer.get('lastName', {}).get('default', '')}".strip()
            if not full_name or not api_pid:
                skipped += 1
                continue

            game_logs = await fetch_player_game_log(api_pid, season)
            if not game_logs:
                skipped += 1
                continue

            async with session_scope() as session:
                player_id = await _resolve_player_id(
                    session, full_name=full_name, nhl_api_id=api_pid
                )
                if player_id is None:
                    skipped += 1
                    continue

                for log in game_logs:
                    game_date = str(log.get("gameDate", ""))
                    team_abbr = str(log.get("teamAbbrev", ""))
                    if not game_date or not team_abbr:
                        skipped += 1
                        continue

                    match_id = await _resolve_match_id_nhl(
                        session, game_date=game_date, team_abbr=team_abbr
                    )
                    if match_id is None:
                        skipped += 1
                        continue

                    stats = {
                        "goals": int(log.get("goals", 0) or 0),
                        "assists": int(log.get("assists", 0) or 0),
                        "points": int(log.get("points", 0) or 0),
                        "shots": int(log.get("shots", 0) or 0),
                        "plus_minus": int(log.get("plusMinus", 0) or 0),
                        "pim": int(log.get("pim", 0) or 0),  # penalty minutes
                        "toi": str(log.get("toi", "0:00")),  # time on ice
                        "power_play_goals": int(log.get("powerPlayGoals", 0) or 0),
                    }
                    try:
                        await session.execute(
                            text(
                                """
                                INSERT INTO player_game_logs
                                    (player_id, match_id, sport_code, stats)
                                VALUES (:pid, :mid, 'nhl', CAST(:stats AS jsonb))
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
                            "seed_nhl.insert_fail",
                            player=full_name,
                            match=match_id,
                            error=str(exc)[:80],
                        )
                        skipped += 1
        except Exception as exc:
            logger.debug("seed_nhl.scorer_fail", error=str(exc)[:80])
            skipped += 1

    return {"inserted": inserted, "skipped": skipped, "scorers": len(scorers)}


async def main(args: argparse.Namespace) -> None:
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    for season in seasons:
        try:
            result = await seed_nhl_season(season)
            print(f"✅ NHL {season}: {result}")
        except Exception as exc:
            print(f"❌ NHL {season}: {str(exc)[:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        default="20232024,20242025",
        help="CSV NHL season format (ej. 20232024 = 2023-24)",
    )
    asyncio.run(main(parser.parse_args()))
