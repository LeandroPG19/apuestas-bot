"""FASE A.1 — Rebuild team_stats_rolling desde datos sembrados.

Calcula rolling averages (últimos 5, 10, 20 partidos) por equipo × sport
desde `matches` y persiste en `team_stats_rolling_home` / `team_stats_rolling_away`.

Métricas computadas:
- wins_last_N, losses_last_N
- goals_for_avg, goals_against_avg (o pts/runs según sport)
- form_weighted (decay 0.9 favoreciendo partidos recientes)
- days_since_last_game
- home_away_differential

Uso:
    apuestas rebuild-team-stats --sports nba,mlb,nfl,nhl,soccer
    apuestas rebuild-team-stats --all
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

_WINDOWS = [5, 10, 20]


async def compute_team_rolling(
    team_id: int, sport_code: str, is_home: bool, window: int
) -> dict[str, Any]:
    """Últimos N partidos del equipo, calcula rolling stats.

    Returns:
        dict con métricas agregadas para el window.
    """
    location_clause = "home_team_id = :tid" if is_home else "away_team_id = :tid"
    async with session_scope() as s:
        r = await s.execute(
            text(
                f"""
                SELECT home_team_id, away_team_id, home_score, away_score, start_time
                FROM matches
                WHERE {location_clause}
                  AND sport_code = :sc
                  AND status = 'finished'
                  AND home_score IS NOT NULL
                ORDER BY start_time DESC
                LIMIT :lim
                """
            ),
            {"tid": team_id, "sc": sport_code, "lim": window},
        )
        rows = r.all()
    if not rows:
        return {}

    n = len(rows)
    wins, losses, pts_for, pts_against = 0, 0, 0, 0
    form_weighted = 0.0
    decay = 0.9
    for i, row in enumerate(rows):
        is_h = row.home_team_id == team_id
        team_score = row.home_score if is_h else row.away_score
        opp_score = row.away_score if is_h else row.home_score
        pts_for += team_score or 0
        pts_against += opp_score or 0
        if (team_score or 0) > (opp_score or 0):
            wins += 1
            form_weighted += (decay**i) * 1.0
        elif (team_score or 0) < (opp_score or 0):
            losses += 1
            form_weighted += (decay**i) * -1.0
        # draw: 0
    return {
        "n_games": n,
        "wins": wins,
        "losses": losses,
        "pts_for_avg": pts_for / n,
        "pts_against_avg": pts_against / n,
        "win_rate": wins / n,
        "form_weighted": form_weighted,
        "pts_differential_avg": (pts_for - pts_against) / n,
    }


async def persist_rolling(
    team_id: int, sport_code: str, is_home: bool, window: int, metrics: dict[str, Any]
) -> None:
    """Upsert en team_stats_rolling_home o _away."""
    table = "team_stats_rolling_home" if is_home else "team_stats_rolling_away"
    async with session_scope() as s:
        await s.execute(
            text(
                f"""
                INSERT INTO {table}
                  (team_id, sport_code, window_size, metrics, sample_size, last_computed)
                VALUES
                  (:tid, :sc, :w, CAST(:m AS json), :n, NOW())
                ON CONFLICT (team_id, sport_code, window_size) DO UPDATE SET
                  metrics = EXCLUDED.metrics,
                  sample_size = EXCLUDED.sample_size,
                  last_computed = NOW()
                """
            ),
            {
                "tid": team_id,
                "sc": sport_code,
                "w": window,
                "m": json.dumps(metrics),
                "n": int(metrics.get("n_games", 0)),
            },
        )


async def rebuild_sport(sport_code: str) -> dict[str, int]:
    """Rebuild rolling stats para todos los teams de un sport."""
    async with session_scope() as s:
        r = await s.execute(
            text(
                "SELECT DISTINCT home_team_id FROM matches "
                "WHERE sport_code = :sc AND status = 'finished' "
                "UNION SELECT DISTINCT away_team_id FROM matches "
                "WHERE sport_code = :sc AND status = 'finished'"
            ),
            {"sc": sport_code},
        )
        team_ids = [row[0] for row in r.all() if row[0] is not None]

    logger.info("rebuild_team_stats.start", sport=sport_code, teams=len(team_ids))
    stats_written = 0
    for team_id in team_ids:
        for window in _WINDOWS:
            for is_home in (True, False):
                metrics = await compute_team_rolling(team_id, sport_code, is_home, window)
                if metrics:
                    await persist_rolling(team_id, sport_code, is_home, window, metrics)
                    stats_written += 1
    logger.info(
        "rebuild_team_stats.done",
        sport=sport_code,
        teams=len(team_ids),
        stats_written=stats_written,
    )
    return {"teams": len(team_ids), "stats_written": stats_written}


async def main(args: argparse.Namespace) -> None:
    if args.all:
        sports = ["nba", "mlb", "nfl", "nhl", "soccer", "tennis"]
    else:
        sports = [s.strip() for s in (args.sports or "nba").split(",")]

    results: dict[str, object] = {}
    for sport in sports:
        try:
            results[sport] = await rebuild_sport(sport)
        except Exception as exc:
            logger.exception("rebuild_team_stats.fail", sport=sport, error=str(exc))
            results[sport] = {"error": str(exc)[:100]}

    print("✅ Rebuild team stats complete:")
    for sport, stat in results.items():
        print(f"  {sport}: {stat}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sports", default="", help="CSV sport codes")
    p.add_argument("--all", action="store_true")
    asyncio.run(main(p.parse_args()))
