"""FASE M.2 — Compute player_stat_std desde player_game_logs (zero hardcoded).

Elimina NBA_STAT_STD y NFL_STAT_STD hardcoded en `sport_props_models.py`
calculando la desviación estándar real por jugador × stat desde la data
histórica.

Persiste en tabla `player_stat_std`:
    (player_id, sport_code, stat_type, std, sample_size, updated_at)

Uso:
    apuestas compute-player-std --sport nba
    apuestas compute-player-std --all

Fallback: si un jugador tiene < 30 games, usa std global del sport.
Si no hay data del sport, usa constante histórica documentada.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_MIN_SAMPLE_FOR_INDIVIDUAL = 30
_NBA_STATS = ["points", "rebounds", "assists", "three_pointers_made", "steals", "blocks"]
_NFL_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "completions", "receptions"]
_MLB_STATS = ["hits", "strikeouts", "home_runs", "total_bases"]
_NHL_STATS = ["goals", "assists", "shots", "saves"]


async def ensure_table_exists() -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS player_stat_std (
                    player_id BIGINT NOT NULL,
                    sport_code TEXT NOT NULL,
                    stat_type TEXT NOT NULL,
                    std NUMERIC(10, 4) NOT NULL,
                    mean_value NUMERIC(10, 4),
                    sample_size INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (player_id, sport_code, stat_type)
                )
                """
            )
        )
        await s.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_player_stat_std_sport "
                "ON player_stat_std (sport_code, stat_type)"
            )
        )


async def compute_std_for_stat(sport_code: str, stat_type: str) -> int:
    """Calcula std por jugador para un sport × stat específico.

    player_game_logs schema real usa JSONB `stats` con la key = stat_type
    (ej. "points", "rebounds"). Extraemos con `stats->>stat_type`.
    """
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT player_id,
                       AVG((stats ->> :st)::numeric)::numeric(10,4) AS mean_v,
                       STDDEV((stats ->> :st)::numeric)::numeric(10,4) AS std_v,
                       COUNT(*) AS n
                FROM player_game_logs
                WHERE sport_code = :sc
                  AND stats ? :st
                  AND (stats ->> :st) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                GROUP BY player_id
                HAVING COUNT(*) >= :min_n
                """
            ),
            {"sc": sport_code, "st": stat_type, "min_n": _MIN_SAMPLE_FOR_INDIVIDUAL},
        )
        rows = r.all()

    if not rows:
        logger.debug("player_stat_std.no_data", sport=sport_code, stat=stat_type)
        return 0

    n_written = 0
    async with session_scope() as s:
        for row in rows:
            if row.std_v is None or float(row.std_v) <= 0:
                continue
            await s.execute(
                text(
                    """
                    INSERT INTO player_stat_std
                      (player_id, sport_code, stat_type, std, mean_value, sample_size)
                    VALUES
                      (:p, :sc, :st, :std, :mn, :n)
                    ON CONFLICT (player_id, sport_code, stat_type) DO UPDATE SET
                      std = EXCLUDED.std,
                      mean_value = EXCLUDED.mean_value,
                      sample_size = EXCLUDED.sample_size,
                      updated_at = NOW()
                    """
                ),
                {
                    "p": int(row.player_id),
                    "sc": sport_code,
                    "st": stat_type,
                    "std": float(row.std_v),
                    "mn": float(row.mean_v) if row.mean_v else 0.0,
                    "n": int(row.n),
                },
            )
            n_written += 1
    return n_written


async def get_player_std(
    player_id: int, sport_code: str, stat_type: str, *, fallback: float
) -> float:
    """Lee std de DB; fallback si no hay data suficiente."""
    async with session_scope() as s:
        r = await s.execute(
            text(
                "SELECT std FROM player_stat_std "
                "WHERE player_id = :p AND sport_code = :sc AND stat_type = :st"
            ),
            {"p": player_id, "sc": sport_code, "st": stat_type},
        )
        row = r.first()
    return float(row.std) if row else fallback


async def compute_all_sports(sports: list[str]) -> dict[str, dict[str, int]]:
    """Compute std para lista de sports."""
    await ensure_table_exists()
    stat_map = {
        "nba": _NBA_STATS,
        "nfl": _NFL_STATS,
        "mlb": _MLB_STATS,
        "nhl": _NHL_STATS,
    }
    results: dict[str, dict[str, int]] = {}
    for sport in sports:
        if sport not in stat_map:
            continue
        per_stat: dict[str, int] = {}
        for stat in stat_map[sport]:
            n = await compute_std_for_stat(sport, stat)
            per_stat[stat] = n
        results[sport] = per_stat
        logger.info(
            "player_stat_std.sport_done",
            sport=sport,
            total_rows=sum(per_stat.values()),
        )
    return results


async def main(args: argparse.Namespace) -> None:
    if args.all:
        sports = ["nba", "nfl", "mlb", "nhl"]
    else:
        sports = [s.strip() for s in (args.sport or "nba").split(",")]
    results = await compute_all_sports(sports)
    print("✅ Player stat std calculado:")
    for sport, stats in results.items():
        print(f"  {sport}: {stats}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sport", default="", help="CSV sport codes")
    p.add_argument("--all", action="store_true")
    asyncio.run(main(p.parse_args()))
