"""FASE M — Compute league averages desde data histórica.

Elimina hardcoded LEAGUE_AVERAGES en `soccer_props_model.py` calculando los
promedios reales por liga desde los matches sembrados.

Lee `matches` + `odds_history` + (futuro) `match_statistics` de Sofascore.
Persiste en tabla `league_stats_averages` para que `soccer_props_model.py`
la consulte en runtime.

Uso:
    apuestas compute-league-averages --leagues epl,laliga,bundesliga
    apuestas compute-league-averages --all
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def ensure_table_exists() -> None:
    """Crea tabla league_stats_averages si no existe."""
    async with session_scope() as s:
        await s.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS league_stats_averages (
                    league_id BIGINT NOT NULL,
                    sport_code TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    avg_value NUMERIC(10, 4) NOT NULL,
                    sample_size INTEGER NOT NULL,
                    window_seasons TEXT,
                    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (league_id, sport_code, metric)
                )
                """
            )
        )
        await s.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_league_stats_avg_sport "
                "ON league_stats_averages (sport_code)"
            )
        )


async def compute_goals_averages(sport_code: str) -> dict[str, float]:
    """Promedio goles/partido por equipo en un sport.

    Para soccer: usa home_score + away_score de matches finished.
    """
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT AVG((home_score + away_score) / 2.0) AS goals_per_team,
                       COUNT(*) AS n
                FROM matches
                WHERE sport_code = :sc
                  AND status = 'finished'
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                """
            ),
            {"sc": sport_code},
        )
        row = r.first()
    if row and row.goals_per_team and row.n:
        return {
            "goals_per_team": float(row.goals_per_team),
            "sample_size": int(row.n),
        }
    return {}


async def persist_averages(
    sport_code: str, averages: dict[str, float], *, league_id: int = 0
) -> int:
    """Persiste averages en tabla league_stats_averages."""
    n_written = 0
    sample_size = int(averages.pop("sample_size", 0)) if "sample_size" in averages else 0
    async with session_scope() as s:
        for metric, value in averages.items():
            await s.execute(
                text(
                    """
                    INSERT INTO league_stats_averages
                        (league_id, sport_code, metric, avg_value, sample_size)
                    VALUES (:lid, :sc, :m, :v, :n)
                    ON CONFLICT (league_id, sport_code, metric) DO UPDATE SET
                        avg_value = EXCLUDED.avg_value,
                        sample_size = EXCLUDED.sample_size,
                        computed_at = NOW()
                    """
                ),
                {
                    "lid": league_id,
                    "sc": sport_code,
                    "m": metric,
                    "v": float(value),
                    "n": sample_size,
                },
            )
            n_written += 1
    return n_written


async def compute_all_averages(sports: list[str]) -> dict[str, dict[str, float]]:
    """Compute + persist promedios para lista de sports.

    Returns dict con el resultado por sport para logging.
    """
    await ensure_table_exists()
    results: dict[str, dict[str, float]] = {}
    for sport in sports:
        avgs = await compute_goals_averages(sport)
        if avgs:
            n = await persist_averages(sport, dict(avgs))
            logger.info(
                "league_averages.computed",
                sport=sport,
                metrics_written=n,
                goals_per_team=avgs.get("goals_per_team"),
                sample=avgs.get("sample_size"),
            )
            results[sport] = avgs
        else:
            logger.warning("league_averages.no_data", sport=sport)
    return results


async def get_league_average(
    sport_code: str, metric: str, *, league_id: int = 0, fallback: float = 0.0
) -> float:
    """Lee avg de DB; fallback si no existe.

    Uso desde soccer_props_model.py:
        from apuestas.scripts.compute_league_averages import get_league_average
        goals_avg = await get_league_average("soccer", "goals_per_team", fallback=1.4)
    """
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT avg_value, sample_size FROM league_stats_averages
                WHERE sport_code = :sc AND metric = :m AND league_id = :lid
                """
            ),
            {"sc": sport_code, "m": metric, "lid": league_id},
        )
        row = r.first()
    if row and row.sample_size >= 50:
        return float(row.avg_value)
    return fallback


async def main(args: argparse.Namespace) -> None:
    if args.all:
        sports = ["soccer", "nba", "mlb", "nfl", "nhl", "tennis", "epl", "laliga", "liga_mx"]
    elif args.sports:
        sports = [s.strip() for s in args.sports.split(",")]
    else:
        sports = ["soccer", "nba", "mlb", "nfl"]
    r = await compute_all_averages(sports)
    print(f"✅ Processed {len(r)} sports:")
    for sport, avgs in r.items():
        print(f"  {sport}: {avgs}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sports", default="", help="CSV sport codes")
    p.add_argument("--all", action="store_true", help="Todos los deportes")
    asyncio.run(main(p.parse_args()))
