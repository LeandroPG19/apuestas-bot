"""Bulk fit de distribuciones props por (player, sport, stat).

Lee `player_game_logs`, fit paramétrico (Poisson/NegBinomial/Gamma) por
jugador × stat, persiste params en `player_prop_distributions` para que
`detect_value_props` use los cached params sin re-fit cada vez.

Uso:
    apuestas bulk-fit-props --sport nba
    apuestas bulk-fit-props --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ml.props_distributions import (
    fit_gamma,
    fit_neg_binomial,
    fit_poisson,
)
from apuestas.obs.logging import get_logger
from apuestas.schemas.props import PropDistribution, get_prop

logger = get_logger(__name__)

_MIN_SAMPLES = 15


_STATS_BY_SPORT = {
    "nba": ("points", "rebounds", "assists", "three_pointers_made", "steals", "blocks"),
    "nfl": (
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "completions",
        "passing_tds",
    ),
    "mlb": ("home_runs", "hits", "rbi", "runs", "total_bases", "strikeouts"),
    "nhl": ("goals", "assists", "points", "shots"),
}


async def ensure_table_exists() -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS player_prop_distributions (
                    player_id BIGINT NOT NULL,
                    sport_code TEXT NOT NULL,
                    stat_type TEXT NOT NULL,
                    distribution TEXT NOT NULL,
                    params JSONB NOT NULL,
                    mean_value NUMERIC(10, 4),
                    std_value NUMERIC(10, 4),
                    sample_size INTEGER NOT NULL,
                    fitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (player_id, sport_code, stat_type)
                )
                """
            )
        )
        await s.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_player_prop_dist_sport_stat "
                "ON player_prop_distributions (sport_code, stat_type)"
            )
        )


async def fit_for_sport(sport_code: str) -> dict[str, int]:
    stats = _STATS_BY_SPORT.get(sport_code, ())
    if not stats:
        return {"fitted": 0, "skipped": 0}

    fitted = 0
    skipped = 0

    for stat in stats:
        prop_code = f"player_{stat}" if sport_code == "nba" else stat
        try:
            prop_def = get_prop(prop_code)
            distribution = prop_def.distribution
        except Exception:
            distribution = PropDistribution.NEG_BINOMIAL

        async with session_scope() as s:
            r = await s.execute(
                text(
                    """
                    SELECT player_id,
                           ARRAY_AGG((stats ->> :st)::float) AS vals
                    FROM player_game_logs
                    WHERE sport_code = :sc
                      AND stats ? :st
                      AND (stats ->> :st) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    GROUP BY player_id
                    HAVING COUNT(*) >= :min_n
                    """
                ),
                {"sc": sport_code, "st": stat, "min_n": _MIN_SAMPLES},
            )
            rows = r.all()

        async with session_scope() as s:
            for row in rows:
                try:
                    samples = np.array(row.vals, dtype=np.float64)
                    samples = samples[~np.isnan(samples)]
                    if len(samples) < _MIN_SAMPLES:
                        skipped += 1
                        continue

                    if distribution == PropDistribution.POISSON:
                        dist = fit_poisson(samples)
                    elif distribution == PropDistribution.NEG_BINOMIAL:
                        dist = fit_neg_binomial(samples)
                    elif distribution == PropDistribution.GAMMA:
                        dist = fit_gamma(samples)
                    else:
                        skipped += 1
                        continue

                    params: dict[str, Any] = {"mean": dist.mean, "std": dist.std}
                    if hasattr(dist, "lam"):
                        params["lam"] = dist.lam
                    if hasattr(dist, "dispersion"):
                        params["dispersion"] = dist.dispersion

                    await s.execute(
                        text(
                            """
                            INSERT INTO player_prop_distributions
                                (player_id, sport_code, stat_type, distribution,
                                 params, mean_value, std_value, sample_size)
                            VALUES (:p, :sc, :st, :d, CAST(:pr AS jsonb),
                                    :mn, :sd, :n)
                            ON CONFLICT (player_id, sport_code, stat_type) DO UPDATE SET
                                distribution = EXCLUDED.distribution,
                                params = EXCLUDED.params,
                                mean_value = EXCLUDED.mean_value,
                                std_value = EXCLUDED.std_value,
                                sample_size = EXCLUDED.sample_size,
                                fitted_at = NOW()
                            """
                        ),
                        {
                            "p": int(row.player_id),
                            "sc": sport_code,
                            "st": stat,
                            "d": distribution.value,
                            "pr": json.dumps(params),
                            "mn": dist.mean,
                            "sd": dist.std,
                            "n": len(samples),
                        },
                    )
                    fitted += 1
                except Exception as exc:
                    logger.debug(
                        "bulk_fit.row_fail",
                        player=row.player_id,
                        stat=stat,
                        error=str(exc)[:80],
                    )
                    skipped += 1
        logger.info(
            "bulk_fit.stat_done", sport=sport_code, stat=stat, fitted=fitted, skipped=skipped
        )

    return {"fitted": fitted, "skipped": skipped}


async def main(args: argparse.Namespace) -> None:
    await ensure_table_exists()
    sports = ["nba", "nfl", "mlb", "nhl"] if args.all else [args.sport]
    total: dict[str, dict[str, int]] = {}
    for sport in sports:
        result = await fit_for_sport(sport)
        total[sport] = result
        print(f"✅ {sport}: {result}")
    print(f"📊 Total: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--all", action="store_true")
    asyncio.run(main(parser.parse_args()))
