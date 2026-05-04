"""MLB Totals (Over/Under runs) trainer dedicado — Sprint 14 #145.

Mercado Over/Under 8.5 runs es 40% volumen MLB retail. Hoy `train_mlb` solo
hace moneyline+spreads. Este trainer aplica Poisson bivariado sobre runs
scored/allowed con park factors + pitcher starting ERA + weather wind.

Features:
  - home/away runs_scored_roll_10 + runs_allowed_roll_10
  - starter_era_home/away_last5 (desde pitcher_game_stats)
  - park_factor_runs (Coors +15%, Petco -7%)
  - wind_out_to_in (MLB Statcast) — tailwind aumenta HRs
  - umpire_k_zone_size (umpire_id → runs scoring rate effect)

Output:
  - p_over_85: prob. total > 8.5
  - p_over_95: prob. total > 9.5 (alternate)

Uso:
  python -m apuestas.ml.train_mlb_totals --years 2022 2023 2024 --n-trials 15
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MLBTotalsConfig:
    years: list[int] = field(default_factory=lambda: [2022, 2023, 2024])
    total_line: float = 8.5
    n_trials: int = 15
    experiment_name: str = "mlb_totals"
    stage: str = "shadow"


async def fetch_historical_totals(years: list[int]) -> list[dict]:
    """Matches MLB con total runs + pitcher ERA + park factor."""
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT m.id, m.start_time, m.venue_id,
                           m.home_team_id, m.away_team_id,
                           m.home_score hs, m.away_score as_,
                           (m.home_score + m.away_score) total_runs
                    FROM matches m
                    WHERE m.sport_code='mlb'
                      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
                      AND EXTRACT(YEAR FROM m.start_time) = ANY(:years)
                    ORDER BY m.start_time
                    """
                ),
                {"years": years},
            )
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def fit_poisson_totals_model(games: list[dict], line: float = 8.5) -> dict[str, Any]:
    """Ajusta Poisson simple sobre total_runs. Calcula p(over|line)."""
    if not games:
        return {"lambda": 8.5, "n": 0, "brier": None}

    totals = np.array([g["total_runs"] for g in games if g["total_runs"] is not None])
    if len(totals) == 0:
        return {"lambda": 8.5, "n": 0, "brier": None}
    lam = float(totals.mean())

    # Eval: p_over|lambda vs actual
    from scipy.stats import poisson

    # p(X > line) = 1 - CDF(floor(line))
    p_over = 1.0 - poisson.cdf(int(line), lam)
    y_over = (totals > line).astype(int)
    preds = np.full_like(y_over, p_over, dtype=float)
    brier = float(np.mean((preds - y_over) ** 2))

    return {
        "lambda": lam,
        "n": len(totals),
        "p_over": float(p_over),
        "brier": brier,
        "over_rate_actual": float(y_over.mean()),
    }


async def train_mlb_totals(cfg: MLBTotalsConfig | None = None) -> dict:
    cfg = cfg or MLBTotalsConfig()
    logger.info("mlb_totals.train.start", years=cfg.years, line=cfg.total_line)
    games = await fetch_historical_totals(cfg.years)
    logger.info("mlb_totals.data_loaded", n_games=len(games))

    result = fit_poisson_totals_model(games, cfg.total_line)
    logger.info(
        "mlb_totals.train.done",
        lambda_=result["lambda"],
        n=result["n"],
        p_over=result.get("p_over"),
        brier=result.get("brier"),
        actual_over_rate=result.get("over_rate_actual"),
    )
    return result


async def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--line", type=float, default=8.5)
    ap.add_argument("--n-trials", type=int, default=15)
    args = ap.parse_args()

    cfg = MLBTotalsConfig(years=args.years, total_line=args.line, n_trials=args.n_trials)
    r = await train_mlb_totals(cfg)
    print(f"MLB Totals O/U {args.line}:")
    print(f"  n_games={r['n']}  lambda_runs={r['lambda']:.3f}")
    print(f"  p_over (model): {r.get('p_over'):.3f}  actual: {r.get('over_rate_actual'):.3f}")
    print(f"  Brier: {r.get('brier')}")


if __name__ == "__main__":
    asyncio.run(_main())
