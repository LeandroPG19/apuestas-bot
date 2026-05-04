"""Soccer BTTS (Both Teams To Score) + Asian Handicap trainer — Sprint 14 #152.

Mercados 20-25% volumen soccer retail sin modelo dedicado hoy.

BTTS: Poisson bivariado P(home ≥1 goal AND away ≥1 goal).
AH (Asian Handicap ±0.5, ±1.0): ajusta 1X2 con handicap continuo.

Features:
  - xg_for/against rolling 10
  - home_clean_sheet_rate
  - away_scoring_rate
  - league_avg_goals_match (liga-specific)

Uso:
  python -m apuestas.ml.train_soccer_btts_ah --league 4 --market btts
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import poisson
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Market = Literal["btts", "ah_plus05", "ah_minus05", "ah_plus1", "ah_minus1"]


@dataclass(slots=True)
class SoccerMarketConfig:
    league_id: int
    market: Market = "btts"
    seasons: list[str] | None = None


async def fetch_soccer_games(league_id: int) -> list[dict]:
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT m.id, m.home_team_id, m.away_team_id,
                           m.home_score hs, m.away_score as_
                    FROM matches m
                    WHERE m.sport_code='soccer' AND m.league_id=:lg
                      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
                    ORDER BY m.start_time
                    """
                ),
                {"lg": league_id},
            )
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def label_btts(row: dict) -> int:
    return int(row["hs"] >= 1 and row["as_"] >= 1)


def label_ah(row: dict, handicap: float, side: str) -> int:
    """AH win: side cubre con handicap. handicap=+0.5 means team gets +0.5 goals."""
    diff = row["hs"] - row["as_"] if side == "home" else row["as_"] - row["hs"]
    return int((diff + handicap) > 0)


def fit_poisson_btts(games: list[dict]) -> dict:
    """Estima λ_home, λ_away + p_btts via Poisson independiente."""
    if not games:
        return {"n": 0}
    hs = np.array([g["hs"] for g in games])
    as_ = np.array([g["as_"] for g in games])
    lam_h = float(hs.mean())
    lam_a = float(as_.mean())
    p_home_score = 1.0 - float(poisson.pmf(0, lam_h))
    p_away_score = 1.0 - float(poisson.pmf(0, lam_a))
    # Independence assumption (overestimates BTTS slightly)
    p_btts = p_home_score * p_away_score

    y = np.array([label_btts(g) for g in games])
    preds = np.full_like(y, p_btts, dtype=float)
    brier = float(np.mean((preds - y) ** 2))

    return {
        "n": len(games),
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "p_btts_model": p_btts,
        "p_btts_actual": float(y.mean()),
        "brier": brier,
    }


def fit_poisson_ah(games: list[dict], handicap: float = 0.5) -> dict:
    """AH ±handicap para home. Usa Poisson bivariado independiente + labels."""
    if not games:
        return {"n": 0}
    y = np.array([label_ah(g, handicap, "home") for g in games])

    hs = np.array([g["hs"] for g in games])
    as_ = np.array([g["as_"] for g in games])
    lam_h = float(hs.mean())
    lam_a = float(as_.mean())

    # Monte Carlo para p(home covers handicap)
    rng = np.random.default_rng(42)
    n_sim = 20_000
    home_sim = rng.poisson(lam_h, n_sim)
    away_sim = rng.poisson(lam_a, n_sim)
    p_home_cover = float(((home_sim - away_sim + handicap) > 0).mean())

    preds = np.full_like(y, p_home_cover, dtype=float)
    brier = float(np.mean((preds - y) ** 2))

    return {
        "n": len(games),
        "handicap": handicap,
        "p_home_cover_model": p_home_cover,
        "p_home_cover_actual": float(y.mean()),
        "brier": brier,
    }


async def train(cfg: SoccerMarketConfig) -> dict:
    games = await fetch_soccer_games(cfg.league_id)
    logger.info("soccer_market.loaded", n=len(games), market=cfg.market)
    if cfg.market == "btts":
        return fit_poisson_btts(games)
    if cfg.market.startswith("ah_"):
        sign = -1.0 if "minus" in cfg.market else 1.0
        mag = 0.5 if "05" in cfg.market else 1.0
        return fit_poisson_ah(games, handicap=sign * mag)
    return {"error": f"unknown market {cfg.market}"}


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", type=int, required=True)
    ap.add_argument(
        "--market",
        default="btts",
        choices=["btts", "ah_plus05", "ah_minus05", "ah_plus1", "ah_minus1"],
    )
    args = ap.parse_args()

    cfg = SoccerMarketConfig(league_id=args.league, market=args.market)
    r = await train(cfg)
    print(f"Soccer L{args.league} — {args.market}:")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(_main())
