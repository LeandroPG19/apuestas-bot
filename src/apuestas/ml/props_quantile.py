"""Player props quantile regression — Sprint 14 #159, #143.

Distributional output para props (puntos NBA, Ks MLB, rebs, asts). Quantile
regression predice q10/q25/q50/q75/q90 — mejor que point estimate para EV
edges sobre líneas over/under específicas.

Outperforms accuracy-optimized models para props (Walsh & Joshi 2024):
  - fit 5 quantile regressors LGBM (objective='quantile')
  - predict (q10, q25, q50, q75, q90) → aproxima CDF del prop
  - EV sobre línea L: p(prop > L) = interp del CDF

Uso:
  python -m apuestas.ml.props_quantile --sport nba --prop points --player 2544
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass(slots=True)
class PropsConfig:
    sport: str
    prop: str = "points"
    quantiles: tuple[float, ...] = QUANTILES


def predict_prob_over(quantile_preds: dict[float, float], line: float) -> float:
    """Linear interp entre quantiles para estimar p(prop > line).

    quantile_preds: {q: value_predicted_at_q}
      - si line < q10: p_over > 0.90
      - si line > q90: p_over < 0.10
      - else: interp linear
    """
    qs = sorted(quantile_preds.keys())
    vals = [quantile_preds[q] for q in qs]
    if line <= vals[0]:
        return 1.0 - qs[0]
    if line >= vals[-1]:
        return 1.0 - qs[-1]
    # Find surrounding quantiles
    for i in range(len(qs) - 1):
        if vals[i] <= line <= vals[i + 1]:
            frac = (line - vals[i]) / (vals[i + 1] - vals[i])
            q_at_line = qs[i] + frac * (qs[i + 1] - qs[i])
            return 1.0 - q_at_line
    return 0.5


async def fetch_nba_player_history(player_id: int, stat: str = "points") -> list[float]:
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    f"""
                    SELECT {stat} FROM nba_player_game_stats
                    WHERE player_id=:pid AND {stat} IS NOT NULL
                    ORDER BY game_date DESC LIMIT 50
                    """
                ),
                {"pid": player_id},
            )
        ).fetchall()
    return [float(r[0]) for r in rows]


def fit_empirical_quantiles(history: list[float], quantiles=QUANTILES) -> dict[float, float]:
    """Empirical quantiles — baseline sin features. Replace con LGBM quantile.

    Para producción real requiere features matchup + opponent defense rating
    + pace + player_rest + back-to-back + home/away split + minutes projected.
    """
    if not history:
        return dict.fromkeys(quantiles, 0.0)
    arr = np.array(history)
    return {q: float(np.quantile(arr, q)) for q in quantiles}


async def predict_prop_over_under(
    *, sport: str, player_id: int, prop: str, line: float
) -> dict[str, Any]:
    """Returns p_over + p_under para línea dada."""
    if sport != "nba":
        return {"error": "only_nba_supported_now"}
    history = await fetch_nba_player_history(player_id, prop)
    if not history:
        return {"error": "no_history"}
    q_preds = fit_empirical_quantiles(history)
    p_over = predict_prob_over(q_preds, line)
    return {
        "player_id": player_id,
        "prop": prop,
        "line": line,
        "p_over": round(p_over, 4),
        "p_under": round(1.0 - p_over, 4),
        "quantiles": {str(k): round(v, 2) for k, v in q_preds.items()},
        "n_history": len(history),
        "mean_historical": float(np.mean(history)),
    }


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nba")
    ap.add_argument("--prop", default="points")
    ap.add_argument("--player", type=int, required=True)
    ap.add_argument("--line", type=float, required=True)
    args = ap.parse_args()

    r = await predict_prop_over_under(
        sport=args.sport, player_id=args.player, prop=args.prop, line=args.line
    )
    print(f"Props {args.sport}/{args.prop} player={args.player} line={args.line}:")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(_main())
