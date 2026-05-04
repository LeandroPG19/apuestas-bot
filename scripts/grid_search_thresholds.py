"""Grid-search thresholds EV sobre picks reales settled — Sprint 14 #139.

Barre combinaciones (ev_min, draw_max) sobre `pick_alerts` con outcome_result
settled. Retorna combinación óptima ROI sin overfitting excesivo (ε en sample
pequeño — reporta confidence interval via bootstrap n=500).

Limitación: con 36 picks reales, sample muy chico. Reporta igual + razón.
Útil como VALIDADOR de thresholds propuestos, no óptimo absoluto.

Uso:
  python scripts/grid_search_thresholds.py --since 2026-04-22
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("PYTHON_GIL", None)

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope


async def fetch_settled(since: str):
    since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC)
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT pa.id, pa.market, pa.outcome, pa.line, pa.odds_placed,
                           pa.outcome_result, m.sport_code, m.league_id,
                           p.probability::float p_model, p.ev::float ev
                    FROM pick_alerts pa
                    JOIN matches m ON m.id=pa.match_id
                    LEFT JOIN predictions p ON p.id=pa.prediction_id
                    WHERE pa.placed_at >= :since
                      AND pa.outcome_result IN ('won','lost')
                    """
                ),
                {"since": since_dt},
            )
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def simulate_with_threshold(
    picks: list[dict], ev_thr: float, exclude_mlb_spreads: bool = False
) -> dict:
    """Reemite picks filtrando por (ev>=ev_thr). Calcula hit_rate+ROI."""
    kept = []
    for p in picks:
        if p["ev"] is None or float(p["ev"]) < ev_thr:
            continue
        if exclude_mlb_spreads and p["sport_code"] == "mlb" and p["market"] == "spreads":
            continue
        kept.append(p)
    if not kept:
        return {"n": 0, "hr": 0.0, "roi": 0.0, "profit": 0.0}
    won = sum(1 for p in kept if p["outcome_result"] == "won")
    profit = sum(
        (float(p["odds_placed"]) - 1) if p["outcome_result"] == "won" else -1.0 for p in kept
    )
    return {
        "n": len(kept),
        "won": won,
        "lost": len(kept) - won,
        "hr": won / len(kept),
        "roi": profit / len(kept),
        "profit": profit,
    }


def bootstrap_ci(picks: list[dict], ev_thr: float, excl_spreads: bool, n_boot: int = 500) -> dict:
    """Bootstrap 95% CI para ROI. Indica si mejora es estadísticamente real."""
    rois = []
    n = len(picks)
    if n == 0:
        return {"roi_mean": 0, "roi_ci_low": 0, "roi_ci_high": 0}
    for _ in range(n_boot):
        sample = [picks[i] for i in np.random.randint(0, n, n)]
        r = simulate_with_threshold(sample, ev_thr, excl_spreads)
        if r["n"] > 0:
            rois.append(r["roi"])
    if not rois:
        return {"roi_mean": 0, "roi_ci_low": 0, "roi_ci_high": 0}
    arr = np.array(rois)
    return {
        "roi_mean": float(arr.mean()),
        "roi_ci_low": float(np.percentile(arr, 2.5)),
        "roi_ci_high": float(np.percentile(arr, 97.5)),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-22")
    args = ap.parse_args()

    picks = await fetch_settled(args.since)
    if not picks:
        print("Sin picks settled.")
        return
    print(f"Total picks settled: {len(picks)}\n")

    # Baseline — no filter
    base = simulate_with_threshold(picks, 0.0, False)
    print(
        f"BASELINE (no filter, EV≥0):  n={base['n']:3d}  HR={base['hr']:.3f}  ROI={base['roi']:+.4f}  profit=${base['profit']:+.2f}"
    )
    print()

    # Grid
    print(f"{'ev_thr':>8} {'mlbspr':>6} {'n':>4} {'hr':>6} {'roi':>8} {'profit':>9} {'CI 95%':>24}")
    best = None
    for ev_thr in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12):
        for excl in (False, True):
            r = simulate_with_threshold(picks, ev_thr, excl)
            if r["n"] < 5:  # muy chico
                continue
            ci = bootstrap_ci(picks, ev_thr, excl, n_boot=300)
            print(
                f"{ev_thr:>8.3f} {excl!s:>6} {r['n']:>4d} {r['hr']:>6.3f} "
                f"{r['roi']:>+8.4f} ${r['profit']:>+8.2f} "
                f"[{ci['roi_ci_low']:+.3f}, {ci['roi_ci_high']:+.3f}]"
            )
            if best is None or r["roi"] > best["roi"]:
                best = {"ev_thr": ev_thr, "excl_spreads": excl, **r, **ci}

    if best:
        print("\n" + "=" * 60)
        print(
            f"ÓPTIMO (max ROI): ev_thr={best['ev_thr']}  exclude_mlb_spreads={best['excl_spreads']}"
        )
        print(
            f"  n={best['n']}  hr={best['hr']:.3f}  ROI={best['roi']:+.4f}  profit=${best['profit']:+.2f}"
        )
        print(f"  Bootstrap 95% CI: [{best['roi_ci_low']:+.4f}, {best['roi_ci_high']:+.4f}]")
        sig = (
            "SIGNIFICATIVA (CI no incluye 0)"
            if best["roi_ci_low"] > 0
            else (
                "NO significativa (CI incluye 0)"
                if best["roi_ci_low"] <= 0 <= best["roi_ci_high"]
                else "NEGATIVA significativa"
            )
        )
        print(f"  Delta vs baseline: {best['roi'] - base['roi']:+.4f} — {sig}")
        if len(picks) < 60:
            print(
                f"\n⚠️  SAMPLE SMALL: n={len(picks)} settled. Buchdahl 2023 recomienda ≥65 para CLV. "
                "Resultados NO conclusivos."
            )


if __name__ == "__main__":
    asyncio.run(main())
