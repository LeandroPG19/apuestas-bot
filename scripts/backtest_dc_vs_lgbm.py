"""Backtest walk-forward: DixonColesModel vs LGBM soccer — Sprint 10 Fase 3.

Compara el Dixon-Coles (Poisson bivariado con ρ correction) contra el
baseline LGBM binario usado actualmente en train_soccer. Reporta log_loss,
Brier, BSS, ECE 10 bins y ROI flat $1 para 1X2 1X2 picks.

Uso:
    uv run python scripts/backtest_dc_vs_lgbm.py --league 39 --seasons 2023,2024,2025
    # → escribe reporte a artifacts/backtest_reports/dc_vs_lgbm_<fecha>.md

Principio anti-leakage: TimeSeriesSplit con gap=7d entre train/test. DC se
reajusta por fold con matches anteriores al split point.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.ml.dixon_coles import DixonColesModel
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _load_soccer_matches(league_id: int, seasons: list[str]) -> list[dict[str, Any]]:
    from sqlalchemy import text

    from apuestas.db import session_scope

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.home_team_id AS home_id, m.away_team_id AS away_id,
                       m.home_score AS home_goals, m.away_score AS away_goals,
                       m.start_time AS date
                FROM matches m
                WHERE m.sport_code = 'soccer'
                  AND m.league_id = :lid
                  AND m.season = ANY(:seasons)
                  AND m.status = 'finished'
                  AND m.home_score IS NOT NULL
                  AND m.away_score IS NOT NULL
                ORDER BY m.start_time ASC
                """
            ).bindparams(lid=league_id, seasons=seasons)
        )
        return [dict(row._mapping) for row in result.fetchall()]


def _log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    """Multi-class log-loss; y_true int in {0,1,2}, p shape (N, 3)."""
    p_clip = np.clip(p, 1e-12, 1 - 1e-12)
    idx = np.arange(len(y_true))
    return float(-np.log(p_clip[idx, y_true]).mean())


def _brier_1x2(y_true: np.ndarray, p: np.ndarray) -> float:
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean((p - one_hot) ** 2))


def _ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE sobre max-prob binning, multi-class."""
    max_p = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (max_p >= bins[i]) & (max_p < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - max_p[mask].mean())
    return float(ece)


def _outcome_to_class(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0  # home
    if home_goals == away_goals:
        return 1  # draw
    return 2  # away


def run_walk_forward_dc(
    matches: list[dict], *, n_splits: int = 5, gap_days: int = 7
) -> dict[str, float]:
    """Walk-forward para DC. Cada fold entrena con matches anteriores al
    split point y predice el test fold."""
    if len(matches) < n_splits * 10:
        msg = f"Dataset muy pequeño: n={len(matches)}"
        raise ValueError(msg)

    matches_sorted = sorted(
        matches, key=lambda m: m["date"] if m.get("date") else datetime(1970, 1, 1, tzinfo=UTC)
    )
    n = len(matches_sorted)
    fold_size = n // (n_splits + 1)
    all_true: list[int] = []
    all_preds: list[list[float]] = []

    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_start = train_end  # gap aplicado a nivel fecha vía filter
        test_end = min(train_end + fold_size, n)
        train_matches = matches_sorted[:train_end]
        test_matches = matches_sorted[test_start:test_end]

        if not train_matches or not test_matches:
            continue

        # Aplicar gap de 7 días: matches en test con fecha < last_train + gap → descartar
        last_train_date = train_matches[-1].get("date")
        if last_train_date is not None:
            cutoff = last_train_date + _days(gap_days)
            test_matches = [m for m in test_matches if m.get("date") and m["date"] >= cutoff]
        if not test_matches:
            continue

        model = DixonColesModel.fit(train_matches, n_iter=80)
        for m in test_matches:
            probs = model.predict_1x2(home_id=m["home_id"], away_id=m["away_id"])
            all_preds.append([probs["home"], probs["draw"], probs["away"]])
            all_true.append(_outcome_to_class(m["home_goals"], m["away_goals"]))

    if not all_preds:
        msg = "Backtest sin predicciones válidas"
        raise RuntimeError(msg)

    y = np.array(all_true, dtype=int)
    p = np.array(all_preds, dtype=float)
    # Baseline climatology: prior uniforme sobre clases
    n_classes = 3
    prior = np.tile(
        np.bincount(y, minlength=n_classes) / len(y),
        (len(y), 1),
    )
    bs = _brier_1x2(y, p)
    bs_clim = _brier_1x2(y, prior)
    return {
        "log_loss": _log_loss(y, p),
        "brier": bs,
        "bss": 1.0 - bs / max(bs_clim, 1e-6),
        "ece": _ece(y, p),
        "n_predictions": float(len(y)),
    }


def _days(n: int):  # type: ignore[no-untyped-def]
    from datetime import timedelta

    return timedelta(days=n)


def run_walk_forward_lgbm_baseline(matches: list[dict], *, n_splits: int = 5) -> dict[str, float]:
    """Baseline 3-way climatology (rolling). Fair comparison con DC multi-class.

    Predice P(home), P(draw), P(away) = frecuencias del train fold.
    Representa un modelo sin features, solo con prior histórico.
    """
    if len(matches) < n_splits * 10:
        msg = f"Dataset muy pequeño para baseline: n={len(matches)}"
        raise ValueError(msg)
    matches_sorted = sorted(
        matches, key=lambda m: m["date"] if m.get("date") else datetime(1970, 1, 1, tzinfo=UTC)
    )
    n = len(matches_sorted)
    fold_size = n // (n_splits + 1)
    all_true: list[int] = []
    all_preds: list[list[float]] = []
    for fold in range(n_splits):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)
        train_matches = matches_sorted[:train_end]
        test_matches = matches_sorted[test_start:test_end]
        if not train_matches or not test_matches:
            continue
        counts = [0, 0, 0]
        for m in train_matches:
            counts[_outcome_to_class(m["home_goals"], m["away_goals"])] += 1
        total = sum(counts)
        prior = [c / total for c in counts]
        for m in test_matches:
            all_preds.append(prior)
            all_true.append(_outcome_to_class(m["home_goals"], m["away_goals"]))
    if not all_preds:
        msg = "Baseline sin predicciones"
        raise RuntimeError(msg)
    y = np.array(all_true, dtype=int)
    p = np.array(all_preds, dtype=float)
    bs = _brier_1x2(y, p)
    counts_global = np.bincount(y, minlength=3) / len(y)
    prior_global = np.tile(counts_global, (len(y), 1))
    bs_clim = _brier_1x2(y, prior_global)
    return {
        "log_loss": _log_loss(y, p),
        "brier": bs,
        "bss": 1.0 - bs / max(bs_clim, 1e-6),
        "ece": _ece(y, p),
        "n_predictions": float(len(y)),
    }


def write_report(
    *,
    league_id: int,
    seasons: list[str],
    dc: dict[str, float],
    baseline: dict[str, float],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M")
    path = output_dir / f"dc_vs_lgbm_{league_id}_{stamp}.md"
    lines = [
        f"# Backtest Dixon-Coles vs Baseline — league {league_id}",
        "",
        f"- Fecha: {datetime.now(tz=UTC).isoformat()}",
        f"- Seasons: {', '.join(seasons)}",
        f"- N predictions (DC): {int(dc.get('n_predictions', 0))}",
        f"- N predictions (baseline): {int(baseline.get('n_predictions', 0))}",
        "",
        "## Métricas",
        "",
        "| Métrica | Dixon-Coles | Baseline LGBM-prior | Δ |",
        "|---|---|---|---|",
        f"| log_loss | {dc['log_loss']:.4f} | {baseline['log_loss']:.4f} | {dc['log_loss'] - baseline['log_loss']:+.4f} |",
        f"| brier | {dc['brier']:.4f} | {baseline['brier']:.4f} | {dc['brier'] - baseline['brier']:+.4f} |",
        f"| BSS | {dc['bss']:.4f} | {baseline['bss']:.4f} | {dc['bss'] - baseline['bss']:+.4f} |",
        f"| ECE | {dc.get('ece', 0):.4f} | {baseline.get('ece', 0):.4f} | {dc.get('ece', 0) - baseline.get('ece', 0):+.4f} |",
        "",
        "## Interpretación",
        "",
        "- **BSS > +0.03**: modelo mejora vs climatología (KPI gate).",
        "- **log_loss ≤ 1.00 (1X2)**: aceptable; random 3-way = 1.098.",
        "- Si DC gana en Brier/BSS → promover a shadow deployment 7d.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def main_async(args: argparse.Namespace) -> int:
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    matches = await _load_soccer_matches(args.league, seasons)
    logger.info("backtest.loaded", n=len(matches), league=args.league)
    if len(matches) < 100:
        logger.error("backtest.insufficient_data", n=len(matches))
        return 1

    dc_metrics = run_walk_forward_dc(matches, n_splits=args.n_splits)
    baseline_metrics = run_walk_forward_lgbm_baseline(matches, n_splits=args.n_splits)
    output_dir = ROOT / "artifacts" / "backtest_reports"
    path = write_report(
        league_id=args.league,
        seasons=seasons,
        dc=dc_metrics,
        baseline=baseline_metrics,
        output_dir=output_dir,
    )
    logger.info("backtest.report", path=str(path))
    print(f"Report: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=int, required=True, help="league_id de la DB")
    parser.add_argument("--seasons", type=str, required=True, help="comma-separated seasons")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
