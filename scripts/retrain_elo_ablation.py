"""Ablation Elo on/off — Sprint 10 Fase 2 validation.

Re-entrena un modelo por deporte CON y SIN Elo features y reporta
Brier / BSS / ECE / log_loss sobre el holdout. Decide si Elo aporta
lift estadísticamente significativo antes de promote a production.

Uso:
    uv run python scripts/retrain_elo_ablation.py --sport nba --years 2023,2024,2025
    # → artifacts/elo_ablation_<sport>_<fecha>.md

KPI gate aplicado:
- Lift Brier ≥ 0.005 → recomienda Elo ON en production
- Lift Brier < 0.005 pero ≥ 0 → Elo ON en shadow
- Lift Brier < 0 → NO promover Elo
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _run_nba(seasons: list[str]) -> dict[str, Any]:
    from apuestas.ml.train_nba import NBATrainConfig, train_nba

    cfg = NBATrainConfig(seasons=seasons, target="win", n_trials=10)
    return await train_nba(cfg)


async def _run_mlb(years: list) -> dict[str, Any]:
    from apuestas.ml.train_mlb import MLBTrainConfig, train_mlb

    years_int = [int(y) for y in years]
    cfg = MLBTrainConfig(years=years_int, target="moneyline", n_trials=10)
    return await train_mlb(cfg)


async def _run_nfl(seasons: list[str]) -> dict[str, Any]:
    from apuestas.ml.train_nfl import NFLTrainConfig, train_nfl

    cfg = NFLTrainConfig(seasons=seasons, n_trials=10)
    return await train_nfl(cfg)


_RUNNERS = {
    "nba": _run_nba,
    "mlb": _run_mlb,
    "nfl": _run_nfl,
}


def _result_to_metrics(result: Any) -> dict[str, float]:
    """Extrae métricas de TrainResult regardless de sport."""
    if result is None:
        return {}
    # TrainResult shape o dict
    candidates = (
        "holdout_log_loss",
        "holdout_brier",
        "holdout_ece",
        "cv_log_loss",
        "cv_brier",
        "cv_ece",
    )
    return {k: float(getattr(result, k, 0.0)) for k in candidates}


async def run_ablation(sport: str, seasons: list[str]) -> dict[str, dict[str, float]]:
    """Corre el pipeline 2 veces: sin Elo y con Elo.

    Usa env var `APUESTAS_ELO_FEATURES_DISABLED=true` para saltarse add_elo_features.
    Requiere que el trainer respete esa flag (si no, ambas corridas serán iguales).
    """
    runner = _RUNNERS[sport]

    logger.info("ablation.baseline_start", sport=sport, seasons=seasons)
    os.environ["APUESTAS_ELO_FEATURES_DISABLED"] = "true"
    baseline = await runner(seasons)  # type: ignore[arg-type]
    baseline_metrics = _result_to_metrics(baseline)

    logger.info("ablation.elo_start", sport=sport)
    os.environ.pop("APUESTAS_ELO_FEATURES_DISABLED", None)
    elo = await runner(seasons)  # type: ignore[arg-type]
    elo_metrics = _result_to_metrics(elo)

    return {"baseline": baseline_metrics, "elo": elo_metrics}


def kpi_gate_decision(baseline: dict[str, float], elo: dict[str, float]) -> str:
    lift = baseline.get("holdout_brier", 1.0) - elo.get("holdout_brier", 1.0)
    if lift >= 0.005:
        return f"PROMOTE: Elo ON en production (lift Brier +{lift:.4f})"
    if lift > 0:
        return f"SHADOW: Elo ON en shadow deployment 7d (lift marginal +{lift:.4f})"
    return f"REJECT: NO promover Elo (lift {lift:+.4f})"


def write_report(sport: str, results: dict[str, dict[str, float]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M")
    path = output_dir / f"elo_ablation_{sport}_{stamp}.md"
    baseline = results["baseline"]
    elo = results["elo"]
    decision = kpi_gate_decision(baseline, elo)
    lines = [
        f"# Ablation Elo on/off — {sport.upper()}",
        f"- Fecha: {datetime.now(tz=UTC).isoformat()}",
        "",
        "## Métricas holdout",
        "",
        "| Métrica | Baseline (sin Elo) | Con Elo | Δ |",
        "|---|---|---|---|",
    ]
    for metric in ("holdout_log_loss", "holdout_brier", "holdout_ece"):
        b = baseline.get(metric, 0.0)
        e = elo.get(metric, 0.0)
        lines.append(f"| {metric} | {b:.4f} | {e:.4f} | {e - b:+.4f} |")
    lines += [
        "",
        "## Decisión KPI gate",
        "",
        f"**{decision}**",
        "",
        "Criterio: lift Brier ≥ 0.005 → production; 0 ≤ lift < 0.005 → shadow; lift < 0 → reject.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def main_async(args: argparse.Namespace) -> int:
    if args.sport not in _RUNNERS:
        logger.error("ablation.unsupported_sport", sport=args.sport)
        return 1
    seasons = [s.strip() for s in args.years.split(",") if s.strip()]
    results = await run_ablation(args.sport, seasons)
    output_dir = ROOT / "artifacts" / "elo_ablation"
    path = write_report(args.sport, results, output_dir)
    logger.info("ablation.report", path=str(path))
    print(f"Report: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", type=str, required=True, choices=["nba", "mlb", "nfl"])
    parser.add_argument("--years", type=str, required=True, help="comma-separated")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
