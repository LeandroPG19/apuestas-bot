"""Walk-forward backtesting con TimeSeriesSplit purgado (gap=7d).

Plan §7.1 / Bergmeir & Benítez 2012. Pipeline:
  1. Toma todas las alertas resueltas (`outcome_result IN ('won','lost')`)
     del deporte y rango solicitado.
  2. Ordena por `result_settled_at` ascendente.
  3. Divide en N splits temporales (TimeSeriesSplit) con gap de `gap_days`
     entre train y test para evitar leakage por dependencia intra-semana.
  4. Para cada split de test, calcula Brier/BSS/ECE/hit_rate/log_loss.
  5. Retorna `list[BacktestResult]` + reporte agregado.

No requiere MLflow — opera sobre `pick_alerts + predictions` existentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ml.metrics import MetricsResult, compute_metrics
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Resultado de un fold del walk-forward."""

    fold: int
    period_start: datetime
    period_end: datetime
    metrics: MetricsResult
    n_picks: int
    n_won: int
    n_lost: int
    roi_fixed_unit: float  # ROI de $1-flat por pick (sin Kelly)
    avg_odds: float

    def passes_mvp_kpis(self, sport: str = "nba") -> bool:
        """True si cumple los 4 KPIs obligatorios de MLflow promotion (plan §7.2).

        - brier ≤ 0.22 (NBA) / 0.23 (NFL); 0.24 conservador para otros.
        - BSS ≥ 0.03
        - ECE ≤ 0.05
        - hit_rate − implied_rate ≥ +2 pp
        """
        brier_cap = {"nba": 0.22, "nfl": 0.23}.get(sport.lower(), 0.24)
        m = self.metrics
        return (
            m.brier <= brier_cap
            and m.brier_skill_score >= 0.03
            and m.ece <= 0.05
            and m.hit_rate_minus_implied >= 0.02
        )


async def _fetch_resolved_picks(
    sport: str,
    start_date: datetime,
    end_date: datetime,
) -> list[dict[str, Any]]:
    """Trae picks con outcome_result ∈ {won,lost} del rango, ordenados por fecha."""
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT pa.id,
                           pa.market, pa.outcome, pa.line,
                           pa.odds_placed, pa.outcome_result,
                           pa.result_settled_at,
                           p.probability AS p_model,
                           m.sport_code, m.start_time
                    FROM pick_alerts pa
                    LEFT JOIN predictions p ON p.id = pa.prediction_id
                    JOIN matches m ON m.id = pa.match_id
                    WHERE m.sport_code = :sport
                      AND pa.outcome_result IN ('won', 'lost')
                      AND pa.result_settled_at BETWEEN :start AND :end
                    ORDER BY pa.result_settled_at ASC
                    """
                ),
                {"sport": sport, "start": start_date, "end": end_date},
            )
        ).all()
    return [dict(r._mapping) for r in rows]


def _time_series_split_purged(
    n: int, n_splits: int, gap: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit manual con gap purgado entre train/test.

    Standard sklearn TimeSeriesSplit no tiene gap; lo implementamos aquí.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        return []
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_start = train_end + gap
        test_end = test_start + fold_size
        if test_start >= n:
            break
        test_end = min(test_end, n)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _roi_fixed_unit(y: np.ndarray, odds: np.ndarray) -> float:
    """PnL de $1 flat por pick: won → (odds − 1), lost → −1. Sin stakes."""
    n = len(y)
    if n == 0:
        return 0.0
    profit = np.where(y == 1, odds - 1.0, -1.0)
    return float(profit.sum() / n)


async def walk_forward_backtest(
    *,
    sport: str,
    start_date: datetime,
    end_date: datetime,
    n_splits: int = 10,
    gap_days: int = 7,
) -> list[BacktestResult]:
    """Ejecuta backtesting walk-forward sobre las alertas resueltas del deporte.

    Returns: lista de BacktestResult, uno por fold de test.
    """
    picks = await _fetch_resolved_picks(sport, start_date, end_date)
    if not picks:
        logger.info("backtest.no_picks", sport=sport, start=start_date, end=end_date)
        return []

    # Arrays: y=1 si won, odds placed, p_model (fallback 0.5 si NULL)
    y = np.array([1 if p["outcome_result"] == "won" else 0 for p in picks], dtype=np.int8)
    odds = np.array([float(p["odds_placed"] or 0) for p in picks], dtype=np.float64)
    p_model_raw = np.array(
        [float(p["p_model"]) if p["p_model"] is not None else 0.5 for p in picks],
        dtype=np.float64,
    )
    dates = np.array([p["result_settled_at"] for p in picks])

    splits = _time_series_split_purged(len(y), n_splits=n_splits, gap=gap_days)
    if not splits:
        logger.warning(
            "backtest.insufficient_samples",
            n=len(y),
            n_splits=n_splits,
            required=(n_splits + 1) * (gap_days + 1),
        )
        return []

    # Baseline climatológico = tasa base del train del primer split.
    # Esto evita que BSS se auto-infle por usar el propio test como referencia.
    p_clim_baseline = float(y[splits[0][0]].mean()) if len(splits[0][0]) > 0 else 0.5

    results: list[BacktestResult] = []
    for i, (_train_idx, test_idx) in enumerate(splits):
        if len(test_idx) == 0:
            continue
        y_test = y[test_idx]
        p_test = p_model_raw[test_idx]
        odds_test = odds[test_idx]

        avg_odds = float(odds_test[odds_test > 1.0].mean()) if (odds_test > 1.0).any() else 0.0
        m = compute_metrics(
            y_test,
            p_test,
            avg_odds=avg_odds or None,
            p_climatology=p_clim_baseline,
        )
        result = BacktestResult(
            fold=i,
            period_start=dates[test_idx[0]],
            period_end=dates[test_idx[-1]],
            metrics=m,
            n_picks=len(test_idx),
            n_won=int(y_test.sum()),
            n_lost=int(len(y_test) - y_test.sum()),
            roi_fixed_unit=_roi_fixed_unit(y_test, odds_test),
            avg_odds=avg_odds,
        )
        results.append(result)

    logger.info(
        "backtest.done",
        sport=sport,
        n_picks=len(y),
        n_folds=len(results),
        mean_brier=float(np.mean([r.metrics.brier for r in results])),
        mean_bss=float(np.mean([r.metrics.brier_skill_score for r in results])),
    )
    return results


def format_report(results: list[BacktestResult], *, sport: str) -> str:
    """Genera reporte markdown con la tabla de folds + resumen."""
    if not results:
        return f"# Backtest {sport}\n\nNo hay datos suficientes.\n"
    lines: list[str] = [
        f"# Backtest walk-forward — {sport.upper()}",
        "",
        f"Folds: **{len(results)}** · Total picks: **{sum(r.n_picks for r in results)}**",
        "",
        "| Fold | n | won | lost | avg_odds | Brier | BSS | ECE | hit_rate | HR−implied | ROI$1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.fold} | {r.n_picks} | {r.n_won} | {r.n_lost} | {r.avg_odds:.2f} | "
            f"{m.brier:.4f} | {m.brier_skill_score:+.4f} | {m.ece:.4f} | "
            f"{m.hit_rate:.3f} | {m.hit_rate_minus_implied:+.3f} | "
            f"{r.roi_fixed_unit:+.4f} |"
        )

    # Agregados
    briers = np.array([r.metrics.brier for r in results])
    bsss = np.array([r.metrics.brier_skill_score for r in results])
    eces = np.array([r.metrics.ece for r in results])
    passed = [r.passes_mvp_kpis(sport) for r in results]
    pass_rate = sum(passed) / max(len(passed), 1)

    lines.extend(
        [
            "",
            "## Agregados (mean ± std)",
            "",
            f"- Brier: `{briers.mean():.4f} ± {briers.std():.4f}`",
            f"- BSS:   `{bsss.mean():+.4f} ± {bsss.std():.4f}`",
            f"- ECE:   `{eces.mean():.4f} ± {eces.std():.4f}`",
            f"- Folds pasando KPIs MVP: **{int(sum(passed))}/{len(passed)}** ({pass_rate:.0%})",
            "",
        ]
    )
    return "\n".join(lines)


# Para que Decimal no moleste en mypy/ruff (reservado)
_DEC = Decimal
