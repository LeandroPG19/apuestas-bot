"""Tests para walk_forward_backtest Sprint 5.

El componente de fetch DB se prueba con integración en Sprint 5 real-data.
Aquí probamos las primitivas puras: _time_series_split_purged, _roi_fixed_unit,
y la lógica de passes_mvp_kpis.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from apuestas.backtest.walk_forward import (
    BacktestResult,
    _roi_fixed_unit,
    _time_series_split_purged,
    format_report,
)
from apuestas.ml.metrics import MetricsResult

# ────────────────── Splits purgados ──────────────────


def test_split_produces_n_folds() -> None:
    splits = _time_series_split_purged(n=200, n_splits=5, gap=5)
    assert len(splits) == 5
    for train, test in splits:
        assert len(train) > 0
        assert len(test) > 0
        # Gap respetado: ningún index de train ≥ primer index de test
        assert int(train.max()) < int(test.min())


def test_split_honors_gap() -> None:
    """El último index de train debe quedar ≥ gap posiciones antes del primer test."""
    splits = _time_series_split_purged(n=100, n_splits=4, gap=10)
    for train, test in splits:
        assert int(test.min()) - int(train.max()) >= 10


def test_split_empty_when_not_enough_data() -> None:
    assert _time_series_split_purged(n=5, n_splits=10, gap=2) == []


def test_split_raises_on_invalid_n_splits() -> None:
    with pytest.raises(ValueError):
        _time_series_split_purged(n=100, n_splits=0, gap=0)


# ────────────────── ROI flat ──────────────────


def test_roi_fixed_unit_all_wins_positive() -> None:
    y = np.array([1, 1, 1])
    odds = np.array([2.0, 2.0, 2.0])
    # cada win = +1u (odds-1); ROI promedio = +1.0
    assert _roi_fixed_unit(y, odds) == pytest.approx(1.0)


def test_roi_fixed_unit_all_losses_negative_one() -> None:
    y = np.array([0, 0, 0])
    odds = np.array([1.90, 2.05, 1.80])
    assert _roi_fixed_unit(y, odds) == pytest.approx(-1.0)


def test_roi_fixed_unit_mixed() -> None:
    # 1 win @2.00 (+1), 1 loss (-1) → avg 0
    y = np.array([1, 0])
    odds = np.array([2.00, 1.90])
    assert _roi_fixed_unit(y, odds) == pytest.approx(0.0)


def test_roi_fixed_unit_empty_returns_zero() -> None:
    assert _roi_fixed_unit(np.array([]), np.array([])) == 0.0


# ────────────────── passes_mvp_kpis ──────────────────


def _result_with(metrics: MetricsResult, *, n: int = 50) -> BacktestResult:
    now = datetime.now(tz=UTC)
    return BacktestResult(
        fold=0,
        period_start=now,
        period_end=now,
        metrics=metrics,
        n_picks=n,
        n_won=int(n * metrics.hit_rate),
        n_lost=n - int(n * metrics.hit_rate),
        roi_fixed_unit=0.0,
        avg_odds=1.9,
    )


def test_passes_mvp_all_kpis_green() -> None:
    m = MetricsResult(
        n=50,
        log_loss=0.60,
        brier=0.20,
        brier_skill_score=0.05,
        ece=0.03,
        hit_rate=0.56,
        implied_rate=0.53,
        hit_rate_minus_implied=0.03,
    )
    r = _result_with(m)
    assert r.passes_mvp_kpis("nba") is True


def test_fails_when_brier_too_high() -> None:
    m = MetricsResult(
        n=50,
        log_loss=0.70,
        brier=0.25,  # > 0.22
        brier_skill_score=0.05,
        ece=0.03,
        hit_rate=0.56,
        implied_rate=0.53,
        hit_rate_minus_implied=0.03,
    )
    assert _result_with(m).passes_mvp_kpis("nba") is False


def test_fails_when_bss_not_positive_enough() -> None:
    m = MetricsResult(
        n=50,
        log_loss=0.60,
        brier=0.20,
        brier_skill_score=0.01,  # < 0.03
        ece=0.03,
        hit_rate=0.56,
        implied_rate=0.53,
        hit_rate_minus_implied=0.03,
    )
    assert _result_with(m).passes_mvp_kpis("nba") is False


def test_fails_when_hit_rate_not_above_implied_2pp() -> None:
    m = MetricsResult(
        n=50,
        log_loss=0.60,
        brier=0.20,
        brier_skill_score=0.05,
        ece=0.03,
        hit_rate=0.54,
        implied_rate=0.53,
        hit_rate_minus_implied=0.01,  # < 0.02
    )
    assert _result_with(m).passes_mvp_kpis("nba") is False


# ────────────────── Reporte ──────────────────


def test_format_report_empty() -> None:
    report = format_report([], sport="nba")
    assert "No hay datos" in report


def test_format_report_non_empty_has_table_and_aggregates() -> None:
    m = MetricsResult(
        n=10,
        log_loss=0.60,
        brier=0.22,
        brier_skill_score=0.04,
        ece=0.05,
        hit_rate=0.55,
        implied_rate=0.52,
        hit_rate_minus_implied=0.03,
    )
    report = format_report([_result_with(m, n=10), _result_with(m, n=10)], sport="nba")
    assert "Backtest" in report
    assert "Brier" in report
    assert "Agregados" in report
