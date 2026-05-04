"""Tests para apuestas.ml.metrics (Sprint 5).

Casos canónicos de Brier/BSS/log-loss/ECE + integración con compute_metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from apuestas.ml.metrics import (
    brier_score,
    brier_skill_score,
    compute_metrics,
    hit_rate,
    implied_rate_from_odds,
    log_loss_binary,
)


def test_brier_perfect_predictor() -> None:
    """p=1 cuando y=1, p=0 cuando y=0 → Brier = 0."""
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, p) == pytest.approx(0.0)


def test_brier_worst_predictor() -> None:
    """Predicciones invertidas → Brier = 1."""
    y = np.array([1, 0])
    p = np.array([0.0, 1.0])
    assert brier_score(y, p) == pytest.approx(1.0)


def test_brier_climatology_matches_definition() -> None:
    """Asignar p̄ a todos da Brier = p̄(1−p̄) cuando y∈{0,1}."""
    y = np.array([1, 1, 0, 0, 0])  # p̄ = 0.4
    p = np.full_like(y, 0.4, dtype=np.float64)
    expected = 0.4 * 0.6
    assert brier_score(y, p) == pytest.approx(expected, rel=1e-6)


def test_brier_skill_score_zero_when_model_matches_climatology() -> None:
    y = np.array([1, 0, 1, 1, 0])
    p_clim = float(y.mean())
    p_model = np.full_like(y, p_clim, dtype=np.float64)
    assert brier_skill_score(y, p_model, p_climatology=p_clim) == pytest.approx(0.0)


def test_brier_skill_score_positive_when_model_beats_baseline() -> None:
    """Modelo perfecto → BSS = 1.0 siempre que BS_climatología > 0."""
    y = np.array([1, 0, 1, 0])
    p_perfect = y.astype(np.float64)
    bss = brier_skill_score(y, p_perfect, p_climatology=0.5)
    assert bss == pytest.approx(1.0)


def test_brier_skill_score_negative_when_model_loses() -> None:
    """Modelo malintencionado → BSS < 0 (peor que climatología)."""
    y = np.array([1, 0, 1, 0])
    p_model = np.array([0.1, 0.9, 0.1, 0.9])  # invertido
    bss = brier_skill_score(y, p_model, p_climatology=0.5)
    assert bss < 0


def test_log_loss_finite_on_extreme_probs() -> None:
    """Clip evita divergencia cuando p=0 o p=1."""
    y = np.array([1, 0])
    p = np.array([1.0, 0.0])  # extremo
    ll = log_loss_binary(y, p)
    assert np.isfinite(ll)


def test_hit_rate_basic() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.3, 0.2, 0.8])
    # threshold 0.5: pred=[1,0,0,1]; correctos: 1,0,1,0 → 2/4
    assert hit_rate(y, p) == pytest.approx(0.5)


def test_implied_rate_from_odds() -> None:
    assert implied_rate_from_odds(2.00) == pytest.approx(0.5)
    assert implied_rate_from_odds(1.91) == pytest.approx(1 / 1.91)
    assert implied_rate_from_odds(0.5) == 0.0  # protección


def test_compute_metrics_integration() -> None:
    """Integración: recibe picks + odds y devuelve todo el dashboard."""
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.70, 0.55, 0.40, 0.65, 0.45, 0.30, 0.60, 0.35])
    r = compute_metrics(y, p, avg_odds=1.85)
    assert r.n == 8
    assert 0 <= r.brier <= 1
    assert 0 <= r.hit_rate <= 1
    assert r.ece >= 0
    assert r.implied_rate == pytest.approx(1 / 1.85)
    assert r.hit_rate_minus_implied == pytest.approx(r.hit_rate - r.implied_rate)


def test_compute_metrics_empty_batch_returns_nan() -> None:
    r = compute_metrics(np.array([]), np.array([]))
    assert r.n == 0
    assert np.isnan(r.brier)
    assert np.isnan(r.hit_rate)
