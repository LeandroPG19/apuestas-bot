"""Tests de módulo calibración."""

from __future__ import annotations

import numpy as np

from apuestas.ml.calibrate import (
    ConformalClassifier,
    compute_calibration_metrics,
    expected_calibration_error,
    select_calibration_method,
)


def test_ece_perfectly_calibrated() -> None:
    """Si p=true_rate en cada bin, ECE debe ser cerca de 0."""
    rng = np.random.default_rng(42)
    n = 5000
    y_prob = rng.uniform(0.01, 0.99, n)
    # Generar y con Bernoulli(p) → perfectamente calibrado
    y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05, f"ECE demasiado alto para sample perfecto: {ece}"


def test_ece_miscalibrated_is_high() -> None:
    """Si siempre predice 0.9 pero hit rate real es 0.5, ECE debe ser alto."""
    y_prob = np.full(1000, 0.9)
    y_true = np.concatenate([np.ones(500), np.zeros(500)])
    np.random.default_rng(0).shuffle(y_true)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece > 0.3


def test_select_method_by_sample_size() -> None:
    assert select_calibration_method(2000) == "isotonic"
    assert select_calibration_method(500) == "sigmoid"
    assert select_calibration_method(50) == "venn_abers"


def test_compute_calibration_metrics_binary() -> None:
    rng = np.random.default_rng(7)
    n = 1000
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)
    m = compute_calibration_metrics(y_true, y_prob)
    assert m.n_samples == n
    assert 0 < m.log_loss < 2
    assert 0 < m.brier < 0.5
    assert m.ece >= 0.0


def test_conformal_manual_fallback_produces_valid_intervals() -> None:
    """Sin MAPIE, el fallback manual debe dar p_low ≤ p_point ≤ p_up."""

    class DummyEstimator:
        classes_ = np.array([0, 1])

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            p = np.full(len(x), 0.6)
            return np.vstack([1 - p, p]).T

    rng = np.random.default_rng(1)
    X_cal = rng.normal(size=(100, 3))
    y_cal = rng.integers(0, 2, 100)
    X_test = rng.normal(size=(20, 3))

    est = DummyEstimator()
    conformal = ConformalClassifier(alpha=0.1)
    # Forzamos el fallback manual: simulamos que MAPIE falla
    import apuestas.ml.calibrate as cal_mod

    real_mapie = cal_mod.__dict__.get("MapieClassifier", None)
    try:
        if real_mapie is not None:
            del cal_mod.__dict__["MapieClassifier"]
        conformal.fit(est, X_cal, y_cal)
    finally:
        if real_mapie is not None:
            cal_mod.__dict__["MapieClassifier"] = real_mapie

    p, p_low, p_up = conformal.predict_intervals(est, X_test)
    assert all(0 <= x <= 1 for x in p_low)
    assert all(0 <= x <= 1 for x in p_up)
    assert all(p_low <= p)
    assert all(p <= p_up)


def test_conformal_is_confident_filter() -> None:
    """La API is_confident debe filtrar picks con intervalo que incluye implied_prob."""
    c = ConformalClassifier(alpha=0.1)
    # implied_prob = 0.50 (odds 2.00). p_low=0.55 con margen 0.01 → confident.
    assert c.is_confident(p_low=0.55, implied_prob=0.50, margin=0.01) is True
    # p_low=0.505 con margen 0.01 → NO confident (no supera 0.51)
    assert c.is_confident(p_low=0.505, implied_prob=0.50, margin=0.01) is False
