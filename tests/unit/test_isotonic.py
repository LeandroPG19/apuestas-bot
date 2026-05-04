"""Tests para isotonic calibration Sprint 5."""

from __future__ import annotations

import numpy as np
import pytest

from apuestas.ml.isotonic import calibrated_predict, fit_isotonic_calibrator


def _synthetic_overconfident(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Genera (y, p_raw) donde p_raw es sistemáticamente sobreconfiado.

    Simula comportamiento típico de LGBM binario: pushea probabilidades
    hacia 0 y 1. La tasa real es menor/mayor que la probabilidad anunciada.
    """
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.30, 0.70, size=n)  # prob real calibrada
    # Sobreconfianza: aplana extremos
    p_raw = np.where(p_true > 0.5, p_true + 0.15, p_true - 0.15)
    p_raw = np.clip(p_raw, 0.01, 0.99)
    y = (rng.uniform(size=n) < p_true).astype(np.int64)
    return y, p_raw


def test_fit_returns_isotonic() -> None:
    y, p = _synthetic_overconfident()
    iso = fit_isotonic_calibrator(y, p)
    assert hasattr(iso, "predict")


def test_fit_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_isotonic_calibrator(np.array([0, 1, 0]), np.array([0.5, 0.5]))


def test_calibrated_monotonic_and_bounded() -> None:
    """El calibrador preserva orden monotónico y las probabilidades siguen en [0,1]."""
    y, p = _synthetic_overconfident()
    iso = fit_isotonic_calibrator(y, p)
    grid = np.linspace(0.0, 1.0, 50)
    cal = iso.predict(grid)
    assert cal.min() >= 0.0
    assert cal.max() <= 1.0
    # monotónica no-decreciente
    assert np.all(np.diff(cal) >= -1e-9)


def test_calibrated_predict_without_calibrator_returns_raw() -> None:
    class _Stub:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.column_stack([1.0 - X[:, 0], X[:, 0]])

    X = np.array([[0.3], [0.7]])
    p = calibrated_predict(_Stub(), X, None)
    assert p[0] == pytest.approx(0.3)
    assert p[1] == pytest.approx(0.7)


def test_calibrated_predict_applies_calibrator() -> None:
    y, p = _synthetic_overconfident()
    iso = fit_isotonic_calibrator(y, p)

    # Stub estimator que regresa p[:,1] = X[:,0]
    class _Stub:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.column_stack([1.0 - X[:, 0], X[:, 0]])

    X_test = np.array([[0.2], [0.8]])
    cal = calibrated_predict(_Stub(), X_test, iso)
    assert cal.min() >= 1e-7
    assert cal.max() <= 1.0 - 1e-7
