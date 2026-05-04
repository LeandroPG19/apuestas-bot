"""Tests para MarketAwareStacker — Sprint 10 Fase 2 (Mejora #2)."""

from __future__ import annotations

import numpy as np
import pytest

from apuestas.ml.stacker import MarketAwareStacker


@pytest.fixture
def toy_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    n = 500
    oof_lgbm = rng.beta(2, 2, n)
    oof_xgb = rng.beta(2, 2, n)
    oof_cat = rng.beta(2, 2, n)
    market_delta = rng.uniform(0, 0.08, n)
    line_vel = rng.normal(0, 0.02, n)
    sharp_agree = rng.integers(0, 5, n).astype(float)
    X = np.column_stack([oof_lgbm, oof_xgb, oof_cat, market_delta, line_vel, sharp_agree])
    # target: mezcla ruidosa de los OOFs (el stacker debe recuperar señal)
    y = (0.4 * oof_lgbm + 0.3 * oof_xgb + 0.3 * oof_cat + rng.normal(0, 0.1, n) > 0.5).astype(int)
    names = [
        "oof_lgbm",
        "oof_xgb",
        "oof_cat",
        "market_consensus_delta",
        "line_movement_velocity",
        "sharp_book_agreement",
    ]
    return X, y, names


def test_stacker_fits_and_predicts(toy_data: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    X, y, names = toy_data
    s = MarketAwareStacker(use_lgbm=True, monotonic=True)
    s.fit(X, y, feature_names=names)
    probs = s.predict_proba(X)
    assert probs.shape == (500, 2)
    # Suma de probas = 1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_stacker_monotonic_constraints(
    toy_data: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    X, y, names = toy_data
    s = MarketAwareStacker(use_lgbm=True, monotonic=True)
    s.fit(X, y, feature_names=names)
    constraints = s.monotonic_constraints
    # oof_lgbm / oof_xgb / oof_cat → +1
    for i in range(3):
        assert constraints[i] == 1, f"OOF feature {names[i]} debería ser +1"
    # market_consensus_delta → -1
    assert constraints[3] == -1
    # sharp_book_agreement → +1
    assert constraints[5] == 1


def test_stacker_predict_before_fit_raises() -> None:
    s = MarketAwareStacker()
    with pytest.raises(RuntimeError, match="fit"):
        s.predict_proba(np.zeros((1, 3)))


def test_stacker_logreg_fallback(toy_data: tuple[np.ndarray, np.ndarray, list[str]]) -> None:
    X, y, names = toy_data
    s = MarketAwareStacker(use_lgbm=False)
    s.fit(X, y, feature_names=names)
    probs = s.predict_proba(X)
    assert probs.shape == (500, 2)


def test_stacker_without_feature_names(
    toy_data: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    X, y, _ = toy_data
    s = MarketAwareStacker(use_lgbm=True, monotonic=False)
    s.fit(X, y)  # sin feature_names → genera f0, f1, ...
    assert s.feature_names == [f"f{i}" for i in range(X.shape[1])]
