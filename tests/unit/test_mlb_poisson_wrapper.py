"""Tests para MLBPoissonSklearnWrapper — Sprint 10 Fase 3."""

from __future__ import annotations

import numpy as np
import pytest

from apuestas.ml.mlb_poisson import MLBPoissonSklearnWrapper


def _toy_matches() -> list[dict]:
    return [
        {"home_id": 1, "away_id": 2, "home_runs": 7, "away_runs": 2},
        {"home_id": 1, "away_id": 2, "home_runs": 5, "away_runs": 3},
        {"home_id": 2, "away_id": 1, "home_runs": 2, "away_runs": 6},
        {"home_id": 1, "away_id": 3, "home_runs": 8, "away_runs": 4},
        {"home_id": 3, "away_id": 1, "home_runs": 3, "away_runs": 7},
    ] * 10


def test_wrapper_has_sklearn_interface() -> None:
    w = MLBPoissonSklearnWrapper()
    assert hasattr(w, "fit")
    assert hasattr(w, "predict_proba")
    assert hasattr(w, "predict")
    assert w.classes_.tolist() == [0, 1]


def test_wrapper_fit_with_matches_kwarg() -> None:
    w = MLBPoissonSklearnWrapper(n_iter=30)
    w.fit(matches=_toy_matches())
    assert w.model_ is not None


def test_wrapper_predict_proba_shape() -> None:
    w = MLBPoissonSklearnWrapper(n_iter=30)
    w.fit(matches=_toy_matches())
    X = np.array([[1, 2, 0, 0, ""]], dtype=object)
    probs = w.predict_proba(X)
    assert probs.shape == (1, 2)
    assert probs[0, 0] + probs[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_wrapper_predict_before_fit_raises() -> None:
    w = MLBPoissonSklearnWrapper()
    with pytest.raises(RuntimeError, match="fit"):
        w.predict_proba(np.array([[1, 2, 0, 0, ""]], dtype=object))


def test_wrapper_fit_from_x_array() -> None:
    # Formato X: cada row = (home_id, away_id, home_runs, away_runs, venue)
    X = np.array(
        [[1, 2, 7, 2, ""], [1, 2, 5, 3, ""], [2, 1, 2, 6, ""]] * 10,
        dtype=object,
    )
    y = np.array([1, 1, 0] * 10, dtype=int)
    w = MLBPoissonSklearnWrapper(n_iter=30)
    w.fit(X, y)
    assert w.model_ is not None


def test_wrapper_fit_empty_raises() -> None:
    w = MLBPoissonSklearnWrapper()
    with pytest.raises(ValueError, match="vacío"):
        w.fit(matches=[])


def test_wrapper_predict_binary() -> None:
    w = MLBPoissonSklearnWrapper(n_iter=30)
    w.fit(matches=_toy_matches())
    X = np.array([[1, 2, 0, 0, ""], [3, 1, 0, 0, ""]], dtype=object)
    y_pred = w.predict(X)
    assert y_pred.shape == (2,)
    assert set(y_pred.tolist()).issubset({0, 1})


def test_wrapper_get_set_params() -> None:
    w = MLBPoissonSklearnWrapper(n_iter=50, learning_rate=0.02)
    params = w.get_params()
    assert params["n_iter"] == 50
    w2 = w.set_params(n_iter=10)
    assert w2.n_iter == 10
