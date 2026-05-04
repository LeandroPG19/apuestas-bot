"""Tests para DixonColesCrossLeagueModel sklearn wrapper.

Cubre h2h/totals/btts con teams sin strength (uniform fallback) y mock con
strength conocida. No requiere DB; mockea `dixon_coles_predict` y similares.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from apuestas.ml.dixon_coles_runtime import DixonColesCrossLeagueModel


def test_h2h_no_strength_returns_uniform() -> None:
    """Teams sin strength → DC retorna None → wrapper devuelve uniforme."""
    with patch("apuestas.ml.dixon_coles_runtime.dixon_coles_predict", return_value=None):
        m = DixonColesCrossLeagueModel()
        probs = m.predict_proba(np.array([[1, 2]]))
        assert probs.shape == (1, 3)
        assert abs(probs[0].sum() - 1.0) < 0.01


def test_h2h_with_strength_orders_correctly() -> None:
    """Wrapper retorna [away, draw, home] en ese orden (sklearn classes_)."""
    fake_pred = {"p_away": 0.20, "p_draw": 0.30, "p_home": 0.50}
    with patch("apuestas.ml.dixon_coles_runtime.dixon_coles_predict", return_value=fake_pred):
        m = DixonColesCrossLeagueModel()
        probs = m.predict_proba(np.array([[1, 2]]))
        assert abs(probs[0, 0] - 0.20) < 1e-6  # away
        assert abs(probs[0, 1] - 0.30) < 1e-6  # draw
        assert abs(probs[0, 2] - 0.50) < 1e-6  # home


def test_totals_returns_under_over_order() -> None:
    fake_pred = {"under": 0.40, "over": 0.60}
    with patch("apuestas.ml.dixon_coles_runtime.dixon_coles_predict_total", return_value=fake_pred):
        m = DixonColesCrossLeagueModel()
        probs = m.predict_proba_total(np.array([[1, 2]]), line=2.5)
        assert abs(probs[0, 0] - 0.40) < 1e-6  # under
        assert abs(probs[0, 1] - 0.60) < 1e-6  # over


def test_totals_no_strength_returns_uniform() -> None:
    with patch("apuestas.ml.dixon_coles_runtime.dixon_coles_predict_total", return_value=None):
        m = DixonColesCrossLeagueModel()
        probs = m.predict_proba_total(np.array([[1, 2]]), line=2.5)
        assert probs[0, 0] == 0.5
        assert probs[0, 1] == 0.5


def test_btts_returns_no_yes_order() -> None:
    fake_pred = {"no": 0.45, "yes": 0.55}
    with patch("apuestas.ml.dixon_coles_runtime.dixon_coles_predict_btts", return_value=fake_pred):
        m = DixonColesCrossLeagueModel()
        probs = m.predict_proba_btts(np.array([[1, 2]]))
        assert abs(probs[0, 0] - 0.45) < 1e-6  # no
        assert abs(probs[0, 1] - 0.55) < 1e-6  # yes


def test_predict_returns_argmax_class_name() -> None:
    fake_pred = {"p_away": 0.10, "p_draw": 0.20, "p_home": 0.70}
    with patch("apuestas.ml.dixon_coles_runtime.dixon_coles_predict", return_value=fake_pred):
        m = DixonColesCrossLeagueModel()
        result = m.predict(np.array([[1, 2]]))
        assert result[0] == "home"


def test_extract_ids_handles_malformed_rows() -> None:
    m = DixonColesCrossLeagueModel()
    # Single-element row → fallback (0, 0)
    ids = m._extract_ids(np.array([[1]], dtype=object))
    assert ids == [(0, 0)]


def test_batch_prediction() -> None:
    fake_pred = {"p_away": 0.3, "p_draw": 0.3, "p_home": 0.4}
    with patch("apuestas.ml.dixon_coles_runtime.dixon_coles_predict", return_value=fake_pred):
        m = DixonColesCrossLeagueModel()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        probs = m.predict_proba(X)
        assert probs.shape == (3, 3)
        for i in range(3):
            assert abs(probs[i].sum() - 1.0) < 1e-6
