"""Tests para DixonColesModel — Sprint 10 (Mejora #1)."""

from __future__ import annotations

import pytest

from apuestas.ml.dixon_coles import DixonColesModel


def _toy_matches() -> list[dict]:
    """Dataset sintético: equipo 1 siempre gana 2-0, equipo 2 pierde."""
    return [
        {"home_id": 1, "away_id": 2, "home_goals": 2, "away_goals": 0},
        {"home_id": 1, "away_id": 2, "home_goals": 3, "away_goals": 1},
        {"home_id": 2, "away_id": 1, "home_goals": 0, "away_goals": 2},
        {"home_id": 1, "away_id": 3, "home_goals": 2, "away_goals": 0},
        {"home_id": 3, "away_id": 1, "home_goals": 1, "away_goals": 3},
        {"home_id": 2, "away_id": 3, "home_goals": 1, "away_goals": 1},
    ] * 5  # repetido para más señal


def test_fit_returns_model() -> None:
    model = DixonColesModel.fit(_toy_matches(), n_iter=50)
    assert model.mu > 0
    assert 1 in model.alpha
    assert 2 in model.alpha


def test_strong_team_has_higher_alpha() -> None:
    model = DixonColesModel.fit(_toy_matches(), n_iter=100)
    # Equipo 1 (atacante fuerte) debe tener α mayor que equipo 2
    assert model.alpha[1] > model.alpha[2]


def test_predict_1x2_sums_to_one() -> None:
    model = DixonColesModel.fit(_toy_matches(), n_iter=50)
    probs = model.predict_1x2(home_id=1, away_id=2)
    assert set(probs.keys()) == {"home", "draw", "away"}
    assert probs["home"] + probs["draw"] + probs["away"] == pytest.approx(1.0, abs=1e-6)


def test_strong_home_wins_probability_higher() -> None:
    model = DixonColesModel.fit(_toy_matches(), n_iter=100)
    probs = model.predict_1x2(home_id=1, away_id=2)
    assert probs["home"] > probs["away"]


def test_predict_total_sums_to_one() -> None:
    model = DixonColesModel.fit(_toy_matches(), n_iter=50)
    r = model.predict_total(home_id=1, away_id=2, line=2.5)
    assert set(r.keys()) == {"over", "under"}
    assert r["over"] + r["under"] == pytest.approx(1.0, abs=1e-6)


def test_predict_btts() -> None:
    model = DixonColesModel.fit(_toy_matches(), n_iter=50)
    r = model.predict_btts(home_id=1, away_id=2)
    assert 0.0 <= r["yes"] <= 1.0
    assert r["yes"] + r["no"] == pytest.approx(1.0, abs=1e-6)


def test_fit_empty_raises() -> None:
    with pytest.raises(ValueError, match="sin matches"):
        DixonColesModel.fit([], n_iter=5)


def test_rho_correction_affects_low_scores() -> None:
    model = DixonColesModel(mu=0.5, alpha={1: 0.1, 2: 0.0}, beta={1: 0.0, 2: 0.1}, rho=-0.2)
    M1 = model.score_matrix(home_id=1, away_id=2)
    model.rho = 0.0
    M2 = model.score_matrix(home_id=1, away_id=2)
    # (0,0) cell different with/without rho
    assert M1[0, 0] != M2[0, 0]


def test_unknown_teams_default_to_zero() -> None:
    model = DixonColesModel(mu=0.5, alpha={}, beta={})
    probs = model.predict_1x2(home_id=999, away_id=888)
    # Solo HFA actúa → home debe tener edge pequeño pero positivo
    assert probs["home"] > probs["away"]
