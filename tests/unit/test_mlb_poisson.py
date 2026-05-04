"""Tests para MLBPoissonModel — Sprint 10 (Mejora #8)."""

from __future__ import annotations

import pytest

from apuestas.ml.mlb_poisson import DEFAULT_PARK_FACTORS, MLBPoissonModel


def _toy_matches() -> list[dict]:
    return [
        {"home_id": 1, "away_id": 2, "home_runs": 7, "away_runs": 2},
        {"home_id": 1, "away_id": 2, "home_runs": 5, "away_runs": 3},
        {"home_id": 2, "away_id": 1, "home_runs": 2, "away_runs": 6},
        {"home_id": 1, "away_id": 3, "home_runs": 8, "away_runs": 4},
        {"home_id": 3, "away_id": 1, "home_runs": 3, "away_runs": 7},
        {"home_id": 2, "away_id": 3, "home_runs": 4, "away_runs": 4},
    ] * 5


def test_default_park_factors_have_coors() -> None:
    assert "coors field" in DEFAULT_PARK_FACTORS
    assert DEFAULT_PARK_FACTORS["coors field"] > 1.0
    assert DEFAULT_PARK_FACTORS["petco park"] < 1.0


def test_fit_returns_model_with_teams() -> None:
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=50)
    assert 1 in model.offense
    assert 2 in model.offense


def test_strong_offense_higher_beta() -> None:
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=100)
    # Equipo 1 anota más → offense[1] > offense[2]
    assert model.offense[1] > model.offense[2]


def test_predict_moneyline_sums_to_one() -> None:
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=50)
    probs = model.predict_moneyline(home_id=1, away_id=2)
    assert probs["home"] + probs["away"] == pytest.approx(1.0, abs=1e-6)


def test_predict_total_sums_to_one() -> None:
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=50)
    r = model.predict_total(home_id=1, away_id=2, line=8.5)
    assert r["over"] + r["under"] == pytest.approx(1.0, abs=1e-6)


def test_park_factor_coors_increases_runs() -> None:
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=50)
    r_coors = model.predict_total(home_id=1, away_id=2, line=8.5, venue_name="Coors Field")
    r_petco = model.predict_total(home_id=1, away_id=2, line=8.5, venue_name="Petco Park")
    # Coors debe dar mayor prob de over que Petco (15% vs -7%)
    assert r_coors["over"] > r_petco["over"]


def test_predict_runline_sums_to_one() -> None:
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=50)
    r = model.predict_runline(home_id=1, away_id=2, line=1.5)
    assert r["home"] + r["away"] == pytest.approx(1.0, abs=1e-6)


def test_runline_home_minus15_vs_away_plus15() -> None:
    # Home favorito en runline −1.5 debería ser menos probable que away +1.5
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=50)
    r_home_fav = model.predict_runline(home_id=1, away_id=2, line=-1.5)
    r_home_dog = model.predict_runline(home_id=1, away_id=2, line=1.5)
    assert r_home_fav["home"] < r_home_dog["home"]


def test_home_field_advantage() -> None:
    model = MLBPoissonModel.fit(_toy_matches(), n_iter=100)
    # Con HFA positivo, equipos iguales → home favorito
    model2 = MLBPoissonModel(mu=1.5, offense={1: 0.0, 2: 0.0}, defense={1: 0.0, 2: 0.0}, hfa=0.1)
    probs = model2.predict_moneyline(home_id=1, away_id=2)
    assert probs["home"] > probs["away"]


def test_fit_empty_raises() -> None:
    with pytest.raises(ValueError, match="sin matches"):
        MLBPoissonModel.fit([], n_iter=5)
