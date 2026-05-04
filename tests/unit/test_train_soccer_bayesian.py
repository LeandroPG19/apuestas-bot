"""Tests para Bayesian prior blend en fit_dixon_coles fallback."""

from __future__ import annotations

import numpy as np

from apuestas.ml.train_soccer import _fit_independent_poisson, _IndependentPoissonModel


def _sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    # 3 teams (1,2,3), 6 partidos
    home = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    away = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    home_goals = np.array([2, 3, 1, 2, 1, 2], dtype=np.int64)
    away_goals = np.array([1, 2, 2, 0, 3, 1], dtype=np.int64)
    team_map = {101: 0, 102: 1, 103: 2}
    return home_goals, away_goals, home, away, team_map


def test_fit_poisson_without_priors() -> None:
    hg, ag, h, a, tm = _sample_data()
    model = _fit_independent_poisson(home_goals=hg, away_goals=ag, home=h, away=a, team_map=tm)
    assert isinstance(model, _IndependentPoissonModel)
    # Cada team debe tener attack/defense coeficiente > 0
    assert model.attack.min() > 0
    assert model.defense.min() > 0


def test_fit_poisson_with_bayesian_priors_blends_values() -> None:
    hg, ag, h, a, tm = _sample_data()

    # Prior extreme para team 101: attack=2.0 (muy ofensivo)
    priors = {101: (2.0, 1.0), 102: (0.5, 1.5)}

    model_with = _fit_independent_poisson(
        home_goals=hg, away_goals=ag, home=h, away=a, team_map=tm, bayesian_priors=priors
    )
    model_without = _fit_independent_poisson(
        home_goals=hg, away_goals=ag, home=h, away=a, team_map=tm
    )

    # Team 101 (index 0) debe estar más cerca del prior attack=2.0 con priors
    # que sin priors.
    assert abs(model_with.attack[0] - 2.0) < abs(model_without.attack[0] - 2.0)


def test_predict_handles_unknown_team_gracefully() -> None:
    hg, ag, h, a, tm = _sample_data()
    model = _fit_independent_poisson(home_goals=hg, away_goals=ag, home=h, away=a, team_map=tm)
    # team 999 no en team_map → no debe crashear (usa defaults)
    pred = model.predict(home_id=999, away_id=101)
    probs = pred.home_draw_away
    assert abs(sum(probs) - 1.0) < 0.02
    assert all(0 <= p <= 1 for p in probs)
