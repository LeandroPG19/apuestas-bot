"""Tests Bayesian prior (Gap 6 / Sprint 2.9)."""

from __future__ import annotations

import pytest

from apuestas.ml.bayesian_prior import (
    adaptive_shrinkage,
    beta_binomial_win_prob,
)


def test_prior_dominates_with_no_evidence() -> None:
    p_mean, _, _ = beta_binomial_win_prob(
        wins_this_season=0,
        games_this_season=0,
        wins_prior_season=48,
        games_prior_season=82,
        shrinkage=0.25,
    )
    # Con prior pesado debería estar cerca de 48/82 ≈ 0.585
    assert 0.50 < p_mean < 0.65


def test_evidence_dominates_with_many_samples() -> None:
    p_mean, _, _ = beta_binomial_win_prob(
        wins_this_season=60,
        games_this_season=80,
        wins_prior_season=30,
        games_prior_season=82,
        shrinkage=0.25,
    )
    # Con 80 games actuales y prior con 30/82, p_mean debe moverse hacia
    # la evidencia actual (0.75) aunque el prior pese shrinkage=0.25.
    # Valor esperado: (0.25*30 + 60 + 0.5)/(0.25*82 + 80 + 1) ≈ 0.669
    assert p_mean > 0.60


def test_ci_widens_with_few_samples() -> None:
    _, low_few, up_few = beta_binomial_win_prob(
        wins_this_season=2,
        games_this_season=5,
        wins_prior_season=40,
        games_prior_season=82,
    )
    _, low_many, up_many = beta_binomial_win_prob(
        wins_this_season=40,
        games_this_season=80,
        wins_prior_season=40,
        games_prior_season=82,
    )
    assert (up_few - low_few) > (up_many - low_many)


def test_adaptive_shrinkage_decays() -> None:
    assert adaptive_shrinkage(0) == 0.5
    assert adaptive_shrinkage(82) == pytest.approx(0.0)
    assert adaptive_shrinkage(41) == pytest.approx(0.25, rel=1e-3)
    assert adaptive_shrinkage(200) == 0.0


def test_raises_on_negative_games() -> None:
    with pytest.raises(ValueError):
        beta_binomial_win_prob(
            wins_this_season=-1,
            games_this_season=-1,
            wins_prior_season=10,
            games_prior_season=20,
        )
