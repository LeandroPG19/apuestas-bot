"""Tests unitarios para soccer_props_model (Poisson corners/shots/fouls)."""

from __future__ import annotations

import pytest

from apuestas.betting.soccer_props_model import (
    PropProbability,
    compute_match_total_prop,
    compute_prop_distribution,
    compute_prop_lambda,
)


def test_compute_prop_lambda_league_avg() -> None:
    """Equipo con 5 corners vs rival que permite 5 en liga de 5 → λ=5."""
    lam = compute_prop_lambda(team_stat_avg=5.0, opponent_allowed_avg=5.0, league_avg=5.0)
    assert lam == pytest.approx(5.0, abs=0.01)


def test_compute_prop_lambda_strong_team_weak_defense() -> None:
    """Equipo ofensivo fuerte vs defensa débil → λ > team_avg."""
    lam = compute_prop_lambda(team_stat_avg=6.0, opponent_allowed_avg=5.5, league_avg=5.0)
    # 6.0 × (5.5/5.0) = 6.6
    assert lam == pytest.approx(6.6, abs=0.01)


def test_compute_prop_lambda_zero_league_avg_fallback() -> None:
    lam = compute_prop_lambda(team_stat_avg=5.0, opponent_allowed_avg=5.0, league_avg=0)
    assert lam == 5.0  # fallback a team_stat_avg


def test_compute_prop_distribution_monotonic() -> None:
    """P(X > line) debe ser monótono decreciente en line."""
    probs = compute_prop_distribution(lam=5.0, lines=[3.5, 4.5, 5.5, 6.5, 7.5])
    prob_list = [probs[line] for line in [3.5, 4.5, 5.5, 6.5, 7.5]]
    for i in range(len(prob_list) - 1):
        assert prob_list[i] >= prob_list[i + 1]


def test_compute_prop_distribution_poisson_stats() -> None:
    """P(X > λ) ≈ 0.5 para Poisson con λ entero."""
    probs = compute_prop_distribution(lam=5.0, lines=[4.5])
    # P(X > 4) = 1 - CDF(4, 5) ≈ 0.56
    assert 0.5 < probs[4.5] < 0.65


def test_compute_match_total_prop_sum_of_poissons() -> None:
    """Suma de Poissons independientes es Poisson con λ_total."""
    probs = compute_match_total_prop(home_lambda=1.5, away_lambda=1.3, lines=[2.5])
    # λ_total = 2.8 → P(X > 2) = 1 - CDF(2, 2.8) ≈ 0.53
    assert 0.4 < probs[2.5] < 0.7


def test_prop_probability_dataclass_frozen() -> None:
    p = PropProbability(
        prop_name="corners",
        team="home",
        lambda_rate=5.0,
        over_prob_for_line={5.5: 0.38},
    )
    # slots=True + frozen=True → no se puede mutar
    with pytest.raises(AttributeError):
        p.team = "away"  # type: ignore[misc]
