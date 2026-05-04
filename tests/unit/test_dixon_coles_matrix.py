"""Tests Fase 3.1 — Dixon-Coles matrix + derived props."""

from __future__ import annotations

import pytest

from apuestas.ml.props_distributions import (
    dc_prob_anytime_scorer,
    dc_prob_away_win,
    dc_prob_btts,
    dc_prob_correct_score,
    dc_prob_double_chance,
    dc_prob_draw,
    dc_prob_home_win,
    dc_prob_over,
    dixon_coles_simulate,
    dixon_coles_tau,
)


def test_matrix_sums_to_one() -> None:
    m = dixon_coles_simulate(1.5, 1.2)
    assert abs(m.sum() - 1.0) < 0.001


def test_matrix_shape_default() -> None:
    m = dixon_coles_simulate(1.5, 1.2)
    assert m.shape == (10, 10)


def test_matrix_shape_custom() -> None:
    m = dixon_coles_simulate(1.5, 1.2, max_goals=6)
    assert m.shape == (6, 6)


def test_dc_tau_corrections() -> None:
    # rho<0 (típico soccer): aumenta 0-0 y 1-1, reduce 0-1 y 1-0
    assert dixon_coles_tau(1, 1, 1.5, 1.2, rho=-0.1) > 1.0  # 1 - (-0.1) = 1.1
    assert dixon_coles_tau(0, 0, 1.5, 1.2, rho=-0.1) > 1.0  # 1 - lam·mu·(-0.1) > 1
    assert dixon_coles_tau(0, 1, 1.5, 1.2, rho=-0.1) < 1.0  # 1 + lam·(-0.1) = 1 - 0.15
    assert dixon_coles_tau(1, 0, 1.5, 1.2, rho=-0.1) < 1.0
    # Para i,j fuera de {0,1} tau=1 (sin corrección)
    assert dixon_coles_tau(2, 3, 1.5, 1.2, rho=-0.1) == 1.0


def test_btts_consistent() -> None:
    m = dixon_coles_simulate(1.8, 1.5)
    p_btts = dc_prob_btts(m)
    assert 0 < p_btts < 1
    # Con equipos que marcan ~1.5 goles promedio, BTTS debería ser ~55-70%
    assert 0.45 < p_btts < 0.80


def test_over_under_consistent() -> None:
    m = dixon_coles_simulate(1.8, 1.5)
    p_over_25 = dc_prob_over(m, 2.5)
    p_over_15 = dc_prob_over(m, 1.5)
    p_over_05 = dc_prob_over(m, 0.5)
    # Monotonicidad: over_05 > over_15 > over_25
    assert p_over_05 > p_over_15 > p_over_25
    # Con 3.3 goles expected, over 2.5 ~60%
    assert 0.45 < p_over_25 < 0.75


def test_correct_score_sum_matches_matrix() -> None:
    m = dixon_coles_simulate(1.5, 1.2)
    # Suma de todos correct scores debe ser 1.0 (mismo que matrix.sum())
    total = sum(
        dc_prob_correct_score(m, i, j) for i in range(m.shape[0]) for j in range(m.shape[1])
    )
    assert abs(total - 1.0) < 0.001


def test_home_draw_away_sum_to_one() -> None:
    m = dixon_coles_simulate(1.5, 1.2)
    total = dc_prob_home_win(m) + dc_prob_draw(m) + dc_prob_away_win(m)
    assert abs(total - 1.0) < 0.001


def test_home_advantage_stronger_team() -> None:
    """Home 2.0 xG vs Away 1.0 xG → home win > away win."""
    m = dixon_coles_simulate(2.0, 1.0)
    assert dc_prob_home_win(m) > dc_prob_away_win(m)
    assert dc_prob_home_win(m) > 0.45


def test_anytime_scorer_home() -> None:
    """P(team marca) × share ≈ probabilidad anytime."""
    m = dixon_coles_simulate(2.5, 1.5)
    # Home team marca con prob alta (avg xG=2.5)
    p_home_scores = 1.0 - float(m[0, :].sum())
    # Jugador con 30% share
    p_player = dc_prob_anytime_scorer(m, "home", 0.3)
    assert abs(p_player - p_home_scores * 0.3) < 0.001


def test_anytime_scorer_boundaries() -> None:
    m = dixon_coles_simulate(2.0, 1.5)
    # Share 0 → prob 0
    assert dc_prob_anytime_scorer(m, "home", 0.0) == 0.0
    # Share 1 (team único scorer) → igual a P(team scores)
    p_team_scores = 1.0 - float(m[0, :].sum())
    assert abs(dc_prob_anytime_scorer(m, "home", 1.0) - p_team_scores) < 0.001


def test_double_chance_variants() -> None:
    m = dixon_coles_simulate(1.5, 1.5)
    dc_1x = dc_prob_double_chance(m, "1X")
    dc_x2 = dc_prob_double_chance(m, "X2")
    dc_12 = dc_prob_double_chance(m, "12")
    # Cada uno es ≥ que el componente individual
    assert dc_1x >= dc_prob_home_win(m)
    assert dc_x2 >= dc_prob_away_win(m)
    # 12 = 1 - draw
    assert abs(dc_12 + dc_prob_draw(m) - 1.0) < 0.001


def test_double_chance_invalid_variant() -> None:
    m = dixon_coles_simulate(1.5, 1.5)
    with pytest.raises(ValueError, match="variant"):
        dc_prob_double_chance(m, "invalid")


def test_correct_score_out_of_range() -> None:
    m = dixon_coles_simulate(1.5, 1.2, max_goals=5)
    # Request fuera del rango
    assert dc_prob_correct_score(m, 10, 10) == 0.0
    assert dc_prob_correct_score(m, -1, 0) == 0.0


def test_all_probabilities_in_range() -> None:
    """Sanity: todas las prob derivadas ∈ [0, 1]."""
    m = dixon_coles_simulate(2.0, 1.7)
    for prob_fn in (
        dc_prob_btts,
        dc_prob_home_win,
        dc_prob_draw,
        dc_prob_away_win,
        lambda x: dc_prob_over(x, 2.5),
        lambda x: dc_prob_correct_score(x, 2, 1),
    ):
        p = prob_fn(m)
        assert 0 <= p <= 1, f"prob fuera de rango: {p}"
