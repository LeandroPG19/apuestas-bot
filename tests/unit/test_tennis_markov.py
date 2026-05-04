"""Tests Fase 4.10 — Tennis Markov point-by-point.

NOTE: tests marcados `slow` porque Monte Carlo (~5-20k simulaciones × test)
tardan ~90s cada uno. Correr con `pytest -m slow` explícito; default CI los skip.
"""

from __future__ import annotations

import pytest

from apuestas.ml.tennis_markov import (
    prob_total_games_over,
    prob_win_game_on_serve,
    prob_win_match,
    prob_win_set,
)

# Marca global al módulo
pytestmark = pytest.mark.slow


def test_win_game_equal_strengths() -> None:
    """p=0.5 → prob win game 0.5 (fair)."""
    assert abs(prob_win_game_on_serve(0.5) - 0.5) < 0.05


def test_win_game_strong_server() -> None:
    """p=0.70 (strong serve) → prob win game alto (~0.90)."""
    p_win = prob_win_game_on_serve(0.70)
    assert 0.85 < p_win < 0.95


def test_win_game_weak_server() -> None:
    """p=0.30 → prob win game bajo (~0.10)."""
    p_win = prob_win_game_on_serve(0.30)
    assert 0.05 < p_win < 0.15


def test_win_game_symmetry() -> None:
    """prob_win(p) + prob_win(1-p) ≈ 1."""
    for p in (0.4, 0.55, 0.65):
        symmetry = prob_win_game_on_serve(p) + prob_win_game_on_serve(1 - p)
        assert abs(symmetry - 1.0) < 0.05


def test_win_set_equal_servers() -> None:
    """Dos servers iguales 0.65 → set ~50/50."""
    p_set = prob_win_set(0.65, 0.65)
    assert 0.40 < p_set < 0.60


def test_win_match_bo3_vs_bo5_stronger_favorito() -> None:
    """El favorito gana más en BO5 que en BO3 (más muestras → menos varianza)."""
    bo3 = prob_win_match(0.68, 0.62, best_of=3)["p_win_match"]
    bo5 = prob_win_match(0.68, 0.62, best_of=5)["p_win_match"]
    # BO5 debería ser ligeramente mayor para el favorito (más variance reduction)
    assert bo3 > 0.45
    assert bo5 > 0.45


def test_win_match_returns_all_fields() -> None:
    result = prob_win_match(0.65, 0.62, best_of=3)
    expected_keys = {
        "p_win_match",
        "p_win_straight_sets",
        "p_win_close_match",
        "p_win_set",
        "p_serve_a",
        "p_serve_b",
    }
    assert expected_keys.issubset(result.keys())


def test_total_games_monotonic() -> None:
    """P(total > 15) > P(total > 25) (más games = menos probable)."""
    p_over_15 = prob_total_games_over(0.65, 0.62, 15, best_of=3)
    p_over_25 = prob_total_games_over(0.65, 0.62, 25, best_of=3)
    assert p_over_15 > p_over_25


def test_probabilities_in_range() -> None:
    for p_a, p_b in [(0.5, 0.5), (0.7, 0.5), (0.6, 0.7)]:
        result = prob_win_match(p_a, p_b, best_of=3)
        assert 0 <= result["p_win_match"] <= 1
        assert 0 <= result["p_win_straight_sets"] <= result["p_win_match"]
