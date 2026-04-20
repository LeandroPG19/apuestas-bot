"""Tests tenis — Elo surface + Barnett-Clarke serve model."""

from __future__ import annotations

import pytest

from apuestas.features.tennis import (
    ServeReturnStats,
    combine_elo_surface,
    elo_update,
    elo_win_probability,
    hold_serve_probability,
    match_probability_bo3,
    match_probability_bo5,
)


def test_elo_win_probability_equal_ratings() -> None:
    assert elo_win_probability(rating_a=1500, rating_b=1500) == pytest.approx(0.5)


def test_elo_win_probability_heavy_favorite() -> None:
    # +400 Elo → ~91% win prob
    p = elo_win_probability(rating_a=1900, rating_b=1500)
    assert p > 0.90


def test_elo_update_winner_gains_loser_loses() -> None:
    new_a, new_b = elo_update(rating_a=1500, rating_b=1500, outcome_a=1, k=32)
    assert new_a > 1500
    assert new_b < 1500
    # K=32 → desplazamiento ~16 en un match equilibrado
    assert new_a == pytest.approx(1516, abs=0.5)


def test_elo_update_upset_bigger_swing() -> None:
    """Underdog wins → big swing."""
    new_a, new_b = elo_update(rating_a=1400, rating_b=1700, outcome_a=1, k=32)
    assert new_a - 1400 > 20  # upset = swing grande


def test_combine_elo_surface_heavier_on_surface() -> None:
    """Surface weight 0.6 → surface domina."""
    combined = combine_elo_surface(
        elo_overall_diff=0.0,
        elo_surface_diff=100.0,
        surface_weight=0.6,
    )
    assert combined == pytest.approx(60.0)


def test_hold_serve_probability_in_range() -> None:
    server = ServeReturnStats(serve_pts_won_pct=0.65, return_pts_won_pct=0.35)
    returner = ServeReturnStats(serve_pts_won_pct=0.60, return_pts_won_pct=0.40)
    p = hold_serve_probability(server_stats=server, returner_stats=returner)
    assert 0.70 <= p <= 0.95  # holds típicos ATP


def test_hold_serve_better_server_holds_more() -> None:
    strong = ServeReturnStats(serve_pts_won_pct=0.75, return_pts_won_pct=0.30)
    weak = ServeReturnStats(serve_pts_won_pct=0.55, return_pts_won_pct=0.45)
    equal = ServeReturnStats(serve_pts_won_pct=0.60, return_pts_won_pct=0.40)

    p_strong = hold_serve_probability(server_stats=strong, returner_stats=equal)
    p_weak = hold_serve_probability(server_stats=weak, returner_stats=equal)
    assert p_strong > p_weak


def test_match_probability_bo3_equal_holds() -> None:
    """Dos jugadores con holds idénticos → ~50%."""
    p = match_probability_bo3(p_hold_a=0.80, p_hold_b=0.80)
    assert p == pytest.approx(0.5, abs=0.05)


def test_match_probability_bo5_vs_bo3_favorite_gains() -> None:
    """Favorito gana más en BO5 (menos varianza)."""
    p_bo3 = match_probability_bo3(p_hold_a=0.85, p_hold_b=0.70)
    p_bo5 = match_probability_bo5(p_hold_a=0.85, p_hold_b=0.70)
    assert p_bo5 > p_bo3


def test_match_probability_bounds() -> None:
    p = match_probability_bo3(p_hold_a=0.95, p_hold_b=0.50)
    assert 0.0 <= p <= 1.0
    p5 = match_probability_bo5(p_hold_a=0.50, p_hold_b=0.95)
    assert 0.0 <= p5 <= 1.0
