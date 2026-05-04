"""Tests para EloBuilder — Sprint 10 Fase 2 (Mejora #7)."""

from __future__ import annotations

import pytest

from apuestas.features.elo_builder import (
    EloBuilder,
    expected_score,
    margin_multiplier,
)


def test_initial_rating_is_1500() -> None:
    b = EloBuilder(sport="nba")
    assert b.rating("LAL") == 1500.0


def test_expected_score_symmetry() -> None:
    # Dos equipos iguales → 50/50
    assert expected_score(1500, 1500) == pytest.approx(0.5, abs=1e-6)
    # A vs mucho peor → cercano a 1
    assert expected_score(1800, 1400) > 0.90


def test_home_win_increases_home_rating() -> None:
    b = EloBuilder(sport="nba")
    b.update_match(home="LAL", away="GSW", home_score=110, away_score=100)
    assert b.rating("LAL") > 1500.0
    assert b.rating("GSW") < 1500.0


def test_away_win_decreases_home_rating() -> None:
    b = EloBuilder(sport="nba")
    b.update_match(home="LAL", away="GSW", home_score=100, away_score=110)
    assert b.rating("LAL") < 1500.0
    assert b.rating("GSW") > 1500.0


def test_margin_multiplier_soccer_uses_log() -> None:
    from apuestas.features.elo_builder import _params_for

    params = _params_for("soccer")
    assert params.use_log_margin is True
    # Log mitiga: entre diff=5 y diff=10, el crecimiento es sublineal.
    mm_small = margin_multiplier(1, 0, params)  # diff=1: log(2)/log(2)+1 = 2
    mm_mid = margin_multiplier(5, 0, params)  # diff=5: log(6)/log(2)+1 ≈ 3.58
    mm_big = margin_multiplier(10, 0, params)  # diff=10: log(11)/log(2)+1 ≈ 4.46
    # Crecimiento decreciente (log sublineal)
    assert mm_big - mm_mid < mm_mid - mm_small


def test_blowout_multiplies_delta() -> None:
    b1 = EloBuilder(sport="nba")
    b2 = EloBuilder(sport="nba")
    b1.update_match(home="LAL", away="GSW", home_score=101, away_score=100)
    b2.update_match(home="LAL", away="GSW", home_score=130, away_score=100)
    # Blowout → delta mayor
    delta1 = b1.rating("LAL") - 1500.0
    delta2 = b2.rating("LAL") - 1500.0
    assert delta2 > delta1


def test_features_for_upcoming_has_all_keys() -> None:
    b = EloBuilder(sport="nba")
    b.update_match(home="LAL", away="GSW", home_score=110, away_score=100)
    feats = b.features_for_upcoming("LAL", "GSW", home_rest_days=2, away_rest_days=0)
    expected_keys = {
        "elo_home",
        "elo_away",
        "elo_diff",
        "elo_p_home",
        "elo_n_matches_home",
        "elo_n_matches_away",
        "rest_days_home",
        "rest_days_away",
        "rest_days_diff",
        "b2b_home",
        "b2b_away",
    }
    assert expected_keys.issubset(feats.keys())
    assert feats["b2b_away"] == 1.0
    assert feats["b2b_home"] == 0.0
    assert feats["rest_days_diff"] == 2.0


def test_hfa_affects_expected_home() -> None:
    # Con HFA +100 NBA, home con rating = away debe tener p > 0.5
    b = EloBuilder(sport="nba")
    feats = b.features_for_upcoming("LAL", "GSW")
    assert feats["elo_p_home"] > 0.5


def test_draw_updates_closer_to_tie() -> None:
    b = EloBuilder(sport="soccer")
    b.update_match(home="RMA", away="BAR", home_score=2, away_score=2)
    # Ambos cerca de 1500 (empate no cambia mucho cuando HFA es grande)
    assert abs(b.rating("RMA") - 1500.0) < 15
    assert abs(b.rating("BAR") - 1500.0) < 15


def test_unknown_sport_falls_to_default() -> None:
    b = EloBuilder(sport="curling")
    b.update_match(home="A", away="B", home_score=5, away_score=3)
    assert b.rating("A") > 1500.0


def test_n_matches_counter() -> None:
    b = EloBuilder(sport="nba")
    b.update_match(home="LAL", away="GSW", home_score=100, away_score=95)
    b.update_match(home="GSW", away="LAL", home_score=120, away_score=105)
    assert b.n_matches["LAL"] == 2
    assert b.n_matches["GSW"] == 2
