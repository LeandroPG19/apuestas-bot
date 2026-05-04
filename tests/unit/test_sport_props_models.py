"""Tests unitarios para sport_props_models (NBA/MLB/NFL/NHL/Tenis)."""

from __future__ import annotations

import pytest

from apuestas.betting.sport_props_models import (
    mlb_batter_prop,
    mlb_pitcher_strikeouts,
    nba_player_prop,
    nfl_player_prop,
    nhl_player_prop,
    tennis_match_markov,
)


def test_nba_player_prop_mean_equals_input() -> None:
    """p.mean == avg_last_10 × opp_factor (1.0 si opp = league)."""
    p = nba_player_prop("LeBron", "points", 25.0)
    assert p.predicted_mean == pytest.approx(25.0, abs=0.01)
    assert p.sport == "nba"
    assert p.model == "normal"


def test_nba_player_prop_over_probability_monotonic() -> None:
    """P(X > line) debe ser DECRECIENTE a medida que line aumenta."""
    p = nba_player_prop("LeBron", "points", 25.0)
    lines_sorted = sorted(p.over_probs.keys())
    probs = [p.over_probs[line] for line in lines_sorted]
    # Debe ser monótono decreciente
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1]


def test_nba_defense_adjustment_monotonic() -> None:
    """Comprobar que diferentes opponent_def_rating cambian el predicted_mean.

    El modelo aplica opp_factor = league_avg_def / opponent_def_rating.
    Mayor rating del oponente → factor menor → predicted_mean menor.
    """
    p_a = nba_player_prop("LeBron", "points", 25.0, opponent_def_rating=115)
    p_b = nba_player_prop("LeBron", "points", 25.0, opponent_def_rating=105)
    assert p_a.predicted_mean != p_b.predicted_mean
    # Relación consistente con la fórmula league_avg / opp_rating
    assert p_a.predicted_mean < p_b.predicted_mean


def test_mlb_pitcher_strikeouts_poisson() -> None:
    """K pitcher sigue Poisson con λ = K/9 × IP/9."""
    p = mlb_pitcher_strikeouts("Ohtani", k_per_9=11.5, expected_ip=6.5)
    # λ = 11.5 × (6.5/9) ≈ 8.306
    assert p.predicted_mean == pytest.approx(8.31, abs=0.05)
    assert p.model == "poisson"
    # Poisson std = sqrt(λ)
    assert p.predicted_std == pytest.approx(2.88, abs=0.1)


def test_mlb_batter_prop_returns_valid() -> None:
    p = mlb_batter_prop("Judge", "HR", avg_per_pa=0.08, expected_pa=4.0)
    assert p.sport == "mlb"
    assert p.model == "monte_carlo"
    assert 0.0 <= p.over_probs[0.5] <= 1.0
    assert 0.0 <= p.over_probs[1.5] <= 1.0


def test_nfl_player_prop_defense_rank_adjustment() -> None:
    """Mejor defensa (rank 1) vs peor (rank 32) → mean diferente."""
    p_top_def = nfl_player_prop("Mahomes", "passing_yards", 270, opponent_def_rank=1)
    p_bad_def = nfl_player_prop("Mahomes", "passing_yards", 270, opponent_def_rank=32)
    assert p_bad_def.predicted_mean > p_top_def.predicted_mean


def test_nhl_player_prop_poisson() -> None:
    p = nhl_player_prop("McDavid", "shots", 3.5)
    assert p.model == "poisson"
    assert p.predicted_mean == 3.5


def test_tennis_markov_probabilities_sum_to_one() -> None:
    """P(p1 wins) + P(p2 wins) = 1.0."""
    result = tennis_match_markov(0.72, 0.65, best_of=3, n_sims=2000)
    assert 0.0 <= result["p_player1_wins"] <= 1.0
    # 2_0 + 2_1 debe ser <= p_player1_wins
    total_p1 = result["p_2_0_sets"] + result["p_2_1_sets"]
    assert abs(total_p1 - result["p_player1_wins"]) < 0.01


def test_tennis_stronger_server_wins_more() -> None:
    """Jugador con mejor p_serve debe ganar más probablemente."""
    result = tennis_match_markov(0.80, 0.60, best_of=3, n_sims=2000)
    # p_serve 80% vs 60% → p1 debe ganar > 60%
    assert result["p_player1_wins"] > 0.6


def test_nba_stat_std_default_fallback() -> None:
    """Para stat desconocida, usa std default 3.0."""
    p = nba_player_prop("Player", "weird_stat", 10.0)
    assert p.predicted_std == 3.0  # fallback
