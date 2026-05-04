"""Fase 4.13 — Game-script Monte Carlo para props.

LeBron 30+ requiere ~25 min jugados. Si Lakers están +20 en Q4, LeBron descansa
→ under. Game-script (blowout vs nailbiter) es feature crítica para props. MC
simulator da distribución realista.

Uso:
    result = simulate_game_script(
        team_a_rating=1700, team_b_rating=1500,
        sport="nba", n_simulations=5000,
    )
    # result = {"scripts": {"blowout_a": 0.32, "close": 0.45, ...},
    #           "p_player_30plus_given_scripts": {...}}
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Sport = Literal["nba", "nfl", "mlb", "nhl"]


def _score_stddev(sport: Sport) -> float:
    """Std dev de puntos por equipo por sport."""
    return {"nba": 11.0, "nfl": 13.0, "mlb": 3.5, "nhl": 2.0}.get(sport, 10.0)


def _expected_score(rating: float, opp_rating: float, sport: Sport) -> float:
    """Expected points scored = base + rating_diff factor."""
    base = {"nba": 112.0, "nfl": 24.0, "mlb": 4.5, "nhl": 3.0}.get(sport, 100.0)
    # Simple linear rating → expected score (+/- 8 points per 200 Elo)
    return base + (rating - opp_rating) * 0.04


def simulate_game_script(
    team_a_rating: float,
    team_b_rating: float,
    *,
    sport: Sport,
    n_simulations: int = 5000,
    player_minutes_when_close: float = 35.0,
    player_minutes_when_blowout: float = 22.0,
    player_ppm: float = 0.85,  # points per minute (proxy)
) -> dict[str, float]:
    """Monte Carlo simulator. Retorna distribución de game scripts + props derivados."""
    rng = np.random.default_rng(42)

    mu_a = _expected_score(team_a_rating, team_b_rating, sport)
    mu_b = _expected_score(team_b_rating, team_a_rating, sport)
    std = _score_stddev(sport)

    scripts: dict[str, int] = {"blowout_a": 0, "blowout_b": 0, "close": 0, "one_score": 0}
    player_points_30plus = 0

    for _ in range(n_simulations):
        score_a = max(0, rng.normal(mu_a, std))
        score_b = max(0, rng.normal(mu_b, std))
        diff = abs(score_a - score_b)

        if (diff > 20 and sport == "nba") or (diff > 14 and sport == "nfl"):
            script = "blowout_a" if score_a > score_b else "blowout_b"
        elif diff < 5 and sport == "nba":
            script = "close"
        elif diff < 8 and sport == "nfl":
            script = "one_score"
        else:
            script = "close"
        scripts[script] += 1

        # Simulate player points based on script
        is_blowout = script.startswith("blowout")
        minutes_played = player_minutes_when_blowout if is_blowout else player_minutes_when_close
        player_points = rng.normal(player_ppm * minutes_played, 7.0)
        if player_points >= 30:
            player_points_30plus += 1

    return {
        "p_blowout_a": scripts["blowout_a"] / n_simulations,
        "p_blowout_b": scripts["blowout_b"] / n_simulations,
        "p_close_game": scripts["close"] / n_simulations,
        "p_one_score": scripts["one_score"] / n_simulations,
        "p_player_30plus_given_simulation": player_points_30plus / n_simulations,
        "n_simulations": float(n_simulations),
    }
