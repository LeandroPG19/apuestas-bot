"""Fase 4.15 — Bayesian shrinkage para small-sample early-season.

Temporada nueva (10 games in), el modelo con rolling simple sufre overfitting
brutal (Trail Blazers 10-0 no significa 82-0). Benham/Smartodds y todos los
sharps aplican shrinkage tipo James-Stein:

    shrunk_rating = n / (n + k) × observed + k / (n + k) × league_prior

Donde `k` = hyperparámetro de confianza en el prior (típicamente 20 games
para NBA, 5 para NFL temporada corta, 30 para MLB 162-game).

Uso:
    from apuestas.features.shrinkage import bayesian_shrink_rating
    adjusted = bayesian_shrink_rating(
        observed_rating=0.85,  # ej. team winrate sample
        n_games=10,
        league_prior=0.5,  # winrate promedio liga
        k=20,
    )
    # Return: 0.5 × (20/30) + 0.85 × (10/30) = 0.617 (shrunk hacia prior)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Sport = Literal["nba", "mlb", "nfl", "nhl", "soccer", "tennis"]

# K-factors por sport (basado en Benham/Hvattum papers)
K_BY_SPORT: dict[Sport, float] = {
    "nba": 20.0,  # 82 games / season
    "mlb": 30.0,  # 162 games / season
    "nfl": 5.0,  # 17 games / season → fast convergence
    "nhl": 15.0,  # 82 games / season
    "soccer": 10.0,  # 38 games EPL / season
    "tennis": 25.0,  # individual matches
}


def bayesian_shrink_rating(
    observed_rating: float,
    n_games: int,
    *,
    league_prior: float = 0.5,
    k: float = 20.0,
) -> float:
    """Shrinkage weighted-average: converge a `league_prior` cuando n→0."""
    if n_games <= 0:
        return league_prior
    weight_observed = n_games / (n_games + k)
    weight_prior = k / (n_games + k)
    return weight_observed * observed_rating + weight_prior * league_prior


def shrink_array(
    observed: np.ndarray,
    n_games: np.ndarray,
    *,
    league_prior: float = 0.5,
    k: float = 20.0,
) -> np.ndarray:
    """Vectorized shrinkage sobre array de teams/players."""
    weights_obs = n_games / (n_games + k)
    return weights_obs * observed + (1 - weights_obs) * league_prior


def shrink_for_sport(
    observed_rating: float,
    n_games: int,
    sport: Sport,
    *,
    league_prior: float = 0.5,
) -> float:
    """Shortcut: usa k apropiado para el sport."""
    k = K_BY_SPORT.get(sport, 20.0)
    return bayesian_shrink_rating(observed_rating, n_games, league_prior=league_prior, k=k)


def is_early_season(n_games: int, sport: Sport) -> bool:
    """True si estamos en fase "early-season" donde el shrinkage es crítico.

    Thresholds: 20% del schedule típico por sport.
    """
    threshold_by_sport: dict[Sport, int] = {
        "nba": 16,  # 20% de 82
        "mlb": 32,  # 20% de 162
        "nfl": 4,  # 20% de 17 (round)
        "nhl": 16,
        "soccer": 8,  # 20% de 38
        "tennis": 10,
    }
    return n_games < threshold_by_sport.get(sport, 20)
