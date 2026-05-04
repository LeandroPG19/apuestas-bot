"""Fase 3.3 — Strength of Schedule (SoS) ratings.

LeBron 30 pts vs Spurs ≠ 30 pts vs Celtics. Rolling simple infla stats de equipos
con calendario fácil. Implementamos 2 sistemas canónicos:

1. **Massey ratings**: solución least-squares (X·r = b) donde r=ratings,
   b=diferencial de puntos. Standard en college basketball/fútbol.

2. **Elo dinámico**: K-factor sport-dependent (NBA 20, NFL 35, soccer 25).
   Update online cada partido: `r_home_new = r_home + K·(result - expected)`.

Ambos se exponen como features:
  - `team_rating_massey`: [-1.0, +1.0] centrado en 0
  - `team_rating_elo`: típicamente [1200, 1800], centrado en 1500
  - `opp_strength`: rating del oponente (para feature `stat / opp_strength`)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# K-factors (standard values in literature)
ELO_K_BY_SPORT: dict[str, float] = {
    "nba": 20.0,
    "mlb": 4.0,  # 162-game season, small K
    "nfl": 35.0,  # 17-game season, fast convergence
    "nhl": 6.0,
    "soccer": 25.0,
    "tennis": 32.0,
}
ELO_BASE = 1500.0
ELO_DIVISOR = 400.0  # standard Elo divisor


@dataclass(slots=True)
class TeamRating:
    team_id: int
    rating_massey: float  # normalized [-1, +1]
    rating_elo: float  # typical [1200, 1800]
    n_matches: int
    updated_at: datetime


def expected_score_elo(r_a: float, r_b: float) -> float:
    """E(A) = 1 / (1 + 10^((r_b - r_a) / 400))."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / ELO_DIVISOR))


def update_elo_pair(
    r_home: float,
    r_away: float,
    result: float,  # 1.0 home win, 0.5 draw, 0.0 away win
    k: float,
    *,
    hfa_bonus: float = 0.0,
) -> tuple[float, float]:
    """Update Elo dinámico tras un partido. Retorna (new_r_home, new_r_away)."""
    expected_home = expected_score_elo(r_home + hfa_bonus, r_away)
    expected_away = 1.0 - expected_home
    delta_home = k * (result - expected_home)
    delta_away = k * ((1 - result) - expected_away)
    return r_home + delta_home, r_away + delta_away


async def compute_elo_ratings(
    sport_code: str,
    *,
    season_start: datetime | None = None,
    hfa_bonus: float = 50.0,  # puntos Elo de ventaja por casa
) -> dict[int, float]:
    """Recorre matches de la temporada en orden cronológico y actualiza Elo.

    Retorna `{team_id: final_elo}`. Si team nuevo → comienza en ELO_BASE.
    """
    k = ELO_K_BY_SPORT.get(sport_code, 20.0)

    async with session_scope() as session:
        query_params: dict[str, Any] = {"sp": sport_code}
        where_clause = "m.sport_code = :sp AND m.status = 'finished'"
        if season_start is not None:
            where_clause += " AND m.start_time >= :since"
            query_params["since"] = season_start

        rows = (
            await session.execute(
                text(
                    f"""
                    SELECT m.id, m.home_team_id, m.away_team_id,
                           m.home_score, m.away_score, m.start_time
                    FROM matches m
                    WHERE {where_clause}
                      AND m.home_score IS NOT NULL
                      AND m.away_score IS NOT NULL
                    ORDER BY m.start_time ASC
                    """
                ),
                query_params,
            )
        ).all()

    ratings: dict[int, float] = defaultdict(lambda: ELO_BASE)
    for r in rows:
        home_id = r.home_team_id
        away_id = r.away_team_id
        if r.home_score > r.away_score:
            result = 1.0
        elif r.home_score < r.away_score:
            result = 0.0
        else:
            result = 0.5
        ratings[home_id], ratings[away_id] = update_elo_pair(
            ratings[home_id], ratings[away_id], result, k=k, hfa_bonus=hfa_bonus
        )

    logger.info(
        "sos.elo_computed",
        sport=sport_code,
        n_teams=len(ratings),
        n_matches=len(rows),
    )
    return dict(ratings)


async def compute_massey_ratings(
    sport_code: str,
    *,
    season_start: datetime | None = None,
    max_games: int | None = None,
) -> dict[int, float]:
    """Massey ratings via least-squares.

    Para cada match: `r_home - r_away = margin` (margin = home_score - away_score).
    Sistema sobredeterminado → pseudoinverse (np.linalg.lstsq).
    Normalizamos ratings a [-1, +1].

    Retorna `{team_id: rating_normalized}`.
    """
    async with session_scope() as session:
        query_params: dict[str, Any] = {"sp": sport_code}
        where_clause = "m.sport_code = :sp AND m.status = 'finished'"
        if season_start is not None:
            where_clause += " AND m.start_time >= :since"
            query_params["since"] = season_start

        limit_clause = "LIMIT :lim" if max_games else ""
        if max_games:
            query_params["lim"] = max_games

        rows = (
            await session.execute(
                text(
                    f"""
                    SELECT m.home_team_id, m.away_team_id,
                           m.home_score - m.away_score AS margin
                    FROM matches m
                    WHERE {where_clause}
                      AND m.home_score IS NOT NULL
                      AND m.away_score IS NOT NULL
                    ORDER BY m.start_time ASC
                    {limit_clause}
                    """
                ),
                query_params,
            )
        ).all()

    if not rows:
        return {}

    teams = sorted({r.home_team_id for r in rows} | {r.away_team_id for r in rows})
    team_to_idx = {tid: i for i, tid in enumerate(teams)}
    n_teams = len(teams)
    n_games = len(rows)

    # X (n_games x n_teams): +1 home, -1 away
    X = np.zeros((n_games, n_teams), dtype=np.float64)
    b = np.zeros(n_games, dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, team_to_idx[r.home_team_id]] = 1
        X[i, team_to_idx[r.away_team_id]] = -1
        b[i] = float(r.margin)

    # Restricción: sum(ratings) = 0 para unicidad
    constraint = np.ones((1, n_teams))
    X_full = np.vstack([X, constraint])
    b_full = np.concatenate([b, [0.0]])

    try:
        ratings_raw, *_ = np.linalg.lstsq(X_full, b_full, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("sos.massey_lstsq_failed", sport=sport_code)
        return {}

    # Normalizar a [-1, +1]
    max_abs = float(np.max(np.abs(ratings_raw)))
    if max_abs < 1e-9:
        normalized = ratings_raw
    else:
        normalized = ratings_raw / max_abs

    logger.info("sos.massey_computed", sport=sport_code, n_teams=n_teams, n_games=n_games)
    return {tid: float(normalized[team_to_idx[tid]]) for tid in teams}


async def persist_team_ratings(
    sport_code: str,
    *,
    season_start: datetime | None = None,
) -> dict[str, Any]:
    """Compute + persist Massey + Elo ratings en una tabla nueva `team_ratings`.

    Si tabla no existe, migración 0012 la creará. Por ahora se guarda en-memoria +
    puede leerse con get_team_rating(). Tabla se añade en migración separada si
    se decide persistir (quick path: compute on-demand con cache).
    """
    massey = await compute_massey_ratings(sport_code, season_start=season_start)
    elo = await compute_elo_ratings(sport_code, season_start=season_start)
    return {
        "sport": sport_code,
        "n_teams_massey": len(massey),
        "n_teams_elo": len(elo),
        "massey": massey,
        "elo": elo,
    }


_RATING_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 86400  # 24h


async def get_team_rating(team_id: int, sport_code: str) -> dict[str, float] | None:
    """Retorna ratings Massey + Elo de un team con cache 24h."""
    cache_key = sport_code
    cached = _RATING_CACHE.get(cache_key)
    now = datetime.now(tz=UTC)
    if cached and (now - cached["computed_at"]).total_seconds() < _CACHE_TTL_SECONDS:
        return {
            "rating_massey": cached["massey"].get(team_id, 0.0),
            "rating_elo": cached["elo"].get(team_id, ELO_BASE),
        }

    result = await persist_team_ratings(sport_code)
    _RATING_CACHE[cache_key] = {
        "massey": result["massey"],
        "elo": result["elo"],
        "computed_at": now,
    }
    return {
        "rating_massey": result["massey"].get(team_id, 0.0),
        "rating_elo": result["elo"].get(team_id, ELO_BASE),
    }


def clear_cache() -> None:
    _RATING_CACHE.clear()
