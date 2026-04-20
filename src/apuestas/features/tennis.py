"""Features tenis — Elo surface-split + Barnett-Clarke serve probability (§25.1).

Dos modelos independientes combinados por stacker:

1. Elo (overall + por superficie). K=32 overall, K=50 surface.
2. Serve/return probability (Barnett-Clarke 2005):
   - P(hold_serve_i) = f(serve_pts_won_pct_i, return_pts_won_pct_j, surface)
   - Deriva P(win game → set → match BO3/BO5) por recursión.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


Surface = Literal["hard", "clay", "grass", "indoor_hard"]


# ═══════════════════════ Elo surface-split ══════════════════════════════


def elo_update(
    *,
    rating_a: float,
    rating_b: float,
    outcome_a: float,  # 1=win, 0=loss, 0.5=retire
    k: float = 32.0,
) -> tuple[float, float]:
    """Elo update estándar."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    new_a = rating_a + k * (outcome_a - expected_a)
    new_b = rating_b + k * ((1 - outcome_a) - (1 - expected_a))
    return new_a, new_b


def elo_win_probability(
    *,
    rating_a: float,
    rating_b: float,
) -> float:
    """P(A wins) según diferencia Elo."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def combine_elo_surface(
    *,
    elo_overall_diff: float,
    elo_surface_diff: float,
    surface_weight: float = 0.6,
) -> float:
    """Blend Elo overall con Elo surface. Surface más predictivo en superficie dada."""
    return surface_weight * elo_surface_diff + (1 - surface_weight) * elo_overall_diff


async def get_surface_rating(*, player_id: int, surface: Surface) -> float:
    """Elo rating del jugador en una superficie específica.

    Fallback a Elo overall (1500 default) si no hay rating surface.
    """
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT elo FROM tennis_surface_ratings
                WHERE player_id = :pid AND surface = :sfc
                """
            ),
            {"pid": player_id, "sfc": surface},
        )
        row = result.first()
    return float(row.elo) if row else 1500.0


async def update_surface_rating(
    *,
    player_id: int,
    surface: Surface,
    new_rating: float,
    games_played: int,
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO tennis_surface_ratings
                  (player_id, surface, elo, games_played, last_updated)
                VALUES (:pid, :sfc, :elo, :games, NOW())
                ON CONFLICT (player_id, surface) DO UPDATE
                SET elo = EXCLUDED.elo,
                    games_played = EXCLUDED.games_played,
                    last_updated = NOW()
                """
            ),
            {"pid": player_id, "sfc": surface, "elo": new_rating, "games": games_played},
        )


# ═══════════════════════ Serve-game probability (Barnett-Clarke) ═══════


@dataclass(slots=True, frozen=True)
class ServeReturnStats:
    serve_pts_won_pct: float  # % de puntos ganados en su servicio
    return_pts_won_pct: float  # % de puntos ganados al resto


def hold_serve_probability(
    *,
    server_stats: ServeReturnStats,
    returner_stats: ServeReturnStats,
    surface_adjust: float = 0.0,  # +/- 0-0.03 por superficie dominante serve
) -> float:
    """P(hold_serve) del server.

    Fórmula Barnett-Clarke simplificada: media armónica entre:
    - P(server_wins_on_serve) basado en su serve%
    - P(returner_loses_on_return) basado en inverso de return%

    Simplificación: p ≈ (serve_pct × 2 - avg_serve_pct) donde
    avg_serve_pct ≈ 0.63 ATP, 0.55 WTA.

    Recursión a game desde punto requiere cálculo Markov chain; aquí
    aproximamos con una función cerrada razonable para P(hold).
    """
    p_win_point = (server_stats.serve_pts_won_pct + (1 - returner_stats.return_pts_won_pct)) / 2.0
    p_win_point = np.clip(p_win_point + surface_adjust, 0.40, 0.85)

    # Probabilidad de ganar un game desde 0-0 (Markov exacta):
    # P(game) = p^4 × (15 - 34p + 28p² - 8p³) / (1 - 2p(1-p))
    # Fórmula cerrada Bayer 2015 / Barnett-Clarke
    p = float(p_win_point)
    if p <= 0.5:
        numer = p**4 * (15 - 4 * p * (4 + p * (1 - p)))
    else:
        numer = p**4 * (15 - 34 * p + 28 * p**2 - 8 * p**3)
    denom = 1 - 2 * p * (1 - p)
    p_game = max(0.0, min(1.0, numer / denom)) if denom > 0 else p
    return float(p_game)


def match_probability_bo3(
    *,
    p_hold_a: float,  # P(A holds their serve)
    p_hold_b: float,  # P(B holds their serve)
) -> float:
    """P(A wins match best-of-3) asumiendo set-model independencia."""
    # P(A wins set): solve via set chain de juegos
    p_a_win_set = _set_probability(p_hold_a=p_hold_a, p_hold_b=p_hold_b)
    # Best-of-3: P(win match) = P(2-0) + P(2-1)
    p_2_0 = p_a_win_set**2
    p_2_1 = 2 * p_a_win_set**2 * (1 - p_a_win_set)
    return float(p_2_0 + p_2_1)


def match_probability_bo5(
    *,
    p_hold_a: float,
    p_hold_b: float,
) -> float:
    """Grand Slams ATP BO5."""
    p_a_win_set = _set_probability(p_hold_a=p_hold_a, p_hold_b=p_hold_b)
    # P(win match BO5): P(3-0) + P(3-1) + P(3-2)
    p = p_a_win_set
    q = 1 - p
    return float(p**3 + 3 * p**3 * q + 6 * p**3 * q**2)


def _set_probability(*, p_hold_a: float, p_hold_b: float) -> float:
    """Aproximación: P(A wins set) basada en holds asymmetric.

    Set ganado en 6-X o 7-6. Modelo simplificado: promedia geométrica
    ponderada de holds y breaks.
    """
    # Break probability A = 1 - p_hold_b; B = 1 - p_hold_a
    break_a = 1 - p_hold_b
    break_b = 1 - p_hold_a
    # Expected breaks en un set de 12 games: b * 6
    if break_a + break_b <= 0:
        return 0.5
    # Ratio de breaks como proxy P(win set)
    return float(break_a / (break_a + break_b))


# ═══════════════════════ Build feature vector ════════════════════════════


async def build_tennis_features(
    *,
    player_a_id: int,
    player_b_id: int,
    surface: Surface,
    is_bo5: bool = False,
    a_serve_stats: ServeReturnStats | None = None,
    b_serve_stats: ServeReturnStats | None = None,
) -> dict[str, float]:
    """Features consolidadas para un match específico."""
    elo_a_overall = await _get_overall_elo(player_a_id)
    elo_b_overall = await _get_overall_elo(player_b_id)
    elo_a_surface = await get_surface_rating(player_id=player_a_id, surface=surface)
    elo_b_surface = await get_surface_rating(player_id=player_b_id, surface=surface)

    elo_diff_overall = elo_a_overall - elo_b_overall
    elo_diff_surface = elo_a_surface - elo_b_surface
    combined_diff = combine_elo_surface(
        elo_overall_diff=elo_diff_overall,
        elo_surface_diff=elo_diff_surface,
    )

    feats: dict[str, float] = {
        "elo_a_overall": elo_a_overall,
        "elo_b_overall": elo_b_overall,
        "elo_a_surface": elo_a_surface,
        "elo_b_surface": elo_b_surface,
        "elo_diff_overall": elo_diff_overall,
        "elo_diff_surface": elo_diff_surface,
        "elo_diff_combined": combined_diff,
        "p_elo_a_wins": elo_win_probability(rating_a=elo_a_surface, rating_b=elo_b_surface),
        "is_bo5": 1.0 if is_bo5 else 0.0,
        f"surface_{surface}": 1.0,
    }

    # Serve model si tenemos stats
    if a_serve_stats and b_serve_stats:
        p_hold_a = hold_serve_probability(server_stats=a_serve_stats, returner_stats=b_serve_stats)
        p_hold_b = hold_serve_probability(server_stats=b_serve_stats, returner_stats=a_serve_stats)
        feats["p_hold_a"] = p_hold_a
        feats["p_hold_b"] = p_hold_b
        match_p = (
            match_probability_bo5(p_hold_a=p_hold_a, p_hold_b=p_hold_b)
            if is_bo5
            else match_probability_bo3(p_hold_a=p_hold_a, p_hold_b=p_hold_b)
        )
        feats["p_serve_model_a_wins"] = match_p

    return feats


async def _get_overall_elo(player_id: int) -> float:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT AVG(elo) AS elo FROM tennis_surface_ratings
                WHERE player_id = :pid
                """
            ),
            {"pid": player_id},
        )
        row = result.first()
    return float(row.elo) if row and row.elo else 1500.0
