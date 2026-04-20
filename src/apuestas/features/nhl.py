"""Features NHL — xG, Corsi, Fenwick, PDO, goalie GSAx (§25.2).

Poisson bivariado (Dixon-Coles hockey variant) + features LightGBM refinement.
Factor #1: goalie confirmado (save% + GSAx).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.features.common import back_to_back_flag, days_since_last, rolling_mean_prev
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

FEATURE_SET_NAME = "nhl_v1"
WINDOWS = [5, 10, 20]

TEAM_METRICS = (
    "xg_for_5v5",
    "xg_against_5v5",
    "corsi_for_pct_5v5",
    "fenwick_for_pct_5v5",
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "high_danger_for",
    "high_danger_against",
    "pdo_5v5",  # PDO = SV% + Shooting% (regression-to-mean indicator)
    "pp_pct",
    "pk_pct",
)


def team_rolling_features(team_games: pl.DataFrame) -> pl.DataFrame:
    result = team_games.sort(["team_id", "game_date"])
    for metric in TEAM_METRICS:
        if metric in result.columns:
            result = rolling_mean_prev(
                result,
                by="team_id",
                order="game_date",
                value=metric,
                windows=WINDOWS,
            )
    result = days_since_last(result, by="team_id", order="game_date", name="rest_days")
    result = back_to_back_flag(result, by="team_id", order="game_date", threshold_hours=30.0)
    return result


# ═══════════════════════ Goalie features ═══════════════════════════════


async def get_goalie_stats(player_id: int) -> dict[str, float]:
    """Recupera stats goalie de nhl_goalie_stats."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT games_played, save_pct, gsax_total, last_10_save_pct
                FROM nhl_goalie_stats WHERE player_id = :pid
                """
            ),
            {"pid": player_id},
        )
        row = result.first()
    if row is None:
        return {
            "goalie_games_played": 0.0,
            "goalie_save_pct": 0.900,
            "goalie_gsax_total": 0.0,
            "goalie_last_10_save_pct": 0.900,
        }
    return {
        "goalie_games_played": float(row.games_played or 0),
        "goalie_save_pct": float(row.save_pct or 0.900),
        "goalie_gsax_total": float(row.gsax_total or 0.0),
        "goalie_last_10_save_pct": float(row.last_10_save_pct or 0.900),
    }


async def upsert_goalie_stats(
    *,
    player_id: int,
    games_played: int,
    saves_total: int,
    shots_against_total: int,
    gsax_total: float,
    last_10_save_pct: float,
) -> None:
    sv_pct = saves_total / shots_against_total if shots_against_total > 0 else 0.0
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO nhl_goalie_stats
                  (player_id, games_played, saves_total, shots_against_total,
                   save_pct, gsax_total, last_10_save_pct, last_updated)
                VALUES (:pid, :gp, :saves, :sa, :svp, :gsax, :l10, NOW())
                ON CONFLICT (player_id) DO UPDATE SET
                  games_played = EXCLUDED.games_played,
                  saves_total = EXCLUDED.saves_total,
                  shots_against_total = EXCLUDED.shots_against_total,
                  save_pct = EXCLUDED.save_pct,
                  gsax_total = EXCLUDED.gsax_total,
                  last_10_save_pct = EXCLUDED.last_10_save_pct,
                  last_updated = NOW()
                """
            ),
            {
                "pid": player_id,
                "gp": games_played,
                "saves": saves_total,
                "sa": shots_against_total,
                "svp": sv_pct,
                "gsax": gsax_total,
                "l10": last_10_save_pct,
            },
        )


# ═══════════════════════ Poisson bivariado hockey ═══════════════════════


def hockey_poisson_bivariate(
    *,
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> np.ndarray:
    """Matriz P(home_goals=i, away_goals=j) Poisson independiente.

    Hockey tiene menor dependencia home/away que fútbol, pero una
    corrección mínima estilo Dixon-Coles baja varianza en empates 0-0.
    """
    from scipy.stats import poisson as scipy_poisson

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_h = scipy_poisson.pmf(i, lambda_home)
            p_a = scipy_poisson.pmf(j, lambda_away)
            matrix[i, j] = p_h * p_a
    # Normalizar por truncamiento
    matrix /= matrix.sum()
    return matrix


def derive_market_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """A partir de matriz conjunta, calcula mercados estándar NHL."""
    max_g = matrix.shape[0] - 1
    p_home_win = float(sum(matrix[i, j] for i in range(max_g + 1) for j in range(i)))
    p_away_win = float(sum(matrix[i, j] for i in range(max_g + 1) for j in range(i + 1, max_g + 1)))
    p_tie = float(matrix.trace())
    # Total goals distribution
    p_over_55 = float(
        sum(matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i + j > 5.5)
    )
    p_over_65 = float(
        sum(matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i + j > 6.5)
    )
    # BTTS
    p_btts_yes = float(sum(matrix[i, j] for i in range(1, max_g + 1) for j in range(1, max_g + 1)))
    return {
        "home_win": p_home_win,
        "away_win": p_away_win,
        "tie_regulation": p_tie,
        "over_5.5": p_over_55,
        "over_6.5": p_over_65,
        "btts_yes": p_btts_yes,
    }


# ═══════════════════════ Build feature frame ═════════════════════════════


async def build_nhl_features(
    *,
    home_team_id: int,
    away_team_id: int,
    home_goalie_id: int | None = None,
    away_goalie_id: int | None = None,
    team_rolling: pl.DataFrame | None = None,
) -> dict[str, float]:
    """Features para un match NHL específico."""
    feats: dict[str, float] = {}

    # Goalies (factor #1)
    if home_goalie_id:
        home_goalie = await get_goalie_stats(home_goalie_id)
        feats.update({f"home_{k}": v for k, v in home_goalie.items()})
    if away_goalie_id:
        away_goalie = await get_goalie_stats(away_goalie_id)
        feats.update({f"away_{k}": v for k, v in away_goalie.items()})

    # Diff goalie GSAx (most predictive single feature)
    if home_goalie_id and away_goalie_id:
        feats["gsax_diff"] = feats.get("home_goalie_gsax_total", 0.0) - feats.get(
            "away_goalie_gsax_total", 0.0
        )

    # Rolling team stats
    if team_rolling is not None:
        home_row = team_rolling.filter(pl.col("team_id") == home_team_id).tail(1)
        away_row = team_rolling.filter(pl.col("team_id") == away_team_id).tail(1)
        for row, prefix in ((home_row, "home"), (away_row, "away")):
            if row.height > 0:
                for col in (
                    "xg_for_5v5_roll_10",
                    "xg_against_5v5_roll_10",
                    "corsi_for_pct_5v5_roll_10",
                    "pdo_5v5_roll_10",
                    "pp_pct_roll_10",
                    "pk_pct_roll_10",
                    "rest_days",
                    "back_to_back",
                ):
                    if col in row.columns:
                        feats[f"{prefix}_{col}"] = float(row[col][0] or 0.0)

    # Home ice advantage baseline
    feats["home_ice_advantage"] = 0.25  # +0.25 goals/game literatura
    return feats
