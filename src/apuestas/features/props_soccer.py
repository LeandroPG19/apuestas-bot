"""Features player props fútbol (§23.3).

- anytime_goal: Dixon-Coles matriz × share xG del jugador.
- shots / shots_on_target: NegBinomial con rolling del jugador.
- yellow_cards: Bernoulli ajustada por árbitro (fouls_avg) + team_fouls.
- assists: Bernoulli con key_passes rolling + team xA share.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from apuestas.features.common import rolling_mean_prev

PLAYER_METRICS = (
    "shots",
    "shots_on_target",
    "goals",
    "key_passes",
    "xg_per_90",
    "xa_per_90",
    "assists",
    "minutes_played",
    "fouls_committed",
    "yellow_cards",
)


def player_rolling_features(player_logs: pl.DataFrame) -> pl.DataFrame:
    result = player_logs.sort(["player_id", "game_date"])
    for m in PLAYER_METRICS:
        if m in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=m,
                windows=[5, 10, 20],
            )
    return result


def goal_probability_from_dc(
    *,
    team_goals_matrix: list[list[float]],  # P(h=i, a=j) Dixon-Coles
    is_home: bool,
    player_xg_share: float,
) -> float:
    """P(anytime_goalscorer) derivada del matriz Dixon-Coles.

    P(player scores >=1) ≈ 1 - P(team scores 0) × (1 - player_xg_share_team)
    Si team es home, sumar columna 0 (team_goals=0); si away, fila 0.
    """
    if is_home:
        p_team_zero = sum(team_goals_matrix[0])  # home row 0 = 0 home goals
    else:
        p_team_zero = sum(row[0] for row in team_goals_matrix)  # away col 0

    # team score >= 1
    p_team_scores = 1 - p_team_zero
    # share del jugador: entre los goles del equipo, cuántos son suyos
    p_player = 1 - (1 - player_xg_share) ** (3.0 * p_team_scores)  # heurística
    return max(0.0, min(1.0, p_player))


def referee_card_adjustment(
    referee_yellow_avg: float | None,
    base_league_avg: float = 4.3,
) -> float:
    """Multiplicador sobre P(player card) por árbitro."""
    if referee_yellow_avg is None or referee_yellow_avg <= 0:
        return 1.0
    return float(referee_yellow_avg / base_league_avg)


def build_props_soccer_features(
    *,
    prop_code: str,
    player_state: dict[str, Any],
    team_matrix: list[list[float]] | None = None,
    is_home: bool = True,
    referee_yellow_avg: float | None = None,
    opp_keeper_sv_pct: float | None = None,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    for k, v in player_state.items():
        if isinstance(v, int | float):
            feats[f"player_{k}"] = float(v)

    if prop_code == "soccer_anytime_goal" and team_matrix is not None:
        share = float(player_state.get("xg_share_team", 0.15))
        feats["dc_p_anytime_goal"] = goal_probability_from_dc(
            team_goals_matrix=team_matrix,
            is_home=is_home,
            player_xg_share=share,
        )
    if prop_code == "soccer_cards":
        feats["ref_card_adjustment"] = referee_card_adjustment(referee_yellow_avg)
    if prop_code == "soccer_shots_on_target" and opp_keeper_sv_pct is not None:
        feats["opp_keeper_sv_pct"] = float(opp_keeper_sv_pct)
    feats["is_home"] = 1.0 if is_home else 0.0
    return feats
