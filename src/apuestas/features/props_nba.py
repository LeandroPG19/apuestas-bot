"""Features específicas de player props NBA (§23.3).

Inputs esperados:
- player_game_logs (stats por game + minutes)
- team_stats_rolling_{home,away}
- lineups (para detectar cambios starter)
- injuries

Output: DataFrame con features para train_props.py por prop_code.
"""

from __future__ import annotations

import polars as pl

from apuestas.features.common import rolling_mean_prev

PROP_STAT_MAP = {
    "nba_points": "points",
    "nba_rebounds": "rebounds",
    "nba_assists": "assists",
    "nba_threes": "fg3m",
    "nba_steals": "steals",
    "nba_blocks": "blocks",
    "nba_pra": None,  # computed = p + r + a
}

WINDOWS = [5, 10, 20]


def minutes_projection(
    player_logs: pl.DataFrame,
    *,
    injury_status_by_player: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Proyecta minutes con: last-N avg + ajuste por starter_change."""
    result = player_logs.sort(["player_id", "game_date"])
    result = rolling_mean_prev(
        result,
        by="player_id",
        order="game_date",
        value="minutes_played",
        windows=[5, 10, 20],
    )
    # Ajuste si teammate lesionado (roster injury → minutes up)
    if injury_status_by_player:
        result = result.with_columns(
            pl.col("team_id")
            .map_elements(
                lambda tid: 1.0,  # placeholder; en prod consulta players OUT del team
                return_dtype=pl.Float64,
            )
            .alias("teammate_out_factor")
        )
    else:
        result = result.with_columns(pl.lit(1.0).alias("teammate_out_factor"))

    # minutes_proj = avg_10 × teammate_out_factor
    if "minutes_played_roll_10" in result.columns:
        result = result.with_columns(
            (pl.col("minutes_played_roll_10") * pl.col("teammate_out_factor")).alias("minutes_proj")
        )
    return result


def usage_rate_rolling(player_logs: pl.DataFrame) -> pl.DataFrame:
    """Usage rate: (FGA + 0.44 × FTA + TOV) / team_possessions."""
    if not all(c in player_logs.columns for c in ("fga", "fta", "turnovers")):
        return player_logs
    result = player_logs.with_columns(
        (pl.col("fga") + 0.44 * pl.col("fta") + pl.col("turnovers")).alias("_touches")
    )
    result = rolling_mean_prev(
        result,
        by="player_id",
        order="game_date",
        value="_touches",
        windows=WINDOWS,
    )
    return result.drop("_touches")


def build_props_nba_features(
    player_logs: pl.DataFrame,
    *,
    prop_code: str,
    opp_allowed_by_position: dict[int, dict[str, float]] | None = None,
    injury_status_by_player: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Pipeline: genera features para un prop_code específico NBA."""
    stat = PROP_STAT_MAP.get(prop_code)
    if stat is None and prop_code == "nba_pra":
        # Computed combo
        if all(c in player_logs.columns for c in ("points", "rebounds", "assists")):
            player_logs = player_logs.with_columns(
                (pl.col("points") + pl.col("rebounds") + pl.col("assists")).alias("pra")
            )
            stat = "pra"
        else:
            return pl.DataFrame()
    if stat is None:
        return pl.DataFrame()

    result = player_logs.sort(["player_id", "game_date"])
    # Rolling de la métrica target
    result = rolling_mean_prev(
        result,
        by="player_id",
        order="game_date",
        value=stat,
        windows=WINDOWS,
    )
    # Minutes projection
    result = minutes_projection(result, injury_status_by_player=injury_status_by_player)
    # Usage rate
    result = usage_rate_rolling(result)

    # Opponent allowed per position
    if opp_allowed_by_position and "position" in result.columns and "opp_team_id" in result.columns:

        def _allowed(row: dict) -> float:
            team = opp_allowed_by_position.get(int(row.get("opp_team_id") or 0), {})
            return float(team.get(str(row.get("position") or ""), 0.0))

        result = result.with_columns(
            pl.struct(["opp_team_id", "position"])
            .map_elements(_allowed, return_dtype=pl.Float64)
            .alias(f"opp_allowed_{stat}")
        )

    # Rolling form_vs_season (lucky detector)
    if f"{stat}_roll_10" in result.columns and f"{stat}_roll_20" in result.columns:
        result = result.with_columns(
            (pl.col(f"{stat}_roll_10") - pl.col(f"{stat}_roll_20")).alias(f"{stat}_form_delta")
        )

    return result
