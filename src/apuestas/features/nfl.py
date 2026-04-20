"""Features NFL — EPA/CPOE (nflfastR), DVOA, pressure rate.

Blueprint §6 referencia canónica Baldwin & Carl 2020 nflfastR.
Stats clave:
- EPA/play (Expected Points Added)
- CPOE (Completion % Over Expected)
- Success rate
- Pressure rate allowed (QB)
- DVOA (Football Outsiders, si disponible)
"""

from __future__ import annotations

import polars as pl

from apuestas.features.common import (
    days_since_last,
    rolling_mean_prev,
)

FEATURE_SET_NAME = "nfl_v1"
WINDOWS = [3, 5, 8]  # semanas NFL (season 18 juegos regular)


TEAM_METRICS = (
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate",
    "off_pass_epa",
    "off_rush_epa",
    "cpoe_avg",
    "pressure_rate_for",
    "pressure_rate_against",
    "turnovers",
    "penalties_per_game",
    "third_down_conv",
)


def team_rolling_features(team_games: pl.DataFrame) -> pl.DataFrame:
    """Rolling 3/5/8 games sobre EPA + success rate."""
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
    # Rest days + mini-bye
    result = days_since_last(result, by="team_id", order="game_date", name="rest_days")
    # B2B no aplica NFL pero Thursday night post-Sunday es similar
    result = result.with_columns((pl.col("rest_days") < 5).cast(pl.Int8).alias("short_week"))
    # Post-bye week boost conocido
    result = result.with_columns((pl.col("rest_days") >= 10).cast(pl.Int8).alias("post_bye_week"))
    return result


def build_nfl_feature_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
) -> pl.DataFrame:
    feats = team_rolling_features(team_games)
    base_cols = [c for c in feats.columns if any(c.endswith(f"_roll_{w}") for w in WINDOWS)]
    base_cols += ["rest_days", "short_week", "post_bye_week"]
    base_cols = [c for c in base_cols if c in feats.columns]

    home_renamed = feats.select(
        pl.col("team_id").alias("home_team_id"),
        pl.col("game_date").alias("start_time"),
        *[pl.col(c).alias(f"{c}_home") for c in base_cols],
    )
    away_renamed = feats.select(
        pl.col("team_id").alias("away_team_id"),
        pl.col("game_date").alias("start_time"),
        *[pl.col(c).alias(f"{c}_away") for c in base_cols],
    )

    merged = matches.join(home_renamed, on=["home_team_id", "start_time"], how="left")
    merged = merged.join(away_renamed, on=["away_team_id", "start_time"], how="left")

    # Diferenciales críticos NFL
    for m in (
        "off_epa_per_play_roll_5",
        "def_epa_per_play_roll_5",
        "off_success_rate_roll_5",
        "cpoe_avg_roll_5",
    ):
        h = f"{m}_home"
        a = f"{m}_away"
        if h in merged.columns and a in merged.columns:
            merged = merged.with_columns((pl.col(h) - pl.col(a)).alias(f"{m}_diff"))

    return merged
