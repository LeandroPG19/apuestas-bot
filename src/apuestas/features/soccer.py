"""Features fútbol — xG (Understat/StatsBomb/FBref) + Elo clubelo + Dixon-Coles.

Blueprint §6: librería principal `penaltyblog` (Dixon-Coles + Bivariate
Poisson + Bayesian hierarchical + decaimiento ξ=0.0018). Aquí features
complementarias para stacker LightGBM sobre residuos DC.
"""

from __future__ import annotations

import polars as pl

from apuestas.features.common import (
    days_since_last,
    rolling_mean_prev,
)

FEATURE_SET_NAME = "soccer_v1"
WINDOWS = [5, 10, 20]


TEAM_METRICS = (
    "xg_for",
    "xg_against",
    "goals_for",
    "goals_against",
    "shots_per_game",
    "shots_on_target",
    "possession_pct",
    "pass_completion_pct",
    "ppda",  # passes per defensive action
    "deep_completions",
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
    # Dixon-Coles decay weight (años desde partido)
    result = (
        result.with_columns(
            pl.col("game_date").max().over("team_id").alias("_latest_date"),
        )
        .with_columns(
            ((pl.col("_latest_date") - pl.col("game_date")).dt.total_days() / 365.0).alias(
                "years_ago"
            )
        )
        .with_columns(
            # ξ=0.0018 per day → convert años
            (-0.0018 * 365.0 * pl.col("years_ago")).exp().alias("dc_decay_weight")
        )
        .drop("_latest_date")
    )

    result = days_since_last(result, by="team_id", order="game_date", name="rest_days")
    return result


def add_elo_features(
    df: pl.DataFrame,
    elo_ratings: dict[int, float],
) -> pl.DataFrame:
    """Añade Elo overall desde dict team_id → Elo."""
    if not elo_ratings or "team_id" not in df.columns:
        return df
    elo_df = pl.DataFrame(
        [{"team_id": int(k), "elo_rating": float(v)} for k, v in elo_ratings.items()]
    )
    return df.join(elo_df, on="team_id", how="left")


def build_soccer_feature_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    elo_ratings: dict[int, float] | None = None,
) -> pl.DataFrame:
    """Pipeline completo fútbol."""
    feats = team_rolling_features(team_games)
    if elo_ratings:
        feats = add_elo_features(feats, elo_ratings)

    base_cols = [c for c in feats.columns if any(c.endswith(f"_roll_{w}") for w in WINDOWS)]
    base_cols += ["rest_days", "dc_decay_weight", "elo_rating"]
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

    # Diferenciales xG + Elo (los más predictivos)
    for m in ("xg_for_roll_10", "xg_against_roll_10", "goals_for_roll_10", "elo_rating"):
        h = f"{m}_home"
        a = f"{m}_away"
        if h in merged.columns and a in merged.columns:
            merged = merged.with_columns((pl.col(h) - pl.col(a)).alias(f"{m}_diff"))

    return merged
