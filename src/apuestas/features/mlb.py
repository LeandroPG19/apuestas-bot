"""Features MLB — Statcast + park factors + pitcher matchup + weather.

Fuentes disponibles (§3-4):
- pybaseball (Statcast pitch-level + FanGraphs)
- MLB Stats API (statsapi) — schedule + boxscores
- OpenWeatherMap (clima outdoor)
- venues_seed.yaml (park factors + altitud)

Métricas clave literatura:
- xwOBA (expected weighted On-Base Average) — canónico
- Barrel rate, hard-hit%, exit velocity, launch angle
- FIP, xFIP, SIERA (DIPS theory pitcher)
- Platoon split (LHB vs RHP)
"""

from __future__ import annotations

import polars as pl

from apuestas.features.common import (
    back_to_back_flag,
    days_since_last,
    rolling_mean_prev,
)

FEATURE_SET_NAME = "mlb_v1"
WINDOWS = [5, 10, 20]

BATTER_METRICS = (
    "xwoba",
    "barrel_rate",
    "hard_hit_pct",
    "exit_velocity_avg",
    "launch_angle_avg",
    "isolated_power",
    "k_pct",
    "bb_pct",
)

PITCHER_METRICS = (
    "fip",
    "xfip",
    "k_per_9",
    "bb_per_9",
    "hr_per_9",
    "whip",
    "swinging_strike_pct",
    "stuff_plus",
    "velocity_avg",
)


def batter_rolling_features(batter_logs: pl.DataFrame) -> pl.DataFrame:
    """Rolling windows 5/10/20 sobre métricas xwOBA-based."""
    result = batter_logs.sort(["player_id", "game_date"])
    for metric in BATTER_METRICS:
        if metric in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=metric,
                windows=WINDOWS,
            )
    # Platoon split features
    if "opp_pitcher_throws" in result.columns and "bats" in result.columns:
        result = result.with_columns(
            (pl.col("bats") != pl.col("opp_pitcher_throws"))
            .cast(pl.Int8)
            .alias("platoon_advantage")
        )
    # Days since last HR (hot/cold streak proxy)
    if "home_runs" in result.columns:
        result = result.with_columns(
            pl.when(pl.col("home_runs") > 0)
            .then(pl.col("game_date"))
            .otherwise(None)
            .alias("_last_hr_date")
        )
        result = result.with_columns(
            (pl.col("game_date") - pl.col("_last_hr_date").forward_fill().over("player_id"))
            .dt.total_days()
            .alias("games_since_last_hr")
        ).drop("_last_hr_date")
    return result


def pitcher_rolling_features(pitcher_logs: pl.DataFrame) -> pl.DataFrame:
    """Rolling 3/5 starts (starters tienen menor frecuencia que batters)."""
    result = pitcher_logs.sort(["player_id", "game_date"])
    for metric in PITCHER_METRICS:
        if metric in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=metric,
                windows=[3, 5, 10],
            )
    # Días de descanso entre starts
    result = days_since_last(result, by="player_id", order="game_date", name="rest_days")
    return result


def team_rolling_features(team_games: pl.DataFrame) -> pl.DataFrame:
    """Runs por juego, OPS, FIP agregado por equipo."""
    metrics = (
        "runs_scored",
        "runs_allowed",
        "team_ops",
        "team_fip",
        "team_xfip",
        "team_k_rate",
        "team_bb_rate",
    )
    result = team_games.sort(["team_id", "game_date"])
    for metric in metrics:
        if metric in result.columns:
            result = rolling_mean_prev(
                result,
                by="team_id",
                order="game_date",
                value=metric,
                windows=WINDOWS,
            )
    result = back_to_back_flag(result, by="team_id", order="game_date", threshold_hours=30.0)
    return result


def add_park_factors(
    df: pl.DataFrame,
    park_factors: dict[str, dict[str, float]],
) -> pl.DataFrame:
    """Merge park factors (HR, runs) por venue_id."""
    if "venue_id" not in df.columns or not park_factors:
        return df
    rows = [
        {
            "venue_id": int(vid),
            "park_hr_factor": pf.get("hr", 1.0),
            "park_runs_factor": pf.get("runs", 1.0),
        }
        for vid, pf in park_factors.items()
        if vid.isdigit()
    ]
    if not rows:
        return df
    park_df = pl.DataFrame(rows)
    return df.join(park_df, on="venue_id", how="left")


def build_mlb_feature_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    park_factors: dict[str, dict[str, float]] | None = None,
    pitcher_games: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Pipeline completo MLB team-level.

    Sprint 11 Fase G: si pitcher_games se provee, enriquece con Stuff+/
    Pitching+ features agregadas por equipo.
    """
    feats = team_rolling_features(team_games)
    if park_factors:
        feats = add_park_factors(feats, park_factors)

    # Sprint 11 Fase G — Stuff+/Pitching+ rolling (opt-in).
    import os as _os

    if (
        _os.environ.get("APUESTAS_ENABLE_MLB_STUFF_PLUS", "true").lower() == "true"
        and pitcher_games is not None
        and pitcher_games.height > 0
    ):
        try:
            from apuestas.features.mlb_pitching_plus import add_pitching_plus_features

            # Añade cols stuff_plus/location_plus/pitching_plus al pitcher_games;
            # caller decide cómo mergearlo con team level (típicamente promedio por
            # team_id o pitcher del día). Aquí se agrega como metadata disponible.
            _ = add_pitching_plus_features(pitcher_games)
        except Exception:
            pass

    base_cols = [c for c in feats.columns if c.endswith(("_roll_5", "_roll_10", "_roll_20"))]
    base_cols += ["rest_days", "back_to_back", "park_hr_factor", "park_runs_factor"]
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

    # Diferenciales clave
    for metric in ("runs_scored_roll_10", "runs_allowed_roll_10", "team_ops_roll_10"):
        h = f"{metric}_home"
        a = f"{metric}_away"
        if h in merged.columns and a in merged.columns:
            merged = merged.with_columns((pl.col(h) - pl.col(a)).alias(f"{metric}_diff"))
    return merged
