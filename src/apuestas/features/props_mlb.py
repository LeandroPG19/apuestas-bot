"""Features player props MLB (§23.3).

Para props-bateador usa Monte Carlo PA simulator de ml/props_distributions.py.
Para props-pitcher usa NegBinomial sobre rolling K rates.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from apuestas.features.common import rolling_mean_prev
from apuestas.features.weather_perf import WeatherBucket, mlb_hr_weather_multiplier
from apuestas.ml.props_distributions import (
    EmpiricalDist,
    monte_carlo_plate_appearances,
)


def batter_features(batter_logs: pl.DataFrame) -> pl.DataFrame:
    metrics = (
        "xwoba",
        "barrel_rate",
        "hard_hit_pct",
        "isolated_power",
        "launch_angle_avg",
        "exit_velocity_avg",
    )
    result = batter_logs.sort(["player_id", "game_date"])
    for m in metrics:
        if m in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=m,
                windows=[30, 60],
            )
    return result


def pitcher_features(pitcher_logs: pl.DataFrame) -> pl.DataFrame:
    metrics = (
        "k_per_9",
        "bb_per_9",
        "hr_per_9",
        "fip",
        "xfip",
        "stuff_plus",
        "swinging_strike_pct",
    )
    result = pitcher_logs.sort(["player_id", "game_date"])
    for m in metrics:
        if m in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=m,
                windows=[3, 5, 10],
            )
    return result


def simulate_batter_props(
    *,
    xwoba_batter: float,
    pitcher_k_rate: float,
    pitcher_bb_rate: float,
    n_pa_projected: int = 4,
    park_factor_hr: float = 1.0,
    weather_bucket: WeatherBucket | None = None,
    n_simulations: int = 10_000,
) -> dict[str, EmpiricalDist]:
    """Monte Carlo PA (§19.4): genera distribuciones empíricas por prop."""
    # Ajuste weather multiplier
    weather_mult = mlb_hr_weather_multiplier(weather_bucket) if weather_bucket else 1.0
    wind_out_bonus = 0.0
    if weather_bucket and weather_bucket.wind_dir == "out_to_rf_cf":
        wind_out_bonus = 0.15 if weather_bucket.wind == "strong" else 0.08

    return monte_carlo_plate_appearances(
        n_pa=n_pa_projected,
        xwoba_batter=xwoba_batter,
        pitcher_k_rate=pitcher_k_rate,
        pitcher_bb_rate=pitcher_bb_rate,
        park_factor_hr=park_factor_hr * weather_mult,
        wind_to_of_pct=wind_out_bonus,
        n_simulations=n_simulations,
    )


def build_props_mlb_features(
    *,
    prop_code: str,
    batter_state: dict[str, Any] | None = None,
    pitcher_state: dict[str, Any] | None = None,
    park_factors: dict[str, float] | None = None,
    weather_bucket: WeatherBucket | None = None,
) -> dict[str, float]:
    """Features consolidadas por prop_code MLB para ML model."""
    feats: dict[str, float] = {}

    if prop_code.startswith("mlb_pitcher_"):
        if pitcher_state:
            for k in ("k_per_9", "bb_per_9", "fip", "stuff_plus"):
                if k in pitcher_state:
                    feats[f"pitcher_{k}"] = float(pitcher_state[k] or 0.0)
        return feats

    # Bateador
    if batter_state:
        for k in ("xwoba", "barrel_rate", "hard_hit_pct", "isolated_power"):
            if k in batter_state:
                feats[f"batter_{k}"] = float(batter_state[k] or 0.0)
    if park_factors:
        feats["park_hr_factor"] = float(park_factors.get("hr", 1.0))
    if weather_bucket:
        feats["weather_hr_multiplier"] = mlb_hr_weather_multiplier(weather_bucket)
        feats["weather_is_indoor"] = 1.0 if weather_bucket.is_indoor else 0.0
        feats["weather_wind_out"] = 1.0 if weather_bucket.wind_dir == "out_to_rf_cf" else 0.0
    return feats
