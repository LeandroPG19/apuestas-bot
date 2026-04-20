"""Features player props NFL (§23.3).

Por rol:
- QB: passing_yards (Gamma), passing_tds (Poisson), completions (NegBin)
- RB: rushing_yards (Gamma), carries (NegBin), receptions (NegBin)
- WR/TE: receiving_yards (Gamma), receptions (NegBin), targets (NegBin)

Clave: target_share, air_yards_share, snap_pct, opp_allowed_per_position.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from apuestas.features.common import rolling_mean_prev
from apuestas.features.weather_perf import WeatherBucket, nfl_passing_weather_multiplier

QB_METRICS = (
    "pass_attempts",
    "pass_completions",
    "pass_yards",
    "pass_tds",
    "interceptions",
    "cpoe",
    "pressure_rate_against",
)
RB_METRICS = (
    "carries",
    "rush_yards",
    "rush_tds",
    "receptions",
    "rec_yards",
    "rec_tds",
    "snap_pct",
    "red_zone_carries",
)
WR_METRICS = (
    "targets",
    "receptions",
    "rec_yards",
    "air_yards",
    "target_share",
    "air_yards_share",
    "yac",
    "snap_pct",
)


def qb_rolling_features(qb_logs: pl.DataFrame) -> pl.DataFrame:
    result = qb_logs.sort(["player_id", "game_date"])
    for m in QB_METRICS:
        if m in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=m,
                windows=[3, 5, 8],
            )
    return result


def rb_rolling_features(rb_logs: pl.DataFrame) -> pl.DataFrame:
    result = rb_logs.sort(["player_id", "game_date"])
    for m in RB_METRICS:
        if m in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=m,
                windows=[3, 5, 8],
            )
    return result


def wr_rolling_features(wr_logs: pl.DataFrame) -> pl.DataFrame:
    result = wr_logs.sort(["player_id", "game_date"])
    for m in WR_METRICS:
        if m in result.columns:
            result = rolling_mean_prev(
                result,
                by="player_id",
                order="game_date",
                value=m,
                windows=[3, 5, 8],
            )
    return result


def build_props_nfl_features(
    *,
    prop_code: str,
    player_state: dict[str, Any],
    opp_allowed: dict[str, float] | None = None,
    weather_bucket: WeatherBucket | None = None,
    game_script: dict[str, float] | None = None,
) -> dict[str, float]:
    """Features agregadas para un prop NFL específico.

    `game_script` ej. {"proj_total": 48.5, "proj_home_spread": -3.5}
    """
    feats: dict[str, float] = {}
    for key, val in player_state.items():
        if isinstance(val, int | float):
            feats[f"player_{key}"] = float(val)

    if opp_allowed:
        for k, v in opp_allowed.items():
            feats[f"opp_{k}"] = float(v)

    # Weather penalties (passing props)
    if weather_bucket and prop_code.startswith("nfl_qb_"):
        feats["weather_passing_mult"] = nfl_passing_weather_multiplier(weather_bucket)
        feats["weather_wind_strong"] = 1.0 if weather_bucket.wind in {"strong", "gale"} else 0.0
        feats["weather_precip"] = 1.0 if weather_bucket.precip != "none" else 0.0

    # Game script affecta volumen
    if game_script:
        feats["proj_total"] = float(game_script.get("proj_total", 45.0))
        feats["proj_home_spread"] = float(game_script.get("proj_home_spread", 0.0))
        # QB pass attempts up en underdog (garbage time)
        if prop_code == "nfl_qb_pass_yds":
            feats["is_underdog"] = 1.0 if feats["proj_home_spread"] > 3 else 0.0

    return feats
