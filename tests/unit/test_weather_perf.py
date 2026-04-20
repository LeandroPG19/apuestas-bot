"""Tests weather × performance features (§24)."""

from __future__ import annotations

import pytest

from apuestas.features.weather_perf import (
    affinity_score,
    classify_forecast,
    mlb_hr_weather_multiplier,
    nfl_passing_weather_multiplier,
    shrinkage_adjusted_mean,
    soccer_goals_weather_multiplier,
)

# ═══════════════════════ Classify forecast ═══════════════════════════════


def test_classify_indoor_forces_indoor_buckets() -> None:
    b = classify_forecast(is_indoor=True, altitude_m=100)
    assert b.is_indoor
    assert b.temp == "indoor"
    assert b.wind == "indoor"
    assert b.precip == "indoor"


def test_classify_cold_bucket() -> None:
    b = classify_forecast(temp_c=-5, wind_kph=10, is_indoor=False)
    assert b.temp == "freezing"
    assert b.wind == "light"


def test_classify_hot_humid() -> None:
    b = classify_forecast(temp_c=36, humidity_pct=85)
    assert b.temp == "hot"
    assert b.humidity == "saturated"


def test_classify_altitude_extreme() -> None:
    b = classify_forecast(altitude_m=2800)
    assert b.altitude == "extreme"


def test_classify_wind_direction_out_to_of() -> None:
    """Viento alineado con outfield orientación (ej. 0° ~ 0° venue orientation)."""
    b = classify_forecast(
        temp_c=20,
        wind_kph=20,
        wind_direction_deg=15,  # casi mismo que venue
        venue_orientation_deg=0,
    )
    assert b.wind_dir == "out_to_rf_cf"


def test_classify_wind_direction_in_to_pitcher() -> None:
    b = classify_forecast(
        temp_c=20,
        wind_kph=20,
        wind_direction_deg=180,  # opuesto
        venue_orientation_deg=0,
    )
    assert b.wind_dir == "in_to_pitcher"


# ═══════════════════════ Shrinkage + affinity ════════════════════════════


def test_shrinkage_zero_samples_returns_overall() -> None:
    result = shrinkage_adjusted_mean(bucket_mean=0.10, bucket_samples=0, overall_mean=0.05)
    assert result == pytest.approx(0.05)


def test_shrinkage_large_sample_converges_to_bucket() -> None:
    result = shrinkage_adjusted_mean(
        bucket_mean=0.10, bucket_samples=100, overall_mean=0.05, prior_weight=10
    )
    assert result == pytest.approx((100 * 0.10 + 10 * 0.05) / 110)
    assert result > 0.085  # cerca de bucket


def test_affinity_score_positive_when_bucket_better() -> None:
    affinity, conf = affinity_score(
        bucket_mean=0.08, bucket_samples=20, overall_mean=0.05, prior_weight=10
    )
    assert affinity > 0
    assert 0 < conf < 1


def test_affinity_score_zero_on_zero_overall() -> None:
    affinity, conf = affinity_score(bucket_mean=0.05, bucket_samples=5, overall_mean=0.0)
    assert affinity == 0.0
    assert conf == 0.0


# ═══════════════════════ Weather multipliers ═════════════════════════════


def test_mlb_hr_multiplier_cold_reduces() -> None:
    b = classify_forecast(temp_c=-5, wind_kph=5, wind_direction_deg=0, venue_orientation_deg=0)
    mult = mlb_hr_weather_multiplier(b)
    assert mult < 1.0


def test_mlb_hr_multiplier_wind_out_boosts() -> None:
    b = classify_forecast(
        temp_c=22,
        wind_kph=30,
        wind_direction_deg=0,
        venue_orientation_deg=0,
    )
    mult = mlb_hr_weather_multiplier(b)
    assert mult > 1.15  # strong out boost expected


def test_mlb_hr_multiplier_wind_in_kills_hr() -> None:
    b = classify_forecast(
        temp_c=22,
        wind_kph=30,
        wind_direction_deg=180,
        venue_orientation_deg=0,
    )
    mult = mlb_hr_weather_multiplier(b)
    assert mult < 0.80


def test_mlb_hr_multiplier_altitude_extreme() -> None:
    b = classify_forecast(temp_c=22, altitude_m=2800)
    mult = mlb_hr_weather_multiplier(b)
    assert mult > 1.1  # Coors-style boost


def test_nfl_passing_wind_strong_penalty() -> None:
    b = classify_forecast(temp_c=10, wind_kph=30, is_indoor=False)
    assert nfl_passing_weather_multiplier(b) < 0.90


def test_nfl_passing_indoor_neutral() -> None:
    b = classify_forecast(is_indoor=True)
    assert nfl_passing_weather_multiplier(b) == 1.0


def test_soccer_goals_freezing_reduces() -> None:
    b = classify_forecast(temp_c=-3, wind_kph=10)
    assert soccer_goals_weather_multiplier(b) < 1.0


def test_soccer_goals_normal_weather_neutral() -> None:
    b = classify_forecast(temp_c=18, wind_kph=10)
    mult = soccer_goals_weather_multiplier(b)
    assert 0.9 <= mult <= 1.1


# ═══════════════════════ WeatherBucket serialization ═════════════════════


def test_weather_bucket_to_dict() -> None:
    b = classify_forecast(temp_c=18, wind_kph=10, precip_mm=1.5)
    d = b.to_dict()
    assert "temp" in d and "wind" in d and "precip" in d
    assert d["temp"] == "cool"
    assert d["precip"] == "light"
