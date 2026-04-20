"""Features clima × rendimiento del jugador (§24).

Pipeline:
1. Clasifica el weather forecast del evento en buckets discretos.
2. Consulta `player_weather_splits` materialized view por jugador+bucket.
3. Computa affinity score con shrinkage bayesiano (anti-overfit).
4. Devuelve features listas para inyectar al modelo props ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════ Buckets ═══════════════════════════════════════

TempBucket = Literal["freezing", "cold", "cool", "mild", "warm", "hot", "indoor"]
WindBucket = Literal["calm", "light", "moderate", "strong", "gale", "indoor"]
WindDirBucket = Literal["in_to_pitcher", "out_to_rf_cf", "cross", "neutral", "indoor"]
PrecipBucket = Literal["none", "light", "moderate", "heavy", "indoor"]
HumidityBucket = Literal["dry", "moderate", "humid", "saturated", "indoor"]
AltitudeBucket = Literal["sea_level", "moderate", "high", "extreme"]


@dataclass(slots=True, frozen=True)
class WeatherBucket:
    temp: TempBucket
    wind: WindBucket
    wind_dir: WindDirBucket
    precip: PrecipBucket
    humidity: HumidityBucket
    altitude: AltitudeBucket
    is_indoor: bool

    def to_dict(self) -> dict[str, str]:
        return {
            "temp": self.temp,
            "wind": self.wind,
            "wind_dir": self.wind_dir,
            "precip": self.precip,
            "humidity": self.humidity,
            "altitude": self.altitude,
        }


def _classify_temp(temp_c: float | None) -> TempBucket:
    if temp_c is None:
        return "mild"
    if temp_c < 0:
        return "freezing"
    if temp_c < 10:
        return "cold"
    if temp_c < 20:
        return "cool"
    if temp_c < 28:
        return "mild"
    if temp_c < 35:
        return "warm"
    return "hot"


def _classify_wind(wind_kph: float | None) -> WindBucket:
    if wind_kph is None:
        return "calm"
    if wind_kph < 5:
        return "calm"
    if wind_kph < 15:
        return "light"
    if wind_kph < 25:
        return "moderate"
    if wind_kph < 40:
        return "strong"
    return "gale"


def _classify_wind_dir(
    wind_direction_deg: int | None, venue_orientation_deg: int | None = None
) -> WindDirBucket:
    """Calcula dirección relativa al home plate → outfield (MLB).

    venue_orientation_deg = ángulo desde home plate a CF. Si no hay,
    devolvemos 'neutral' (otros deportes).
    """
    if wind_direction_deg is None or venue_orientation_deg is None:
        return "neutral"
    # Diferencia angular wrap-around
    diff = (wind_direction_deg - venue_orientation_deg) % 360
    if diff > 180:
        diff -= 360
    diff_abs = abs(diff)
    if diff_abs <= 30:
        return "out_to_rf_cf"  # viento sale al outfield (boost HR)
    if diff_abs >= 150:
        return "in_to_pitcher"  # viento entra al pitcher (mata HR)
    return "cross"


def _classify_precip(precip_mm: float | None) -> PrecipBucket:
    if precip_mm is None or precip_mm == 0:
        return "none"
    if precip_mm < 2:
        return "light"
    if precip_mm < 7:
        return "moderate"
    return "heavy"


def _classify_humidity(humidity_pct: int | None) -> HumidityBucket:
    if humidity_pct is None:
        return "moderate"
    if humidity_pct < 40:
        return "dry"
    if humidity_pct < 60:
        return "moderate"
    if humidity_pct < 80:
        return "humid"
    return "saturated"


def _classify_altitude(altitude_m: int | None) -> AltitudeBucket:
    if altitude_m is None:
        return "sea_level"
    if altitude_m < 200:
        return "sea_level"
    if altitude_m < 1500:
        return "moderate"
    if altitude_m < 2500:
        return "high"
    return "extreme"


def classify_forecast(
    *,
    temp_c: float | None = None,
    wind_kph: float | None = None,
    wind_direction_deg: int | None = None,
    venue_orientation_deg: int | None = None,
    precip_mm: float | None = None,
    humidity_pct: int | None = None,
    altitude_m: int | None = None,
    is_indoor: bool = False,
) -> WeatherBucket:
    """Clasifica un pronóstico en un WeatherBucket canónico.

    Si `is_indoor=True`, todas las dimensiones se fijan a 'indoor' excepto
    altitud (afecta incluso en dome, caso Denver/Mexico).
    """
    if is_indoor:
        return WeatherBucket(
            temp="indoor",
            wind="indoor",
            wind_dir="indoor",
            precip="indoor",
            humidity="indoor",
            altitude=_classify_altitude(altitude_m),
            is_indoor=True,
        )
    return WeatherBucket(
        temp=_classify_temp(temp_c),
        wind=_classify_wind(wind_kph),
        wind_dir=_classify_wind_dir(wind_direction_deg, venue_orientation_deg),
        precip=_classify_precip(precip_mm),
        humidity=_classify_humidity(humidity_pct),
        altitude=_classify_altitude(altitude_m),
        is_indoor=False,
    )


# ═══════════════════════ Shrinkage bayesiano ═══════════════════════════


def shrinkage_adjusted_mean(
    *,
    bucket_mean: float,
    bucket_samples: int,
    overall_mean: float,
    prior_weight: float = 10.0,
) -> float:
    """Media ajustada con shrinkage bayesiano hacia overall_mean.

    Formula: (n × bucket + k × overall) / (n + k)
    Con k=10, 0 samples → devuelve overall; 10 samples → 50/50; ≥50 → bucket
    domina.
    """
    denom = bucket_samples + prior_weight
    if denom <= 0:
        return overall_mean
    return (bucket_samples * bucket_mean + prior_weight * overall_mean) / denom


def affinity_score(
    *,
    bucket_mean: float,
    bucket_samples: int,
    overall_mean: float,
    prior_weight: float = 10.0,
) -> tuple[float, float]:
    """Retorna (affinity_shrunk_pct, confidence_tanh).

    affinity = (adjusted_mean / overall_mean) - 1, +15% → jugador rinde 15% mejor
    confidence = tanh(n/10) → saturates at ~1 en 30+ samples
    """
    if overall_mean <= 0:
        return 0.0, 0.0
    adjusted = shrinkage_adjusted_mean(
        bucket_mean=bucket_mean,
        bucket_samples=bucket_samples,
        overall_mean=overall_mean,
        prior_weight=prior_weight,
    )
    affinity = adjusted / overall_mean - 1.0
    confidence = float(np.tanh(bucket_samples / 10.0))
    return float(affinity), confidence


# ═══════════════════════ Queries + feature build ═══════════════════════


@dataclass(slots=True)
class PlayerWeatherSplit:
    player_id: int
    bucket: WeatherBucket
    metric: str
    sample_size: int
    bucket_mean: float
    bucket_std: float
    overall_mean: float
    overall_std: float
    affinity: float
    confidence: float
    first_game: str | None
    last_game: str | None


async def fetch_player_weather_split(
    *,
    player_id: int,
    bucket: WeatherBucket,
    metric: str,
    min_games: int = 3,
) -> PlayerWeatherSplit | None:
    """Consulta la materialized view. Si menos de min_games, retorna None."""
    async with session_scope() as session:
        # Match flexible: bucket puede requerir partial match (temp, wind only)
        result = await session.execute(
            text(
                """
                WITH bucket_stats AS (
                  SELECT
                    COUNT(*) AS n,
                    AVG((stats->>:metric)::numeric) AS mean,
                    STDDEV((stats->>:metric)::numeric) AS std,
                    MIN(ingested_at)::text AS first_game,
                    MAX(ingested_at)::text AS last_game
                  FROM player_game_logs
                  WHERE player_id = :player_id
                    AND weather_bucket->>'temp' = :temp
                    AND weather_bucket->>'wind' = :wind
                    AND weather_bucket->>'precip' = :precip
                    AND stats->>:metric IS NOT NULL
                ),
                overall_stats AS (
                  SELECT
                    AVG((stats->>:metric)::numeric) AS mean,
                    STDDEV((stats->>:metric)::numeric) AS std
                  FROM player_game_logs
                  WHERE player_id = :player_id
                    AND stats->>:metric IS NOT NULL
                )
                SELECT
                  b.n, b.mean AS bucket_mean, b.std AS bucket_std,
                  b.first_game, b.last_game,
                  o.mean AS overall_mean, o.std AS overall_std
                FROM bucket_stats b CROSS JOIN overall_stats o
                """
            ),
            {
                "player_id": player_id,
                "metric": metric,
                "temp": bucket.temp,
                "wind": bucket.wind,
                "precip": bucket.precip,
            },
        )
        row = result.first()

    if row is None or row.n is None or int(row.n) < min_games:
        return None

    affinity, confidence = affinity_score(
        bucket_mean=float(row.bucket_mean or 0),
        bucket_samples=int(row.n),
        overall_mean=float(row.overall_mean or 0),
    )
    return PlayerWeatherSplit(
        player_id=player_id,
        bucket=bucket,
        metric=metric,
        sample_size=int(row.n),
        bucket_mean=float(row.bucket_mean or 0),
        bucket_std=float(row.bucket_std or 0),
        overall_mean=float(row.overall_mean or 0),
        overall_std=float(row.overall_std or 0),
        affinity=affinity,
        confidence=confidence,
        first_game=row.first_game,
        last_game=row.last_game,
    )


async def build_weather_features(
    *,
    player_id: int,
    sport_code: str,
    event_bucket: WeatherBucket,
) -> dict[str, float]:
    """Features específicas por deporte, con todos los buckets relevantes."""
    features: dict[str, float] = {}

    # Indoor NBA → sin features weather
    if event_bucket.is_indoor and sport_code == "nba":
        features["weather_is_indoor"] = 1.0
        features["weather_affinity"] = 0.0
        features["weather_confidence"] = 1.0
        return features

    features["weather_is_indoor"] = 1.0 if event_bucket.is_indoor else 0.0

    # Metrics clave por deporte
    metric_map = {
        "mlb": ["home_runs", "total_bases", "hits"],
        "nfl": ["passing_yards", "rushing_yards", "receiving_yards"],
        "soccer": ["goals", "shots_on_target", "yellow_cards"],
        "nba": ["points", "rebounds", "assists"],
        "boxing": ["rounds_completed"],
    }
    metrics = metric_map.get(sport_code, [])

    for metric in metrics:
        split = await fetch_player_weather_split(
            player_id=player_id,
            bucket=event_bucket,
            metric=metric,
        )
        if split is None:
            features[f"weather_{metric}_affinity"] = 0.0
            features[f"weather_{metric}_samples"] = 0.0
            features[f"weather_{metric}_confidence"] = 0.0
            continue
        features[f"weather_{metric}_affinity"] = split.affinity
        features[f"weather_{metric}_samples"] = float(split.sample_size)
        features[f"weather_{metric}_confidence"] = split.confidence

    # Flags binarios específicos para LLM/alertas
    features["weather_cold"] = 1.0 if event_bucket.temp in {"freezing", "cold"} else 0.0
    features["weather_hot"] = 1.0 if event_bucket.temp == "hot" else 0.0
    features["weather_wind_strong"] = 1.0 if event_bucket.wind in {"strong", "gale"} else 0.0
    features["weather_wind_out"] = 1.0 if event_bucket.wind_dir == "out_to_rf_cf" else 0.0
    features["weather_wind_in"] = 1.0 if event_bucket.wind_dir == "in_to_pitcher" else 0.0
    features["weather_precip"] = 1.0 if event_bucket.precip != "none" else 0.0
    features["weather_altitude_high"] = 1.0 if event_bucket.altitude in {"high", "extreme"} else 0.0

    return features


# ═══════════════════════ Boosts específicos MLB HR ═════════════════════


def mlb_hr_weather_multiplier(bucket: WeatherBucket) -> float:
    """Ajuste multiplicativo sobre P(HR) base por condiciones del park.

    Basado en Statcast 2015-2024 park factors rolling.
    """
    mult = 1.0
    # Temperatura: cada 10°F bajo 72°F reduce HR ~4%
    if bucket.temp == "freezing":
        mult *= 0.72
    elif bucket.temp == "cold":
        mult *= 0.85
    elif bucket.temp == "cool":
        mult *= 0.93
    elif bucket.temp == "warm":
        mult *= 1.05
    elif bucket.temp == "hot":
        mult *= 1.10

    # Viento salida boost
    if bucket.wind_dir == "out_to_rf_cf":
        if bucket.wind == "moderate":
            mult *= 1.18
        elif bucket.wind == "strong":
            mult *= 1.35
        elif bucket.wind == "gale":
            mult *= 1.55
    elif bucket.wind_dir == "in_to_pitcher":
        if bucket.wind == "moderate":
            mult *= 0.85
        elif bucket.wind == "strong":
            mult *= 0.72
        elif bucket.wind == "gale":
            mult *= 0.60

    # Humedad alta → aire denso → menos carry
    if bucket.humidity == "saturated":
        mult *= 0.95
    elif bucket.humidity == "humid":
        mult *= 0.97

    # Altitud extrema (Coors) ~+15% HR
    if bucket.altitude == "high":
        mult *= 1.08
    elif bucket.altitude == "extreme":
        mult *= 1.15

    return mult


def nfl_passing_weather_multiplier(bucket: WeatherBucket) -> float:
    """Ajuste sobre P(QB over passing yards) y CPOE."""
    if bucket.is_indoor:
        return 1.0
    mult = 1.0
    if bucket.wind == "moderate":
        mult *= 0.95
    elif bucket.wind == "strong":
        mult *= 0.85
    elif bucket.wind == "gale":
        mult *= 0.70

    if bucket.precip == "moderate":
        mult *= 0.95
    elif bucket.precip == "heavy":
        mult *= 0.88

    if bucket.temp == "freezing":
        mult *= 0.92
    elif bucket.temp == "cold":
        mult *= 0.97

    return mult


def soccer_goals_weather_multiplier(bucket: WeatherBucket) -> float:
    """Ajuste sobre P(gol) / totales."""
    mult = 1.0
    if bucket.temp == "freezing":
        mult *= 0.90
    elif bucket.temp == "hot":
        mult *= 0.93

    if bucket.precip == "heavy":
        mult *= 0.92

    # Altitud: favorece home last-30-min goals, afecta visitante
    # Este factor se modela como interaction term, no simple multiplier
    return mult
