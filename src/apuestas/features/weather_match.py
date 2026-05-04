"""Weather helpers match-level para el detector (Sprint 4b).

`weather_perf.build_weather_features` opera por player; aquí exponemos
helpers match-level que el flow `deep_analysis` puede anexar al detail
del pick para enriquecer el mensaje Telegram:
  - `fetch_match_weather_bucket(match_id)` lee `weather_forecast` más
    reciente y devuelve un `WeatherBucket` clasificado.
  - `summarize_for_pick(bucket)` → string corto legible humano.
  - `multiplier_hint(bucket, sport)` → float multiplicativo informativo
    (NO altera p_blended del modelo; sólo se usa para display).

Decisión post-pivote: no ajustamos p_blended automáticamente porque no
hay backtesting que respalde la magnitud del ajuste. El usuario ve el
summary + hint y decide manualmente si amplifica/reduce su conviction.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from apuestas.features.weather_perf import (
    WeatherBucket,
    classify_forecast,
    mlb_hr_weather_multiplier,
    nfl_passing_weather_multiplier,
    soccer_goals_weather_multiplier,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def fetch_match_weather_bucket(session: Any, match_id: int) -> WeatherBucket | None:
    """Lee el forecast más reciente para el match y lo clasifica.

    Retorna None si no hay forecast — el caller debe no-opear.
    """
    row = (
        await session.execute(
            text(
                """
                SELECT temp_c, wind_kph, wind_direction_deg,
                       precip_mm, humidity_pct
                FROM weather_forecast
                WHERE match_id = :mid
                ORDER BY captured_at DESC
                LIMIT 1
                """
            ),
            {"mid": match_id},
        )
    ).first()
    if row is None:
        return None
    try:
        return classify_forecast(
            temp_c=float(row.temp_c) if row.temp_c is not None else None,
            wind_kph=float(row.wind_kph) if row.wind_kph is not None else None,
            wind_direction_deg=row.wind_direction_deg,
            precip_mm=float(row.precip_mm) if row.precip_mm is not None else None,
            humidity_pct=row.humidity_pct,
            altitude_m=None,
            is_indoor=False,
        )
    except Exception as exc:
        logger.debug("weather_match.classify_fail", match_id=match_id, error=str(exc)[:80])
        return None


def summarize_for_pick(bucket: WeatherBucket) -> str:
    """String corto human-readable para Telegram/TUI.

    Ej: "☀️ 24°C · viento débil · sin lluvia".
    """
    if bucket.is_indoor:
        return "🏟 Indoor (sin efecto clima)"
    parts: list[str] = []
    # temp (aproximado)
    if bucket.temp == "freezing":
        parts.append("🥶 helado")
    elif bucket.temp == "cold":
        parts.append("❄️ frío")
    elif bucket.temp == "hot":
        parts.append("🔥 caluroso")
    elif bucket.temp == "mild":
        parts.append("🌤 templado")
    # viento
    if bucket.wind == "gale":
        parts.append("💨 ventarrón")
    elif bucket.wind == "strong":
        parts.append("🌬 viento fuerte")
    elif bucket.wind == "moderate":
        parts.append("viento moderado")
    # lluvia
    if bucket.precip in ("heavy", "extreme"):
        parts.append("🌧 lluvia intensa")
    elif bucket.precip == "moderate":
        parts.append("🌦 lluvia")
    return " · ".join(parts) or "🌤 condiciones neutrales"


def multiplier_hint(bucket: WeatherBucket, sport: str) -> float | None:
    """Multiplicador informativo por deporte (1.0 = neutro).

    Devuelve None si el deporte no es outdoor relevante. No modifica
    p_blended — sólo se usa para mostrar `"weather_hint": 1.08` etc.
    """
    if bucket.is_indoor:
        return None
    sport_lower = sport.lower()
    if sport_lower == "mlb":
        return float(mlb_hr_weather_multiplier(bucket))
    if sport_lower == "nfl":
        return float(nfl_passing_weather_multiplier(bucket))
    if sport_lower == "soccer":
        return float(soccer_goals_weather_multiplier(bucket))
    return None


__all__ = [
    "fetch_match_weather_bucket",
    "multiplier_hint",
    "summarize_for_pick",
]
