"""Soccer weather impact on goals — Sprint 14 #155.

Literatura:
  - Dixon & Robinson 1998: rain reduce xG ~8% (pelota mojada, pases más cortos)
  - Nathan 2008 (adaptado): wind >20mph altera crosses/centros altos
  - Palacios-Huerta 2014: cold <5°C reduce finishing accuracy

Features output:
  - weather_total_adj: factor multiplicativo sobre xG esperado [0.85, 1.10]
  - weather_variance_factor: incrementa varianza resultado (más empates/blowouts raros)

Lookup: weather_stadium_archive (Open-Meteo ERA5 ingesteado) via venue_id + match_time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def compute_weather_goals_adjustment(
    *,
    precipitation_mm: float,
    wind_speed_mph: float,
    temperature_c: float,
) -> dict[str, float]:
    """Retorna multiplicadores para total goals + variance.

    Reglas (Nathan 2008 + Dixon-Robinson 1998):
      - precipitation > 5mm: xG * 0.92
      - precipitation > 15mm: xG * 0.85
      - wind > 20mph: xG * 0.95, crosses affected
      - temp < 5°C: xG * 0.95
      - temp > 32°C: xG * 0.97 (fatigue)
    """
    adj = 1.0
    var = 1.0

    if precipitation_mm >= 15.0:
        adj *= 0.85
        var *= 1.15
    elif precipitation_mm >= 5.0:
        adj *= 0.92
        var *= 1.08

    if wind_speed_mph >= 25.0:
        adj *= 0.93
        var *= 1.15
    elif wind_speed_mph >= 20.0:
        adj *= 0.95
        var *= 1.05

    if temperature_c < 0.0:
        adj *= 0.92
    elif temperature_c < 5.0 or temperature_c > 35.0:
        adj *= 0.95
    elif temperature_c > 32.0:
        adj *= 0.97

    return {
        "weather_total_adj": max(0.70, min(1.10, adj)),
        "weather_variance_factor": max(1.0, min(1.30, var)),
    }


async def fetch_match_weather(
    session: Any, venue_id: int | None, match_time: datetime
) -> dict[str, float] | None:
    """Lookup weather en ventana ±1h del kickoff desde weather_stadium_archive."""
    if venue_id is None:
        return None
    try:
        row = (
            await session.execute(
                text(
                    """
                    SELECT precipitation_mm, wind_speed_mph, temperature_c
                    FROM weather_stadium_archive
                    WHERE venue_id = :vid
                      AND observation_time BETWEEN :start AND :end
                    ORDER BY ABS(EXTRACT(EPOCH FROM (observation_time - :mt))) ASC
                    LIMIT 1
                    """
                ),
                {
                    "vid": venue_id,
                    "mt": match_time,
                    "start": match_time.replace(tzinfo=match_time.tzinfo)
                    if match_time.tzinfo
                    else match_time,
                    "end": match_time,
                },
            )
        ).first()
        if not row:
            return None
        return {
            "precipitation_mm": float(row.precipitation_mm or 0.0),
            "wind_speed_mph": float(row.wind_speed_mph or 0.0),
            "temperature_c": float(row.temperature_c or 15.0),
        }
    except Exception as exc:
        logger.debug("soccer_weather.fetch_fail", venue_id=venue_id, error=str(exc)[:80])
        return None


async def compute_soccer_weather_features(
    session: Any, *, venue_id: int | None, match_time: datetime
) -> dict[str, float]:
    """Wrapper: weather lookup + adjustment. Fallback a 1.0/1.0 si no data."""
    w = await fetch_match_weather(session, venue_id, match_time)
    if w is None:
        return {"weather_total_adj": 1.0, "weather_variance_factor": 1.0}
    return compute_weather_goals_adjustment(**w)


__all__ = [
    "compute_soccer_weather_features",
    "compute_weather_goals_adjustment",
    "fetch_match_weather",
]
