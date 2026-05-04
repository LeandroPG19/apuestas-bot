"""Captura forecasts pre-match desde Open-Meteo (gratis, sin auth).

Para cada match scheduled próximas 48h con venue.lat/lon, consulta Open-Meteo
hourly forecast y guarda el snapshot más cercano al kickoff. Skip si venue
es indoor/dome (no afecta el juego) o si ya tenemos forecast reciente (<3h).

Open-Meteo:
  - Endpoint: https://api.open-meteo.com/v1/forecast
  - Params: latitude, longitude, hourly=temperature_2m,wind_speed_10m,
            wind_direction_10m,precipitation,relative_humidity_2m
  - Rate limit: 10,000 calls/día gratis (más que suficiente: 50 matches × 1 call)

Schedule: cada 3h vía Prefect/timer (los forecasts cambian intra-día).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
INDOOR_ROOFS = {"dome", "indoor", "retractable_closed"}


async def _fetch_open_meteo(lat: float, lon: float, target_ts: datetime) -> dict[str, Any] | None:
    """Fetch hourly forecast desde Open-Meteo. Devuelve snapshot al hour de target_ts."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation,relative_humidity_2m,weathercode",
        "timezone": "UTC",
        "forecast_days": 3,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("weather.openmeteo_fail", error=str(exc)[:120])
        return None

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None

    # Encuentra hora más cercana al target
    target_iso = target_ts.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
    try:
        idx = times.index(target_iso)
    except ValueError:
        # Fallback: hora más cercana en magnitud absoluta
        target_dt = target_ts.replace(minute=0, second=0, microsecond=0)
        diffs = [
            abs((datetime.fromisoformat(t).replace(tzinfo=UTC) - target_dt).total_seconds())
            for t in times
        ]
        idx = diffs.index(min(diffs))

    def _at(key: str) -> float | None:
        arr = hourly.get(key) or []
        return float(arr[idx]) if idx < len(arr) and arr[idx] is not None else None

    wc = _at("weathercode")
    conditions = _wmo_to_text(int(wc)) if wc is not None else ""

    return {
        "temp_c": _at("temperature_2m"),
        "wind_kph": _at("wind_speed_10m"),  # Open-Meteo default km/h
        "wind_direction_deg": int(_at("wind_direction_10m") or 0),
        "precip_mm": _at("precipitation"),
        "humidity_pct": int(_at("relative_humidity_2m") or 0),
        "conditions": conditions,
    }


def _wmo_to_text(code: int) -> str:
    """WMO weather code → texto legible (subset, suficiente para detector)."""
    mapping = {
        0: "clear",
        1: "mainly_clear",
        2: "partly_cloudy",
        3: "overcast",
        45: "fog",
        48: "fog",
        51: "drizzle_light",
        53: "drizzle",
        55: "drizzle_heavy",
        61: "rain_light",
        63: "rain",
        65: "rain_heavy",
        71: "snow_light",
        73: "snow",
        75: "snow_heavy",
        80: "showers_light",
        81: "showers",
        82: "showers_violent",
        95: "thunderstorm",
        96: "thunderstorm_hail",
        99: "thunderstorm_hail",
    }
    return mapping.get(code, f"wmo_{code}")


@task(retries=1, retry_delay_seconds=10)
async def capture_for_match(match_id: int, lat: float, lon: float, kickoff: datetime) -> bool:
    """Captura y persiste forecast para un match outdoor con venue."""
    snapshot = await _fetch_open_meteo(lat, lon, kickoff)
    if snapshot is None:
        return False
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO weather_forecast
                  (match_id, forecast_ts, temp_c, wind_kph, wind_direction_deg,
                   precip_mm, humidity_pct, conditions, source)
                VALUES
                  (:match_id, :forecast_ts, :temp_c, :wind_kph, :wind_direction_deg,
                   :precip_mm, :humidity_pct, :conditions, 'open-meteo')
                """
            ),
            {"match_id": match_id, "forecast_ts": kickoff, **snapshot},
        )
    return True


@flow(name="apuestas-weather-forecasts", log_prints=True)
async def capture_weather_forecasts_flow(*, hours_ahead: int = 48) -> dict[str, int]:
    """Captura forecast para todos los matches outdoor scheduled próximas N horas.

    Skip:
      - Match sin venue.lat/lon → no se puede consultar Open-Meteo.
      - Roof = dome/indoor/retractable_closed → no afecta el juego.
      - Forecast existente <3h → ya está fresh.
    """
    cutoff = datetime.now(tz=UTC) + timedelta(hours=hours_ahead)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.start_time, v.lat, v.lon
                FROM matches m
                JOIN venues v ON v.id = m.venue_id
                WHERE m.status = 'scheduled'
                  AND m.start_time > NOW()
                  AND m.start_time < :cutoff
                  AND v.lat IS NOT NULL AND v.lon IS NOT NULL
                  AND COALESCE(v.roof, '') NOT IN ('dome', 'indoor', 'retractable_closed')
                  AND NOT EXISTS (
                      SELECT 1 FROM weather_forecast wf
                      WHERE wf.match_id = m.id
                        AND wf.captured_at > NOW() - INTERVAL '3 hours'
                  )
                ORDER BY m.start_time ASC
                LIMIT 200
                """
            ),
            {"cutoff": cutoff},
        )
        matches = result.all()

    captured = 0
    failed = 0
    for m in matches:
        try:
            ok = await capture_for_match(int(m.id), float(m.lat), float(m.lon), m.start_time)
            if ok:
                captured += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            logger.warning("weather.match_fail", match_id=int(m.id), error=str(exc)[:100])

    logger.info("weather.flow_done", captured=captured, failed=failed, total=len(matches))
    return {"captured": captured, "failed": failed, "total_matches": len(matches)}


if __name__ == "__main__":
    import asyncio

    asyncio.run(capture_weather_forecasts_flow())
