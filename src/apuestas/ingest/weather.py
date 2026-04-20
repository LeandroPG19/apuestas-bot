"""Cliente OpenWeatherMap — forecast por venue para deportes outdoor.

Free tier: 60 req/min, 1M req/mes. Suficiente para NFL/MLB outdoor games.

Se captura forecast en 2 momentos:
- T-6h: primera estimación
- T-1h: forecast más preciso
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text

from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


class OpenWeatherMapClient(BaseAPIClient):
    base_url = "https://api.openweathermap.org/data/3.0"
    source_name = "openweathermap"
    rate_limit = (50, 60.0)  # 50/min < 60/min free tier

    def __init__(self, *, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.apis.openweathermap_key.get_secret_value()
            if settings.apis.openweathermap_key
            else None
        )
        if not key:
            msg = "OPENWEATHERMAP_KEY requerida"
            raise ValueError(msg)
        super().__init__(api_key=key)
        self._key = key

    async def fetch_forecast(
        self, *, lat: float, lon: float, units: str = "metric"
    ) -> dict[str, Any]:
        """One Call API 3.0: actual + 48h hourly + 7d daily."""
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._key,
            "units": units,
            "exclude": "minutely,alerts",
        }
        return await self.get("/onecall", params=params)


def extract_forecast_at_time(raw: dict[str, Any], target_ts: datetime) -> dict[str, Any] | None:
    """De response OneCall, buscar la hora más cercana a target_ts."""
    hourly = raw.get("hourly", [])
    if not hourly:
        return None

    target_unix = int(target_ts.timestamp())
    best = min(hourly, key=lambda h: abs(h["dt"] - target_unix))

    wind = best.get("wind_speed", 0.0) * 3.6  # m/s → km/h
    return {
        "temp_c": best.get("temp"),
        "wind_kph": wind,
        "wind_direction_deg": best.get("wind_deg"),
        "precip_mm": best.get("rain", {}).get("1h", 0.0)
        if isinstance(best.get("rain"), dict)
        else 0.0,
        "humidity_pct": best.get("humidity"),
        "conditions": (best.get("weather", [{}])[0] or {}).get("description", ""),
    }


async def capture_forecast_for_match(match_id: int) -> bool:
    """Captura forecast para un match si el venue tiene lat/lon y es outdoor."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.start_time, v.lat, v.lon, v.roof
                FROM matches m
                LEFT JOIN venues v ON v.id = m.venue_id
                WHERE m.id = :match_id
                """
            ),
            {"match_id": match_id},
        )
        row = result.first()

    if row is None:
        return False

    # Dome/indoor = no weather relevante
    if row.roof == "dome" or row.roof == "indoor":
        return False
    if row.lat is None or row.lon is None:
        return False

    client = OpenWeatherMapClient()
    async with client.session():
        raw = await client.fetch_forecast(lat=float(row.lat), lon=float(row.lon))

    snapshot = extract_forecast_at_time(raw, row.start_time)
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
                   :precip_mm, :humidity_pct, :conditions, 'openweathermap')
                """
            ),
            {
                "match_id": match_id,
                "forecast_ts": row.start_time,
                **snapshot,
            },
        )

    logger.info("weather.captured", match_id=match_id, conditions=snapshot.get("conditions"))
    return True
