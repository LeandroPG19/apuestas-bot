"""Weather archive por estadio — Open-Meteo ERA5 (sin API key).

Fuente: https://open-meteo.com/en/docs/historical-weather-api
Licencia: CC-BY-NC 4.0 (free no commercial).

Cubre 1940-presente, resolución horaria. Sin auth, rate limit generoso
(~10k req/día).

Carga coordenadas estadio desde tabla `venues` y descarga weather
matching `matches.start_time`. Inserta en `weather_stadium_archive`.

Uso:
    uv run python scripts/ingest_weather_archive.py --since 2023-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

BASE = "https://archive-api.open-meteo.com/v1/archive"


async def _fetch_weather(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> dict | None:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m,precipitation",
        "timezone": "UTC",
    }
    try:
        r = await client.get(BASE, params=params, timeout=60.0)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as exc:
        logger.debug("weather.fetch_fail", lat=lat, lon=lon, error=str(exc)[:100])
        return None


async def ingest_venues_weather(since: str) -> int:
    """Para cada venue con coords, descarga weather histórico desde `since`."""
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    async with session_scope() as session:
        venues = (
            await session.execute(
                _text(
                    """
                    SELECT id, name, COALESCE(lat, 0) AS lat, COALESCE(lon, 0) AS lon
                    FROM venues
                    WHERE lat IS NOT NULL AND lon IS NOT NULL
                    """
                )
            )
        ).fetchall()

    if not venues:
        logger.warning("weather.no_venues_with_coords")
        return 0

    today = datetime.now(tz=UTC).date().isoformat()
    total = 0
    async with httpx.AsyncClient() as client:
        for v in venues:
            lat = float(v.lat)
            lon = float(v.lon)
            if lat == 0 and lon == 0:
                continue

            data = await _fetch_weather(client, lat, lon, since, today)
            if not data or "hourly" not in data:
                continue

            hourly = data["hourly"]
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            winds = hourly.get("windspeed_10m", [])
            wind_dirs = hourly.get("winddirection_10m", [])
            hums = hourly.get("relativehumidity_2m", [])
            precips = hourly.get("precipitation", [])

            async with session_scope() as session:
                await session.execute(
                    _text("DELETE FROM weather_stadium_archive WHERE venue_id = :vid"),
                    {"vid": v.id},
                )
                for i, t in enumerate(times):
                    temp_c = temps[i] if i < len(temps) else None
                    wind_ms = winds[i] if i < len(winds) else None
                    wind_dir = wind_dirs[i] if i < len(wind_dirs) else None
                    hum = hums[i] if i < len(hums) else None
                    precip = precips[i] if i < len(precips) else None
                    try:
                        await session.execute(
                            _text(
                                """
                                INSERT INTO weather_stadium_archive (
                                    venue_id, ts, temp_f, wind_mph, wind_dir_deg,
                                    humidity_pct, precip_mm, source
                                ) VALUES (
                                    :vid, :ts, :tf, :wmph, :wd, :hum, :pr, 'open-meteo'
                                )
                                ON CONFLICT (venue_id, ts) DO UPDATE SET
                                    temp_f = EXCLUDED.temp_f,
                                    wind_mph = EXCLUDED.wind_mph,
                                    humidity_pct = EXCLUDED.humidity_pct
                                """
                            ),
                            {
                                "vid": v.id,
                                "ts": datetime.fromisoformat(t).replace(tzinfo=UTC),
                                "tf": (float(temp_c) * 9.0 / 5.0 + 32.0)
                                if temp_c is not None
                                else None,
                                "wmph": float(wind_ms) * 2.237 if wind_ms is not None else None,
                                "wd": int(wind_dir) if wind_dir is not None else None,
                                "hum": int(hum) if hum is not None else None,
                                "pr": float(precip) if precip is not None else None,
                            },
                        )
                        total += 1
                    except Exception as exc:
                        logger.debug("weather.insert_fail", error=str(exc)[:80])
                await session.commit()
            logger.info("weather.venue_done", venue=v.name, rows=len(times))
    logger.info("weather.done", total=total)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=str, default="2023-01-01")
    args = parser.parse_args()
    n = asyncio.run(ingest_venues_weather(args.since))
    print(f"✓ Inserted {n} weather rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
