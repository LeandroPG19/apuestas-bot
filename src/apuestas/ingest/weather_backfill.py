"""Backfill histórico de weather para eventos outdoor (§24.4).

Cascade de fuentes:
1. Meteostat (gratis, `pip install meteostat`, NOAA stations) — US/EU/MX cobertura.
2. Open-Meteo (gratis, global, rate limit generoso).
3. Visual Crossing (paid $0.0001/record) — histórico detallado.

Ejecutar UNA sola vez al configurar el bot, ~20 min para 5 temporadas MLB.
Idempotente: detecta weather existente y skip.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import httpx
from sqlalchemy import text

from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.features.weather_perf import classify_forecast
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Source = Literal["meteostat", "open_meteo", "visual_crossing"]


@dataclass(slots=True)
class HistoricalWeather:
    temp_c: float | None
    wind_kph: float | None
    wind_direction_deg: int | None
    precip_mm: float | None
    humidity_pct: int | None
    conditions: str | None
    source: Source


# ═══════════════════════ Fuente 1: Open-Meteo (preferida, gratis) ══════


async def fetch_open_meteo_historical(
    *,
    lat: float,
    lon: float,
    ts: datetime,
    http: httpx.AsyncClient,
) -> HistoricalWeather | None:
    """Open-Meteo archive API — gratis, 10k req/día, cobertura global."""
    day = ts.strftime("%Y-%m-%d")
    hour = ts.hour
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": day,
        "end_date": day,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "timezone": "UTC",
    }
    try:
        resp = await http.get(url, params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, httpx.ReadTimeout) as exc:
        logger.debug("weather_backfill.open_meteo_fail", lat=lat, lon=lon, err=str(exc))
        return None

    hourly = data.get("hourly", {})
    temps = hourly.get("temperature_2m", [])
    if not temps or hour >= len(temps):
        return None

    # Wind speed viene en m/s por default → multiplicar por 3.6 para km/h
    wind_ms = hourly.get("wind_speed_10m", [0])[hour] or 0
    return HistoricalWeather(
        temp_c=temps[hour],
        wind_kph=float(wind_ms) * 3.6,
        wind_direction_deg=int(hourly.get("wind_direction_10m", [0])[hour] or 0),
        precip_mm=float(hourly.get("precipitation", [0])[hour] or 0),
        humidity_pct=int(hourly.get("relative_humidity_2m", [0])[hour] or 0),
        conditions=None,
        source="open_meteo",
    )


# ═══════════════════════ Fuente 2: Meteostat (librería local) ══════════


async def fetch_meteostat_historical(
    *,
    lat: float,
    lon: float,
    ts: datetime,
) -> HistoricalWeather | None:
    """Meteostat Python package — sin red directa, cache local NOAA."""
    try:
        from meteostat import Hourly, Point  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("weather_backfill.meteostat_not_installed")
        return None

    def _fetch() -> HistoricalWeather | None:
        location = Point(lat, lon)
        start = ts.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
        data = Hourly(location, start, end)
        df = data.fetch()
        if df.empty:
            return None
        row = df.iloc[0]
        return HistoricalWeather(
            temp_c=float(row.get("temp") or 0) if row.get("temp") is not None else None,
            wind_kph=float(row.get("wspd") or 0) if row.get("wspd") is not None else None,
            wind_direction_deg=int(row.get("wdir") or 0) if row.get("wdir") is not None else None,
            precip_mm=float(row.get("prcp") or 0) if row.get("prcp") is not None else None,
            humidity_pct=int(row.get("rhum") or 0) if row.get("rhum") is not None else None,
            conditions=None,
            source="meteostat",
        )

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as exc:
        logger.debug("weather_backfill.meteostat_err", err=str(exc))
        return None


# ═══════════════════════ Fuente 3: Visual Crossing (paid fallback) ══════


async def fetch_visual_crossing_historical(
    *,
    lat: float,
    lon: float,
    ts: datetime,
    http: httpx.AsyncClient,
) -> HistoricalWeather | None:
    """Visual Crossing Weather API — paid; fallback último recurso."""
    settings = get_settings()
    if not settings.apis.visual_crossing_key:
        return None
    api_key = settings.apis.visual_crossing_key.get_secret_value()

    day = ts.strftime("%Y-%m-%dT%H:%M:%S")
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices"
        f"/rest/services/timeline/{lat},{lon}/{day}"
    )
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": api_key,
        "contentType": "json",
    }
    try:
        resp = await http.get(url, params=params, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, httpx.ReadTimeout) as exc:
        logger.debug("weather_backfill.vcrossing_fail", err=str(exc))
        return None

    days = data.get("days", [])
    if not days:
        return None
    hours = days[0].get("hours", [])
    if not hours:
        return None
    h = hours[ts.hour] if ts.hour < len(hours) else hours[0]

    return HistoricalWeather(
        temp_c=h.get("temp"),
        wind_kph=h.get("windspeed"),
        wind_direction_deg=int(h.get("winddir") or 0),
        precip_mm=h.get("precip"),
        humidity_pct=int(h.get("humidity") or 0),
        conditions=h.get("conditions"),
        source="visual_crossing",
    )


# ═══════════════════════ Cascade principal ═════════════════════════════


async def fetch_historical_weather(
    *,
    lat: float,
    lon: float,
    ts: datetime,
    http: httpx.AsyncClient | None = None,
) -> HistoricalWeather | None:
    """Intenta Open-Meteo → Meteostat → Visual Crossing en ese orden."""
    should_close = False
    if http is None:
        http = httpx.AsyncClient()
        should_close = True

    try:
        snapshot = await fetch_open_meteo_historical(lat=lat, lon=lon, ts=ts, http=http)
        if snapshot is not None:
            return snapshot

        snapshot = await fetch_meteostat_historical(lat=lat, lon=lon, ts=ts)
        if snapshot is not None:
            return snapshot

        return await fetch_visual_crossing_historical(lat=lat, lon=lon, ts=ts, http=http)
    finally:
        if should_close:
            await http.aclose()


# ═══════════════════════ Persistencia + backfill orquestador ════════════


async def _persist_weather(
    match_id: int, snapshot: HistoricalWeather, forecast_ts: datetime
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO weather_forecast
                  (match_id, forecast_ts, temp_c, wind_kph, wind_direction_deg,
                   precip_mm, humidity_pct, conditions, source)
                VALUES
                  (:match_id, :forecast_ts, :temp, :wind, :wind_dir,
                   :precip, :humidity, :conditions, :source)
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "match_id": match_id,
                "forecast_ts": forecast_ts,
                "temp": snapshot.temp_c,
                "wind": snapshot.wind_kph,
                "wind_dir": snapshot.wind_direction_deg,
                "precip": snapshot.precip_mm,
                "humidity": snapshot.humidity_pct,
                "conditions": snapshot.conditions,
                "source": snapshot.source,
            },
        )


async def _compute_bucket_for_match(
    match_id: int,
    snapshot: HistoricalWeather,
    altitude_m: int | None,
    is_indoor: bool,
    venue_orientation_deg: int | None = None,
) -> None:
    """Clasifica bucket y lo escribe en player_game_logs.weather_bucket para todos
    los players del match."""
    bucket = classify_forecast(
        temp_c=snapshot.temp_c,
        wind_kph=snapshot.wind_kph,
        wind_direction_deg=snapshot.wind_direction_deg,
        venue_orientation_deg=venue_orientation_deg,
        precip_mm=snapshot.precip_mm,
        humidity_pct=snapshot.humidity_pct,
        altitude_m=altitude_m,
        is_indoor=is_indoor,
    )
    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE player_game_logs
                SET weather_bucket = :bucket
                WHERE match_id = :match_id AND weather_bucket IS NULL
                """
            ),
            {"match_id": match_id, "bucket": bucket.to_dict()},
        )


async def backfill_weather_for_matches(
    match_ids: list[int],
    *,
    rate_limit_per_second: float = 5.0,
) -> dict[str, int]:
    """Backfill weather + bucket para N partidos outdoor. Idempotente.

    Rate-limit cliente (Open-Meteo tiene soft limit ~10/s)."""
    stats = {"checked": 0, "skipped_indoor": 0, "fetched": 0, "no_data": 0, "errors": 0}

    async with httpx.AsyncClient() as http:
        for mid in match_ids:
            stats["checked"] += 1
            async with session_scope() as session:
                meta = await session.execute(
                    text(
                        """
                        SELECT m.start_time, v.lat, v.lon, v.altitude_m, v.roof,
                               (SELECT COUNT(*) FROM weather_forecast w
                                WHERE w.match_id = m.id) AS has_weather
                        FROM matches m
                        LEFT JOIN venues v ON v.id = m.venue_id
                        WHERE m.id = :id
                        """
                    ),
                    {"id": mid},
                )
                row = meta.first()

            if row is None:
                stats["errors"] += 1
                continue
            if row.roof in {"dome", "indoor"}:
                stats["skipped_indoor"] += 1
                continue
            if row.has_weather and row.has_weather > 0:
                # Ya tiene weather registrado
                continue
            if row.lat is None or row.lon is None:
                stats["no_data"] += 1
                continue

            snapshot = await fetch_historical_weather(
                lat=float(row.lat),
                lon=float(row.lon),
                ts=row.start_time,
                http=http,
            )
            if snapshot is None:
                stats["no_data"] += 1
                continue

            await _persist_weather(mid, snapshot, row.start_time)
            await _compute_bucket_for_match(
                match_id=mid,
                snapshot=snapshot,
                altitude_m=row.altitude_m,
                is_indoor=False,
            )
            stats["fetched"] += 1

            # Rate limiting cliente-side
            await asyncio.sleep(1.0 / rate_limit_per_second)

    logger.info("weather_backfill.done", **stats)
    return stats


async def refresh_player_weather_splits() -> None:
    """REFRESH CONCURRENTLY sobre la materialized view §24.3.

    Correr semanalmente o tras backfill grande.
    """
    async with session_scope() as session:
        await session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY player_weather_splits"))
    logger.info("weather_backfill.view_refreshed")
