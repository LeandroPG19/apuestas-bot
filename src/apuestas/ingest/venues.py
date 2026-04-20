"""Seed de venues conocidos para los 5 deportes.

Mejor enfoque práctico: seed manual/YAML para venues principales +
enriquecimiento automático desde fixtures con geocoding ligero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_SEED_PATH = Path(__file__).resolve().parents[3] / "config" / "venues_seed.yaml"


async def seed_venues_from_yaml() -> int:
    """Cargar venues base desde YAML si existe."""
    if not _SEED_PATH.exists():
        logger.info("venues.seed.no_yaml", path=str(_SEED_PATH))
        return 0

    with _SEED_PATH.open(encoding="utf-8") as f:
        data: list[dict[str, Any]] = yaml.safe_load(f) or []

    count = 0
    async with session_scope() as session:
        for v in data:
            await session.execute(
                text(
                    """
                    INSERT INTO venues
                      (external_id, name, city, country, timezone,
                       lat, lon, altitude_m, capacity, surface, roof)
                    VALUES
                      (:external_id, :name, :city, :country, :timezone,
                       :lat, :lon, :altitude_m, :capacity, :surface, :roof)
                    ON CONFLICT (external_id) DO UPDATE
                      SET lat = EXCLUDED.lat,
                          lon = EXCLUDED.lon,
                          altitude_m = EXCLUDED.altitude_m,
                          timezone = EXCLUDED.timezone
                    """
                ),
                {
                    "external_id": v.get("external_id"),
                    "name": v["name"],
                    "city": v.get("city"),
                    "country": v.get("country"),
                    "timezone": v.get("timezone"),
                    "lat": v.get("lat"),
                    "lon": v.get("lon"),
                    "altitude_m": v.get("altitude_m"),
                    "capacity": v.get("capacity"),
                    "surface": v.get("surface"),
                    "roof": v.get("roof"),
                },
            )
            count += 1
    logger.info("venues.seeded", count=count)
    return count
