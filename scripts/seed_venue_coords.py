"""Seed coordenadas estadios MLB + NFL en tabla `venues`.

Sin coords no funciona `ingest_weather_archive.py`. Coords oficiales
(Wikipedia + estadios oficiales) para todos los estadios principales.

Uso:
    uv run python scripts/seed_venue_coords.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# MLB venues: 30 estadios + algunos minor league comunes
MLB_VENUES: dict[str, tuple[float, float, int]] = {
    # AL East
    "Yankee Stadium": (40.8296, -73.9262, 17),
    "Fenway Park": (42.3467, -71.0972, 6),
    "Rogers Centre": (43.6414, -79.3894, 91),
    "Tropicana Field": (27.7683, -82.6534, 5),
    "Camden Yards": (39.2839, -76.6217, 13),
    "Oriole Park at Camden Yards": (39.2839, -76.6217, 13),
    # AL Central
    "Progressive Field": (41.4962, -81.6852, 199),
    "Comerica Park": (42.3390, -83.0485, 176),
    "Kauffman Stadium": (39.0517, -94.4803, 270),
    "Target Field": (44.9817, -93.2776, 251),
    "Guaranteed Rate Field": (41.8300, -87.6338, 183),
    # AL West
    "Minute Maid Park": (29.7573, -95.3555, 13),
    "Angel Stadium": (33.8003, -117.8827, 46),
    "Globe Life Field": (32.7473, -97.0847, 168),
    "T-Mobile Park": (47.5914, -122.3324, 4),
    "Oakland Coliseum": (37.7516, -122.2005, 4),
    # NL East
    "Citi Field": (40.7571, -73.8458, 7),
    "Citizens Bank Park": (39.9061, -75.1665, 11),
    "Nationals Park": (38.8730, -77.0074, 3),
    "Truist Park": (33.8907, -84.4677, 318),
    "LoanDepot Park": (25.7781, -80.2197, 2),
    "Marlins Park": (25.7781, -80.2197, 2),
    # NL Central
    "Wrigley Field": (41.9484, -87.6553, 182),
    "PNC Park": (40.4469, -80.0057, 219),
    "Great American Ball Park": (39.0974, -84.5081, 149),
    "Busch Stadium": (38.6226, -90.1928, 140),
    "American Family Field": (43.0280, -87.9712, 195),
    # NL West
    "Chase Field": (33.4453, -112.0667, 333),
    "Coors Field": (39.7559, -104.9942, 1580),
    "Dodger Stadium": (34.0739, -118.2400, 165),
    "Oracle Park": (37.7786, -122.3893, 0),
    "Petco Park": (32.7073, -117.1566, 17),
}

# NFL venues: 32 estadios
NFL_VENUES: dict[str, tuple[float, float, int]] = {
    # AFC East
    "MetLife Stadium": (40.8135, -74.0745, 2),
    "Highmark Stadium": (42.7738, -78.7870, 185),
    "Gillette Stadium": (42.0909, -71.2643, 90),
    "Hard Rock Stadium": (25.9580, -80.2389, 2),
    # AFC North
    "M&T Bank Stadium": (39.2780, -76.6227, 11),
    "Paycor Stadium": (39.0955, -84.5160, 149),
    "Cleveland Browns Stadium": (41.5061, -81.6995, 178),
    "Huntington Bank Field": (41.5061, -81.6995, 178),
    "Acrisure Stadium": (40.4468, -80.0158, 218),
    # AFC South
    "NRG Stadium": (29.6847, -95.4107, 13),
    "Lucas Oil Stadium": (39.7601, -86.1639, 218),
    "EverBank Stadium": (30.3239, -81.6373, 4),
    "Nissan Stadium": (36.1664, -86.7713, 122),
    # AFC West
    "GEHA Field at Arrowhead Stadium": (39.0489, -94.4839, 246),
    "Arrowhead Stadium": (39.0489, -94.4839, 246),
    "Empower Field at Mile High": (39.7439, -105.0201, 1609),
    "Allegiant Stadium": (36.0909, -115.1836, 638),
    "SoFi Stadium": (33.9535, -118.3391, 20),
    # NFC East
    "AT&T Stadium": (32.7473, -97.0945, 161),
    "Lincoln Financial Field": (39.9008, -75.1675, 7),
    "FedExField": (38.9076, -76.8645, 41),
    "Commanders Field": (38.9076, -76.8645, 41),
    # NFC North
    "Soldier Field": (41.8623, -87.6167, 181),
    "Ford Field": (42.3400, -83.0456, 184),
    "Lambeau Field": (44.5013, -88.0622, 200),
    "U.S. Bank Stadium": (44.9736, -93.2575, 249),
    # NFC South
    "Mercedes-Benz Stadium": (33.7553, -84.4006, 317),
    "Bank of America Stadium": (35.2258, -80.8528, 229),
    "Caesars Superdome": (29.9511, -90.0812, 1),
    "Raymond James Stadium": (27.9758, -82.5033, 15),
    # NFC West
    "State Farm Stadium": (33.5276, -112.2626, 337),
    "Lumen Field": (47.5952, -122.3316, 12),
    "Levi's Stadium": (37.4033, -121.9694, 4),
}

# NBA venues principales
NBA_VENUES: dict[str, tuple[float, float, int]] = {
    "Madison Square Garden": (40.7505, -73.9934, 10),
    "TD Garden": (42.3663, -71.0622, 10),
    "Crypto.com Arena": (34.0430, -118.2673, 85),
    "Staples Center": (34.0430, -118.2673, 85),
    "Chase Center": (37.7680, -122.3878, 10),
    "United Center": (41.8807, -87.6742, 181),
    "American Airlines Center": (32.7905, -96.8103, 131),
    "Ball Arena": (39.7487, -105.0078, 1609),
    "Target Center": (44.9795, -93.2760, 251),
    "FedExForum": (35.1383, -90.0505, 79),
    "State Farm Arena": (33.7573, -84.3963, 320),
    "Footprint Center": (33.4457, -112.0712, 331),
    "Moda Center": (45.5316, -122.6668, 19),
    "Paycom Center": (35.4634, -97.5151, 367),
    "Gainbridge Fieldhouse": (39.7640, -86.1555, 218),
    "Kaseya Center": (25.7814, -80.1870, 2),
    "Amalie Arena": (27.9427, -82.4517, 8),
    "Wells Fargo Center": (39.9012, -75.1720, 7),
    "Barclays Center": (40.6826, -73.9754, 21),
    "Capital One Arena": (38.8982, -77.0209, 5),
    "Fiserv Forum": (43.0451, -87.9169, 189),
    "Frost Bank Center": (29.4267, -98.4375, 210),
    "Rocket Mortgage FieldHouse": (41.4965, -81.6882, 193),
    "Little Caesars Arena": (42.3411, -83.0553, 180),
    "Golden 1 Center": (38.5802, -121.4998, 10),
    "Smoothie King Center": (29.9490, -90.0820, 2),
    "Intuit Dome": (33.9430, -118.3416, 16),
    "Delta Center": (40.7683, -111.9011, 1288),
    "Spectrum Center": (35.2251, -80.8392, 230),
    "Toyota Center": (29.7508, -95.3621, 13),
}

ALL_VENUES = {**MLB_VENUES, **NFL_VENUES, **NBA_VENUES}


async def seed() -> int:
    from sqlalchemy import text

    from apuestas.db import session_scope

    updated = 0
    inserted = 0
    async with session_scope() as s:
        for name, (lat, lon, alt) in ALL_VENUES.items():
            try:
                r = await s.execute(
                    text("SELECT id FROM venues WHERE LOWER(name) = LOWER(:n) LIMIT 1"),
                    {"n": name},
                )
                row = r.first()
                if row is None:
                    await s.execute(
                        text(
                            "INSERT INTO venues (name, lat, lon, altitude_m) "
                            "VALUES (:n, :lat, :lon, :alt)"
                        ),
                        {"n": name, "lat": lat, "lon": lon, "alt": alt},
                    )
                    inserted += 1
                else:
                    await s.execute(
                        text(
                            "UPDATE venues SET lat = :lat, lon = :lon, altitude_m = :alt "
                            "WHERE id = :vid"
                        ),
                        {"vid": row[0], "lat": lat, "lon": lon, "alt": alt},
                    )
                    updated += 1
            except Exception as exc:
                print(f"Skip {name}: {exc}")
        await s.commit()
    print(f"✓ Venues coords: {updated} updated, {inserted} inserted")
    return inserted + updated


def main() -> int:
    return (asyncio.run(seed()) > 0 and 0) or 1


if __name__ == "__main__":
    raise SystemExit(main())
