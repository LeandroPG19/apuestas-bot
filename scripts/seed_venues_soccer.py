"""Seed venues + mapping team→venue para soccer (Liga MX, EPL, LaLiga, Serie A, Bundesliga, Ligue 1, MLS, Brasil, Argentina, copas).

Cobertura: ~80 estadios + ~120 fuzzy aliases de teams. Resuelve el gap del 38%
de soccer matches sin venue documentado en CLAUDE.md.

Uso:
    uv run python scripts/seed_venues_soccer.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── Venues soccer (lat, lon, altitude_m) — coords Wikipedia ──
SOCCER_VENUES: dict[str, tuple[float, float, int]] = {
    # Liga MX (18 clubes activos 2025/26)
    "Estadio Azteca": (19.3029, -99.1505, 2240),
    "Estadio Akron": (20.6818, -103.4625, 1561),
    "Estadio BBVA": (25.6691, -100.2447, 488),
    "Estadio Universitario de Nuevo León": (25.7235, -100.3128, 538),
    "Estadio Caliente": (32.5050, -116.9722, 33),
    "Estadio Cuauhtémoc": (19.0399, -98.2123, 2160),
    "Estadio Hidalgo": (20.1186, -98.7625, 2432),
    "Estadio TSM Corona": (25.5644, -103.4609, 1140),
    "Estadio Olímpico Universitario": (19.3322, -99.1942, 2300),
    "Estadio Jalisco": (20.7059, -103.3289, 1561),
    "Estadio Nemesio Diez": (19.2925, -99.6586, 2680),
    "Estadio Cuauhtémoc Puebla": (19.0399, -98.2123, 2160),
    "Estadio Corregidora": (20.6063, -100.4253, 1820),
    "Estadio Victoria": (21.8867, -102.2925, 1875),
    "Estadio Banorte": (25.7235, -100.3128, 538),
    # EPL (20 clubes)
    "Old Trafford": (53.4631, -2.2913, 38),
    "Anfield": (53.4308, -2.9608, 51),
    "Stamford Bridge": (51.4817, -0.1910, 11),
    "Emirates Stadium": (51.5549, -0.1084, 25),
    "Tottenham Hotspur Stadium": (51.6043, -0.0664, 30),
    "Etihad Stadium": (53.4831, -2.2004, 36),
    "London Stadium": (51.5386, -0.0166, 10),
    "Goodison Park": (53.4388, -2.9663, 29),
    "Hill Dickinson Stadium": (53.3680, -2.9527, 5),
    "Selhurst Park": (51.3983, -0.0856, 47),
    "Craven Cottage": (51.4750, -0.2217, 9),
    "Villa Park": (52.5092, -1.8848, 96),
    "St James' Park": (54.9756, -1.6217, 36),
    "St. James' Park": (54.9756, -1.6217, 36),
    "Molineux Stadium": (52.5904, -2.1303, 70),
    "American Express Stadium": (50.8617, -0.0833, 40),
    "City Ground": (52.9400, -1.1328, 28),
    "Vitality Stadium": (50.7353, -1.8385, 12),
    "Gtech Community Stadium": (51.4906, -0.2887, 10),
    "Kenilworth Road": (51.8842, -0.4317, 60),
    "Bramall Lane": (53.3702, -1.4708, 70),
    # LaLiga (20 clubes)
    "Santiago Bernabéu": (40.4531, -3.6883, 660),
    "Spotify Camp Nou": (41.3809, 2.1228, 14),
    "Camp Nou": (41.3809, 2.1228, 14),
    "Estadi Olímpic Lluís Companys": (41.3650, 2.1556, 130),
    "Civitas Metropolitano": (40.4362, -3.5994, 660),
    "Ramón Sánchez-Pizjuán": (37.3839, -5.9706, 11),
    "Mestalla": (39.4748, -0.3582, 17),
    "San Mamés": (43.2642, -2.9495, 17),
    "Reale Arena": (43.3014, -1.9736, 14),
    "Coliseum Alfonso Pérez": (40.3198, -3.7251, 605),
    "Estadio de la Cerámica": (39.9442, -0.1031, 56),
    "Benito Villamarín": (37.3564, -5.9819, 10),
    "Estadio de Vallecas": (40.3919, -3.6586, 668),
    "Estadio José Zorrilla": (41.6444, -4.7611, 690),
    "Estadio de Mendizorrotza": (42.8369, -2.6878, 519),
    "Estadio Municipal de Butarque": (40.3478, -3.7639, 670),
    "Estadi Mallorca Son Moix": (39.5897, 2.6303, 39),
    "Estadi Montilivi": (41.9617, 2.8281, 71),
    "RCDE Stadium": (41.3478, 2.0775, 75),
    "Stage Front Stadium": (41.3478, 2.0775, 75),
    "Estadio Municipal de Anoeta": (43.3014, -1.9736, 14),
    "Power Horse Stadium": (36.8403, -2.4356, 13),
    # Serie A (top stadiums)
    "Stadio San Siro": (45.4781, 9.1240, 122),
    "Allianz Stadium": (45.1097, 7.6411, 305),
    "Stadio Olimpico": (41.9341, 12.4547, 26),
    "Stadio Diego Armando Maradona": (40.8281, 14.1928, 17),
    "Stadio Renato Dall'Ara": (44.4920, 11.3097, 80),
    "Stadio Artemio Franchi": (43.7807, 11.2823, 60),
    "Stadio Marc'Antonio Bentegodi": (45.4356, 10.9686, 64),
    "Stadio Luigi Ferraris": (44.4164, 8.9522, 9),
    "Mapei Stadium": (44.7140, 10.6125, 48),
    # Bundesliga
    "Allianz Arena": (48.2188, 11.6248, 506),
    "Signal Iduna Park": (51.4926, 7.4519, 53),
    "Olympiastadion": (52.5147, 13.2395, 39),
    "MHPArena": (48.7926, 9.2317, 220),
    "BayArena": (51.0382, 7.0024, 75),
    "Veltins-Arena": (51.5546, 7.0676, 60),
    "Volkswagen Arena": (52.4321, 10.8033, 56),
    "Red Bull Arena": (51.3458, 12.3486, 117),
    "Deutsche Bank Park": (50.0686, 8.6453, 113),
    # Ligue 1
    "Parc des Princes": (48.8414, 2.2530, 38),
    "Orange Vélodrome": (43.2698, 5.3958, 5),
    "Groupama Stadium": (45.7651, 4.9817, 230),
    "Allianz Riviera": (43.7053, 7.1928, 30),
    "Stade Pierre-Mauroy": (50.6119, 3.1306, 19),
    "Stade Bollaert-Delelis": (50.4325, 2.8156, 50),
    "Roazhon Park": (48.1075, -1.7128, 28),
    "Stade Geoffroy-Guichard": (45.4108, 4.3903, 510),
    # MLS (top venues)
    "Mercedes-Benz Stadium MLS": (33.7553, -84.4006, 317),
    "Audi Field": (38.8696, -77.0123, 6),
    "Yankee Stadium MLS": (40.8296, -73.9262, 17),
    "Banc of California Stadium": (34.0125, -118.2853, 86),
    "BMO Stadium": (34.0125, -118.2853, 86),
    "Subaru Park": (39.8330, -75.3789, 5),
    "Inter Miami CF Stadium": (26.1944, -80.1611, 3),
    "Chase Stadium": (26.1944, -80.1611, 3),
    "GEODIS Park": (36.1311, -86.7656, 130),
    "TQL Stadium": (39.1108, -84.5224, 156),
    "Allianz Field": (44.9522, -93.1656, 254),
    "Toyota Stadium": (33.1556, -96.8347, 197),
    "Lower.com Field": (39.9686, -83.0181, 222),
    # Brasileirão (top)
    "Arena Corinthians": (-23.5453, -46.4742, 760),
    "Maracanã": (-22.9122, -43.2302, 9),
    "Allianz Parque": (-23.5275, -46.6781, 779),
    "Mineirão": (-19.8658, -43.9711, 859),
    "Arena MRV": (-19.8983, -43.9214, 850),
    "Beira-Rio": (-30.0656, -51.2369, 22),
    "Arena do Grêmio": (-29.9706, -51.1956, 7),
    "Vila Belmiro": (-23.9514, -46.3380, 4),
    "Morumbi": (-23.6000, -46.7203, 750),
    # Primera Argentina (top)
    "El Monumental": (-34.5454, -58.4498, 24),
    "La Bombonera": (-34.6356, -58.3651, 14),
    "Estadio Pedro Bidegain": (-34.6716, -58.4339, 22),
    "Cilindro de Avellaneda": (-34.6678, -58.3717, 19),
    "Estadio Libertadores de América": (-34.6648, -58.3719, 20),
    # UCL/Internacional commonly used
    "Wembley Stadium": (51.5560, -0.2796, 49),
}


# ── Mapping team_name (lower normalized) → venue_name canonical ──
# Solo aliases necesarios para fuzzy match. Nombres tal cual aparecen en `teams.name`.
TEAM_TO_VENUE: dict[str, str] = {
    # Liga MX
    "club américa": "Estadio Azteca",
    "america": "Estadio Azteca",
    "cruz azul": "Estadio Olímpico Universitario",
    "guadalajara": "Estadio Akron",
    "chivas": "Estadio Akron",
    "monterrey": "Estadio BBVA",
    "tigres": "Estadio Universitario de Nuevo León",
    "tijuana": "Estadio Caliente",
    "xolos": "Estadio Caliente",
    "puebla": "Estadio Cuauhtémoc",
    "pachuca": "Estadio Hidalgo",
    "santos laguna": "Estadio TSM Corona",
    "atlas": "Estadio Jalisco",
    "toluca": "Estadio Nemesio Diez",
    "querétaro": "Estadio Corregidora",
    "queretaro": "Estadio Corregidora",
    "necaxa": "Estadio Victoria",
    "rayados": "Estadio BBVA",
    "pumas": "Estadio Olímpico Universitario",
    "pumas unam": "Estadio Olímpico Universitario",
    "leon": "Estadio Nou Camp",
    "león": "Estadio Nou Camp",
    "mazatlan": "Estadio Mazatlán",
    "mazatlán": "Estadio Mazatlán",
    "juarez": "Estadio Olímpico Benito Juárez",
    "juárez": "Estadio Olímpico Benito Juárez",
    "fc juarez": "Estadio Olímpico Benito Juárez",
    "fc juárez": "Estadio Olímpico Benito Juárez",
    # EPL
    "manchester united": "Old Trafford",
    "liverpool": "Anfield",
    "chelsea": "Stamford Bridge",
    "arsenal": "Emirates Stadium",
    "tottenham": "Tottenham Hotspur Stadium",
    "tottenham hotspur": "Tottenham Hotspur Stadium",
    "manchester city": "Etihad Stadium",
    "west ham": "London Stadium",
    "west ham united": "London Stadium",
    "everton": "Hill Dickinson Stadium",
    "crystal palace": "Selhurst Park",
    "fulham": "Craven Cottage",
    "aston villa": "Villa Park",
    "newcastle": "St James' Park",
    "newcastle united": "St James' Park",
    "wolves": "Molineux Stadium",
    "wolverhampton": "Molineux Stadium",
    "brighton": "American Express Stadium",
    "brighton & hove albion": "American Express Stadium",
    "nottingham forest": "City Ground",
    "bournemouth": "Vitality Stadium",
    "brentford": "Gtech Community Stadium",
    "luton town": "Kenilworth Road",
    "sheffield united": "Bramall Lane",
    # LaLiga
    "real madrid": "Santiago Bernabéu",
    "barcelona": "Spotify Camp Nou",
    "fc barcelona": "Spotify Camp Nou",
    "atlético madrid": "Civitas Metropolitano",
    "atletico madrid": "Civitas Metropolitano",
    "atlético de madrid": "Civitas Metropolitano",
    "sevilla": "Ramón Sánchez-Pizjuán",
    "valencia": "Mestalla",
    "athletic club": "San Mamés",
    "athletic bilbao": "San Mamés",
    "real sociedad": "Reale Arena",
    "getafe": "Coliseum Alfonso Pérez",
    "villarreal": "Estadio de la Cerámica",
    "real betis": "Benito Villamarín",
    "betis": "Benito Villamarín",
    "rayo vallecano": "Estadio de Vallecas",
    "real valladolid": "Estadio José Zorrilla",
    "alavés": "Estadio de Mendizorrotza",
    "alaves": "Estadio de Mendizorrotza",
    "leganés": "Estadio Municipal de Butarque",
    "leganes": "Estadio Municipal de Butarque",
    "mallorca": "Estadi Mallorca Son Moix",
    "girona": "Estadi Montilivi",
    "espanyol": "RCDE Stadium",
    "almería": "Power Horse Stadium",
    "almeria": "Power Horse Stadium",
    # Serie A (top)
    "milan": "Stadio San Siro",
    "ac milan": "Stadio San Siro",
    "inter": "Stadio San Siro",
    "internazionale": "Stadio San Siro",
    "juventus": "Allianz Stadium",
    "roma": "Stadio Olimpico",
    "lazio": "Stadio Olimpico",
    "napoli": "Stadio Diego Armando Maradona",
    "bologna": "Stadio Renato Dall'Ara",
    "fiorentina": "Stadio Artemio Franchi",
    "verona": "Stadio Marc'Antonio Bentegodi",
    "hellas verona": "Stadio Marc'Antonio Bentegodi",
    "genoa": "Stadio Luigi Ferraris",
    "sampdoria": "Stadio Luigi Ferraris",
    "sassuolo": "Mapei Stadium",
    # Bundesliga
    "bayern munich": "Allianz Arena",
    "bayern münchen": "Allianz Arena",
    "borussia dortmund": "Signal Iduna Park",
    "dortmund": "Signal Iduna Park",
    "hertha bsc": "Olympiastadion",
    "vfb stuttgart": "MHPArena",
    "stuttgart": "MHPArena",
    "bayer leverkusen": "BayArena",
    "leverkusen": "BayArena",
    "schalke 04": "Veltins-Arena",
    "schalke": "Veltins-Arena",
    "wolfsburg": "Volkswagen Arena",
    "vfl wolfsburg": "Volkswagen Arena",
    "rb leipzig": "Red Bull Arena",
    "leipzig": "Red Bull Arena",
    "eintracht frankfurt": "Deutsche Bank Park",
    "frankfurt": "Deutsche Bank Park",
    # Ligue 1
    "psg": "Parc des Princes",
    "paris saint-germain": "Parc des Princes",
    "paris saint germain": "Parc des Princes",
    "marseille": "Orange Vélodrome",
    "olympique marseille": "Orange Vélodrome",
    "lyon": "Groupama Stadium",
    "olympique lyonnais": "Groupama Stadium",
    "nice": "Allianz Riviera",
    "ogc nice": "Allianz Riviera",
    "lille": "Stade Pierre-Mauroy",
    "losc lille": "Stade Pierre-Mauroy",
    "lens": "Stade Bollaert-Delelis",
    "rc lens": "Stade Bollaert-Delelis",
    "rennes": "Roazhon Park",
    "saint-étienne": "Stade Geoffroy-Guichard",
    "saint etienne": "Stade Geoffroy-Guichard",
    # Brasileirão (top)
    "corinthians": "Arena Corinthians",
    "flamengo": "Maracanã",
    "fluminense": "Maracanã",
    "palmeiras": "Allianz Parque",
    "atlético mineiro": "Arena MRV",
    "atletico mineiro": "Arena MRV",
    "cruzeiro": "Mineirão",
    "internacional": "Beira-Rio",
    "grêmio": "Arena do Grêmio",
    "gremio": "Arena do Grêmio",
    "santos": "Vila Belmiro",
    "são paulo": "Morumbi",
    "sao paulo": "Morumbi",
    # Primera Argentina (top)
    "river plate": "El Monumental",
    "boca juniors": "La Bombonera",
    "san lorenzo": "Estadio Pedro Bidegain",
    "racing club": "Cilindro de Avellaneda",
    "independiente": "Estadio Libertadores de América",
}


async def seed() -> dict[str, int]:
    from sqlalchemy import text

    from apuestas.db import session_scope

    inserted = 0
    updated = 0
    teams_mapped = 0
    teams_already = 0
    async with session_scope() as s:
        # FASE 1: upsert venues
        for name, (lat, lon, alt) in SOCCER_VENUES.items():
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
                        "WHERE id = :vid AND (lat IS NULL OR lon IS NULL)"
                    ),
                    {"vid": row[0], "lat": lat, "lon": lon, "alt": alt},
                )
                updated += 1
        # FASE 2: mapping team→venue. UPDATE teams.venue_id donde venue_id IS NULL.
        for team_alias, venue_name in TEAM_TO_VENUE.items():
            r = await s.execute(
                text("SELECT id FROM venues WHERE LOWER(name) = LOWER(:n) LIMIT 1"),
                {"n": venue_name},
            )
            v = r.first()
            if v is None:
                continue
            venue_id = int(v[0])
            # Match team por nombre normalizado lower o ILIKE prefix
            await s.execute(
                text(
                    """
                    UPDATE teams
                    SET venue_id = :vid
                    WHERE venue_id IS NULL
                      AND (LOWER(name) = :alias OR LOWER(name) LIKE :alias_like)
                    """
                ),
                {"vid": venue_id, "alias": team_alias, "alias_like": f"{team_alias}%"},
            )
            # Cuenta cuántos teams quedaron mapeados (post-update)
            r2 = await s.execute(
                text(
                    """
                    SELECT COUNT(*) FROM teams
                    WHERE venue_id = :vid
                      AND (LOWER(name) = :alias OR LOWER(name) LIKE :alias_like)
                    """
                ),
                {"vid": venue_id, "alias": team_alias, "alias_like": f"{team_alias}%"},
            )
            n = int(r2.scalar() or 0)
            if n > 0:
                teams_mapped += n
        # FASE 3: stats finales
        r = await s.execute(
            text(
                """
                SELECT
                  COUNT(*) FILTER (WHERE sport_code = 'soccer' OR sport_code IS NULL) AS total_soccer_teams,
                  COUNT(*) FILTER (WHERE (sport_code = 'soccer' OR sport_code IS NULL) AND venue_id IS NOT NULL) AS with_venue
                FROM teams
                """
            )
        )
        stats = r.first()
        await s.commit()
    print(
        f"✓ Soccer venues: {inserted} new, {updated} updated. "
        f"Teams mapped this run: {teams_mapped}. "
        f"Total soccer teams with venue: {stats.with_venue}/{stats.total_soccer_teams}"
    )
    return {
        "venues_inserted": inserted,
        "venues_updated": updated,
        "teams_mapped": teams_mapped,
        "soccer_teams_total": int(stats.total_soccer_teams or 0),
        "soccer_teams_with_venue": int(stats.with_venue or 0),
    }


if __name__ == "__main__":
    asyncio.run(seed())
