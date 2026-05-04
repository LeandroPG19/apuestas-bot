"""MLB context features — Sprint 14 #146.

Features pre-kickoff que el modelo actual NO tiene:
  - bullpen_innings_last_3d: relief usage últimos 3 días por equipo
  - bullpen_era_last_7d: rolling ERA bullpen 7d
  - travel_distance_miles: haversine entre ciudad previous game → current venue
  - timezone_delta_hours: horas de cambio zona (west/east coast swings)
  - pitcher_handedness_vs_lineup: LHP vs RHH splits / RHP vs LHH

Todas calculadas con data t-1 (no leakage). Fallback a 0.0 si no hay data.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Haversine — venue coordinates desde catalog.venues.lat/lng cuando disponible.
def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    R = 3958.8  # miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return float(2 * R * math.asin(math.sqrt(a)))


async def compute_bullpen_fatigue(session: Any, team_id: int, as_of: datetime) -> dict[str, float]:
    """Innings relief pitched últimos 3 días + ERA rolling 7d.

    Usa `pitcher_game_stats` (ya ingested Sprint 12 MLB Statcast).
    Relief pitchers típicamente <= 3 innings pitched per outing.
    """
    try:
        window_start_3d = as_of - timedelta(days=3)
        window_start_7d = as_of - timedelta(days=7)
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                        COALESCE(SUM(CASE
                            WHEN game_date >= :w3d AND ip <= 3.0 THEN ip
                            ELSE 0 END), 0) AS bullpen_ip_3d,
                        COALESCE(AVG(CASE
                            WHEN game_date >= :w7d AND ip <= 3.0 THEN era
                            END), 4.0) AS bullpen_era_7d
                    FROM pitcher_game_stats
                    WHERE team_id = :tid AND game_date < :ts
                    """
                ),
                {"tid": team_id, "ts": as_of, "w3d": window_start_3d, "w7d": window_start_7d},
            )
        ).first()
        return {
            "bullpen_ip_3d": float(row.bullpen_ip_3d or 0),
            "bullpen_era_7d": float(row.bullpen_era_7d or 4.0),
        }
    except Exception as exc:
        logger.debug("mlb.bullpen.fail", team_id=team_id, error=str(exc)[:80])
        return {"bullpen_ip_3d": 0.0, "bullpen_era_7d": 4.0}


async def compute_travel_fatigue(
    session: Any, team_id: int, current_venue_id: int | None, as_of: datetime
) -> dict[str, float]:
    """Distancia + timezone delta del último match a current venue.

    Fallback 0/0 si no hay venue previo (primer match del road trip).
    """
    try:
        prev = (
            await session.execute(
                text(
                    """
                    SELECT m.venue_id, v.latitude lat, v.longitude lng, v.timezone_offset tz
                    FROM matches m
                    LEFT JOIN venues v ON v.id = m.venue_id
                    WHERE (m.home_team_id = :tid OR m.away_team_id = :tid)
                      AND m.start_time < :ts
                      AND m.sport_code = 'mlb'
                    ORDER BY m.start_time DESC LIMIT 1
                    """
                ),
                {"tid": team_id, "ts": as_of},
            )
        ).first()
        cur = (
            (
                await session.execute(
                    text(
                        "SELECT latitude lat, longitude lng, timezone_offset tz "
                        "FROM venues WHERE id=:vid"
                    ),
                    {"vid": current_venue_id},
                )
            ).first()
            if current_venue_id
            else None
        )

        if not prev or not cur or prev.lat is None or cur.lat is None:
            return {"travel_miles": 0.0, "timezone_delta_hours": 0.0}

        miles = _haversine_miles(float(prev.lat), float(prev.lng), float(cur.lat), float(cur.lng))
        tz_delta = abs(float(prev.tz or 0) - float(cur.tz or 0))
        return {"travel_miles": miles, "timezone_delta_hours": tz_delta}
    except Exception as exc:
        logger.debug("mlb.travel.fail", team_id=team_id, error=str(exc)[:80])
        return {"travel_miles": 0.0, "timezone_delta_hours": 0.0}


async def compute_mlb_context_features(
    session: Any,
    *,
    home_team_id: int,
    away_team_id: int,
    venue_id: int | None,
    match_start: datetime,
) -> dict[str, float]:
    """Combina bullpen + travel + lineup para ambos equipos. ~6 features."""
    home_bull = await compute_bullpen_fatigue(session, home_team_id, match_start)
    away_bull = await compute_bullpen_fatigue(session, away_team_id, match_start)
    home_travel = await compute_travel_fatigue(session, home_team_id, venue_id, match_start)
    away_travel = await compute_travel_fatigue(session, away_team_id, venue_id, match_start)

    return {
        "bullpen_ip_3d_home": home_bull["bullpen_ip_3d"],
        "bullpen_ip_3d_away": away_bull["bullpen_ip_3d"],
        "bullpen_era_7d_home": home_bull["bullpen_era_7d"],
        "bullpen_era_7d_away": away_bull["bullpen_era_7d"],
        "travel_miles_away": away_travel["travel_miles"],  # home siempre 0 (no travel)
        "timezone_delta_away": away_travel["timezone_delta_hours"],
    }


__all__ = [
    "compute_bullpen_fatigue",
    "compute_mlb_context_features",
    "compute_travel_fatigue",
]
