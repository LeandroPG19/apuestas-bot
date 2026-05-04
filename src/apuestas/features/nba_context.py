"""NBA context features — Sprint 14 #149.

Features que el modelo NBA actual NO tiene:
  - days_rest_home/away: días desde último match
  - is_b2b_home/away: 1 si back-to-back (0 días rest)
  - travel_miles_away: distancia del último venue al current
  - referee_home_bias_90d: rolling home bias del árbitro principal (Belasen 2025)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(v is None for v in (lat1, lon1, lat2, lon2)):
        return 0.0
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return float(2 * R * math.asin(math.sqrt(a)))


async def compute_rest_days(session: Any, team_id: int, as_of: datetime) -> dict[str, float]:
    """Días desde último match NBA del equipo + flag b2b."""
    try:
        prev = (
            await session.execute(
                text(
                    """
                    SELECT start_time FROM matches
                    WHERE (home_team_id=:tid OR away_team_id=:tid)
                      AND sport_code='nba' AND start_time < :ts
                    ORDER BY start_time DESC LIMIT 1
                    """
                ),
                {"tid": team_id, "ts": as_of},
            )
        ).first()
        if not prev:
            return {"days_rest": 7.0, "is_b2b": 0.0}
        days = (as_of - prev.start_time).total_seconds() / 86400.0
        return {"days_rest": max(0.0, days), "is_b2b": 1.0 if days < 1.2 else 0.0}
    except Exception as exc:
        logger.debug("nba.rest.fail", team_id=team_id, error=str(exc)[:80])
        return {"days_rest": 2.0, "is_b2b": 0.0}


async def compute_travel_nba(
    session: Any, team_id: int, current_venue_id: int | None, as_of: datetime
) -> float:
    if current_venue_id is None:
        return 0.0
    try:
        prev = (
            await session.execute(
                text(
                    """
                    SELECT m.venue_id, v.latitude, v.longitude
                    FROM matches m LEFT JOIN venues v ON v.id=m.venue_id
                    WHERE (m.home_team_id=:tid OR m.away_team_id=:tid)
                      AND m.sport_code='nba' AND m.start_time < :ts
                    ORDER BY m.start_time DESC LIMIT 1
                    """
                ),
                {"tid": team_id, "ts": as_of},
            )
        ).first()
        cur = (
            await session.execute(
                text("SELECT latitude, longitude FROM venues WHERE id=:vid"),
                {"vid": current_venue_id},
            )
        ).first()
        if not prev or not cur or prev.latitude is None or cur.latitude is None:
            return 0.0
        return _haversine_miles(
            float(prev.latitude),
            float(prev.longitude),
            float(cur.latitude),
            float(cur.longitude),
        )
    except Exception as exc:
        logger.debug("nba.travel.fail", team_id=team_id, error=str(exc)[:80])
        return 0.0


async def compute_referee_home_bias(
    session: Any, referee_name: str | None, as_of: datetime
) -> float:
    """Rolling home win% del árbitro principal últimos 90 días.

    Belasen 2025: certain refs consistently +3pp home bias. Si hay referee_log,
    retorna (home_wins / total_games). Si no hay data, fallback 0.585 (promedio NBA).
    """
    if not referee_name:
        return 0.585
    try:
        window = as_of - timedelta(days=90)
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE home_score > away_score)::float /
                            NULLIF(COUNT(*), 0) AS home_win_rate,
                        COUNT(*) n
                    FROM nba_referee_log rl
                    JOIN matches m ON m.id = rl.match_id
                    WHERE rl.referee_name = :ref
                      AND m.start_time >= :w AND m.start_time < :ts
                    """
                ),
                {"ref": referee_name, "w": window, "ts": as_of},
            )
        ).first()
        if row is None or row.n < 10:
            return 0.585
        return float(row.home_win_rate or 0.585)
    except Exception:
        return 0.585


async def compute_nba_context_features(
    session: Any,
    *,
    home_team_id: int,
    away_team_id: int,
    venue_id: int | None,
    match_start: datetime,
    referee_name: str | None = None,
) -> dict[str, float]:
    """Combina rest + travel + referee. 7 features."""
    home_rest = await compute_rest_days(session, home_team_id, match_start)
    away_rest = await compute_rest_days(session, away_team_id, match_start)
    away_travel = await compute_travel_nba(session, away_team_id, venue_id, match_start)
    ref_bias = await compute_referee_home_bias(session, referee_name, match_start)

    return {
        "days_rest_home": home_rest["days_rest"],
        "days_rest_away": away_rest["days_rest"],
        "is_b2b_home": home_rest["is_b2b"],
        "is_b2b_away": away_rest["is_b2b"],
        "days_rest_diff": home_rest["days_rest"] - away_rest["days_rest"],
        "travel_miles_away": away_travel,
        "referee_home_bias_90d": ref_bias,
    }


__all__ = [
    "compute_nba_context_features",
    "compute_referee_home_bias",
    "compute_rest_days",
    "compute_travel_nba",
]
