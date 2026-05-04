"""Fase 4.6 — Dynamic Home-Field Advantage (HFA) por venue.

HFA no es constante. Coors Field (altitude) ≠ Wrigley (wind) ≠ Fenway (Green
Monster) ≠ MetLife (weather). Starlizard estima HFA por estadio individual.

Implementación: regresión `home_score - away_score` sobre historial del venue
last 5 years. Diferencia vs HFA de la liga = venue-specific adjustment.

Uso:
    hfa = await compute_venue_hfa(venue_id=42, sport="mlb")
    # hfa = {"avg_margin": +1.2, "league_avg_margin": +0.5, "venue_specific_hfa": +0.7}
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def compute_venue_hfa(
    venue_id: int,
    sport_code: str,
    *,
    lookback_years: int = 5,
    min_games: int = 50,
) -> dict[str, float | int]:
    """Computa HFA específico del venue comparado con avg liga.

    Retorna `{avg_margin, league_avg_margin, venue_specific_hfa, n_games}`.
    `venue_specific_hfa` es el "bonus" sobre el HFA genérico.
    """
    since = datetime.now(tz=UTC) - timedelta(days=365 * lookback_years)

    async with session_scope() as session:
        venue_row = (
            await session.execute(
                text(
                    """
                    SELECT AVG(home_score - away_score) AS avg_margin,
                           COUNT(*) AS n
                    FROM matches
                    WHERE venue_id = :vid
                      AND sport_code = :sp
                      AND status = 'finished'
                      AND start_time > :since
                      AND home_score IS NOT NULL
                    """
                ),
                {"vid": venue_id, "sp": sport_code, "since": since},
            )
        ).first()

        league_row = (
            await session.execute(
                text(
                    """
                    SELECT AVG(home_score - away_score) AS league_avg_margin
                    FROM matches
                    WHERE sport_code = :sp
                      AND status = 'finished'
                      AND start_time > :since
                      AND home_score IS NOT NULL
                    """
                ),
                {"sp": sport_code, "since": since},
            )
        ).first()

    n_games = int(venue_row.n or 0) if venue_row else 0
    if n_games < min_games:
        return {
            "avg_margin": 0.0,
            "league_avg_margin": 0.0,
            "venue_specific_hfa": 0.0,
            "n_games": n_games,
            "insufficient_data": True,  # type: ignore[dict-item]
        }

    avg_margin = float(venue_row.avg_margin or 0) if venue_row else 0.0
    league_avg = float(league_row.league_avg_margin or 0) if league_row else 0.0
    venue_hfa = avg_margin - league_avg

    logger.info(
        "venue_hfa.computed",
        venue_id=venue_id,
        sport=sport_code,
        avg_margin=avg_margin,
        league_avg=league_avg,
        venue_hfa=venue_hfa,
        n_games=n_games,
    )

    return {
        "avg_margin": avg_margin,
        "league_avg_margin": league_avg,
        "venue_specific_hfa": venue_hfa,
        "n_games": n_games,
    }
