"""Fase 1.4 — Market efficiency ranking + Kelly scaling.

Los pros no juegan todo igual. NBA Pinnacle (mercado líquido) tiene ~2% edge
máximo; Challenger tennis o Liga Expansión MX tienen ~5-8% edge. Este módulo
calcula un `efficiency_score ∈ [0, 1]` por liga donde:

  - 0 = muy eficiente (Pinnacle NBA) → Kelly muy reducido
  - 1 = muy ineficiente (ligas pequeñas) → Kelly completo

Factores del score (pesos):
  - avg_hold_last_30d       (peso 0.40) — libros con más hold = menos eficientes
  - sharp_book_coverage     (peso 0.30) — menos sharp books = menos eficientes
  - volume_proxy            (peso 0.15) — menos volumen (# matchups/semana) = menos eficiente
  - line_movement_volatility (peso 0.15) — más volatilidad = menos eficiente

Resultado cacheado en Valkey 24h (recomputa diaria).

Uso en detector:
    from apuestas.betting.market_efficiency import scale_kelly_by_efficiency
    kelly_adjusted = kelly_base * await scale_kelly_by_efficiency(league_id)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Sharp books canónicos (mercado los sigue)
SHARP_BOOKS = frozenset({"pinnacle", "circa", "betfair", "matchbook", "pinnacle_close"})

# Weights suman 1.0
WEIGHT_HOLD = 0.40
WEIGHT_SHARP_COVERAGE = 0.30
WEIGHT_VOLUME = 0.15
WEIGHT_VOLATILITY = 0.15

_CACHE: dict[int, tuple[float, datetime]] = {}
_CACHE_TTL_SECONDS = 86400  # 24h


async def compute_league_efficiency(league_id: int, *, use_cache: bool = True) -> float:
    """Calcula efficiency score para una liga [0, 1].

    Retorna 0.5 default si no hay suficiente data histórica (30 matchups mínimo).
    """
    if use_cache:
        cached = _CACHE.get(league_id)
        if cached:
            score, ts = cached
            if (datetime.now(tz=UTC) - ts).total_seconds() < _CACHE_TTL_SECONDS:
                return score

    since = datetime.now(tz=UTC) - timedelta(days=30)
    async with session_scope() as session:
        # Avg hold: overround medio usando best odds disponible per outcome
        # Simplificación: promediamos hold por match.
        hold_q = await session.execute(
            text(
                """
                WITH match_overround AS (
                    SELECT oh.match_id,
                           SUM(1.0 / oh.odds) - 1 AS hold
                    FROM odds_history oh
                    JOIN matches m ON m.id = oh.match_id
                    WHERE m.league_id = :lid
                      AND oh.ts > :since
                      AND oh.market = 'h2h'
                    GROUP BY oh.match_id
                    HAVING COUNT(*) BETWEEN 2 AND 3
                )
                SELECT AVG(hold) AS avg_hold, COUNT(*) AS n_matches
                FROM match_overround
                """
            ),
            {"lid": league_id, "since": since},
        )
        hold_row = hold_q.first()
        avg_hold = float(hold_row.avg_hold or 0.05) if hold_row else 0.05
        n_matches = int(hold_row.n_matches or 0) if hold_row else 0

        # Sharp book coverage
        sharp_q = await session.execute(
            text(
                """
                SELECT COUNT(DISTINCT oh.bookmaker) AS n_sharp
                FROM odds_history oh
                JOIN matches m ON m.id = oh.match_id
                WHERE m.league_id = :lid
                  AND oh.ts > :since
                  AND oh.bookmaker = ANY(:sharps)
                """
            ),
            {"lid": league_id, "since": since, "sharps": list(SHARP_BOOKS)},
        )
        sharp_row = sharp_q.first()
        n_sharp_books = int(sharp_row.n_sharp or 0) if sharp_row else 0

        # Volatility: std del line_movement por match (proxy: std de odds por match)
        vol_q = await session.execute(
            text(
                """
                WITH per_match AS (
                    SELECT oh.match_id, STDDEV_POP(oh.odds) AS vol
                    FROM odds_history oh
                    JOIN matches m ON m.id = oh.match_id
                    WHERE m.league_id = :lid
                      AND oh.ts > :since
                      AND oh.market = 'h2h'
                      AND oh.outcome = 'home'
                    GROUP BY oh.match_id
                    HAVING COUNT(*) >= 2
                )
                SELECT AVG(vol) AS avg_vol
                FROM per_match
                """
            ),
            {"lid": league_id, "since": since},
        )
        vol_row = vol_q.first()
        avg_vol = float(vol_row.avg_vol or 0.02) if vol_row else 0.02

    # Modelo: transformar cada factor a [0,1] donde 1 = ineficiente
    # Hold: clamp [0.01, 0.15] → normalized linearly
    hold_score = min(max((avg_hold - 0.01) / (0.15 - 0.01), 0.0), 1.0)
    # Sharp coverage: 0-4+ sharps; más sharps = más eficiente
    # Invertimos: (4 - n_sharp) / 4 (clamped)
    sharp_score = min(max((4 - n_sharp_books) / 4, 0.0), 1.0)
    # Volume: menos matchups = menos eficiente
    # Clamp [10, 100] → invertido
    vol_matches_score = min(max((100 - n_matches) / 90, 0.0), 1.0)
    # Volatility: más std de odds = menos eficiente
    volatility_score = min(max((avg_vol - 0.01) / (0.10 - 0.01), 0.0), 1.0)

    score = (
        WEIGHT_HOLD * hold_score
        + WEIGHT_SHARP_COVERAGE * sharp_score
        + WEIGHT_VOLUME * vol_matches_score
        + WEIGHT_VOLATILITY * volatility_score
    )
    score = max(0.05, min(1.0, score))  # floor 0.05 para no anular Kelly totalmente

    # Cache
    _CACHE[league_id] = (score, datetime.now(tz=UTC))

    logger.info(
        "market_efficiency.computed",
        league_id=league_id,
        score=score,
        avg_hold=avg_hold,
        n_matches=n_matches,
        n_sharp_books=n_sharp_books,
    )
    return score


async def scale_kelly_by_efficiency(league_id: int | None) -> float:
    """Retorna multiplicador Kelly [0.05, 1.0] basado en eficiencia de la liga.

    Si league_id es None → 1.0 (no scaling, sin info).
    """
    if league_id is None:
        return 1.0
    try:
        return await compute_league_efficiency(league_id)
    except Exception as exc:
        logger.debug("market_efficiency.fail", league_id=league_id, error=str(exc)[:80])
        return 1.0


def clear_cache() -> None:
    """Limpia el cache in-memory (útil para tests o tras actualizar data)."""
    _CACHE.clear()
