"""Fase 3.6 — Regime-dependent Kelly multiplier.

Mercados trending vs mean-reverting se comportan distinto. Kelly óptimo ajusta:
  - trending (H > 0.55): equipos calientes siguen ganando → Kelly × 1.3
  - mean-reverting (H < 0.45): regression to mean → Kelly × 0.7
  - neutral: sin ajuste × 1.0

El Hurst exponent es el indicador canónico. Se calcula sobre la secuencia
de win rates recientes del equipo (proxy del régimen del sport).

Bloom/Benham documentan Kelly multiplier ∈ [0.15, 0.35] según régimen.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Regime = Literal["trending", "reverting", "neutral"]


def hurst_exponent(series: np.ndarray) -> float:
    """Calcula Hurst exponent via R/S (rescaled range analysis).

    H > 0.5 → trending (persistente).
    H = 0.5 → random walk (neutral).
    H < 0.5 → mean-reverting (anti-persistent).
    """
    n = len(series)
    if n < 20:
        return 0.5  # insufficient data → neutral

    # Serie de deltas (demeaned)
    mean_series = series - np.mean(series)
    cumulative = np.cumsum(mean_series)
    ranges = np.max(cumulative) - np.min(cumulative)
    std = np.std(series, ddof=1)
    if std < 1e-9 or ranges < 1e-9:
        return 0.5
    rs = ranges / std
    # Hurst = log(R/S) / log(N)
    return float(np.log(rs) / np.log(n))


async def detect_regime(
    sport_code: str,
    *,
    lookback_days: int = 21,
    trending_threshold: float = 0.55,
    reverting_threshold: float = 0.45,
) -> Regime:
    """Detecta régimen del sport basado en Hurst exponent del home-win-rate rolling."""
    since = datetime.now(tz=UTC) - timedelta(days=lookback_days)
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT DATE_TRUNC('day', start_time) AS day,
                           AVG(CASE WHEN home_score > away_score THEN 1.0 ELSE 0.0 END)
                             AS home_wr
                    FROM matches
                    WHERE sport_code = :sp
                      AND status = 'finished'
                      AND start_time > :since
                      AND home_score IS NOT NULL
                    GROUP BY day
                    ORDER BY day ASC
                    """
                ),
                {"sp": sport_code, "since": since},
            )
        ).all()

    if len(rows) < 10:
        logger.debug("regime.insufficient_data", sport=sport_code, n_days=len(rows))
        return "neutral"

    series = np.array([float(r.home_wr) for r in rows], dtype=np.float64)
    h = hurst_exponent(series)

    regime: Regime
    if h > trending_threshold:
        regime = "trending"
    elif h < reverting_threshold:
        regime = "reverting"
    else:
        regime = "neutral"

    logger.info("regime.detected", sport=sport_code, hurst=h, regime=regime, n_days=len(rows))
    return regime


REGIME_KELLY_MULTIPLIER: dict[Regime, float] = {
    "trending": 1.3,
    "neutral": 1.0,
    "reverting": 0.7,
}


async def kelly_multiplier_for_sport(sport_code: str) -> float:
    """Shortcut: devuelve el multiplier Kelly para un sport dado su régimen actual."""
    regime = await detect_regime(sport_code)
    return REGIME_KELLY_MULTIPLIER[regime]
