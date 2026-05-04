"""Fase 4.7 + 4.17 — Model decay + regime change detection.

Los modelos ML pierden edge con tiempo (mercado se adapta). Pros retrenen
cada 2-4 semanas. Sin monitoring, modelo operativo se degrada silencioso.

Este módulo:
1. `check_model_decay()` — compara CLV+rate 14d vs 30d. Si decay >10% → trigger retrain.
2. `check_regime_change()` — Kolmogorov-Smirnov test sobre distribución
   (p_model − p_pinnacle_fair) últimos 7d vs baseline 90d. Si KS p<0.01 → pausar.

Ambos corren via cron semanal (systemd timer).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

DecayStatus = Literal["ok", "degraded", "critical"]


async def check_model_decay(
    *,
    short_window_days: int = 14,
    long_window_days: int = 30,
    decay_threshold: float = 0.10,
) -> dict[str, float | str | int]:
    """Compara CLV+rate corto vs largo. Decay >10% → trigger retrain."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*)::float FROM bets
                         WHERE test_data = false
                           AND placed_at > :short_since
                           AND clv IS NOT NULL) AS n_short,
                        (SELECT AVG(CASE WHEN clv > 0 THEN 1.0 ELSE 0.0 END)
                         FROM bets WHERE test_data = false
                           AND placed_at > :short_since
                           AND clv IS NOT NULL) AS rate_short,
                        (SELECT COUNT(*)::float FROM bets
                         WHERE test_data = false
                           AND placed_at BETWEEN :long_since AND :short_since
                           AND clv IS NOT NULL) AS n_long,
                        (SELECT AVG(CASE WHEN clv > 0 THEN 1.0 ELSE 0.0 END)
                         FROM bets WHERE test_data = false
                           AND placed_at BETWEEN :long_since AND :short_since
                           AND clv IS NOT NULL) AS rate_long
                    """
                ),
                {
                    "short_since": datetime.now(tz=UTC) - timedelta(days=short_window_days),
                    "long_since": datetime.now(tz=UTC) - timedelta(days=long_window_days),
                },
            )
        ).first()

    n_short = int(rows.n_short or 0) if rows else 0
    n_long = int(rows.n_long or 0) if rows else 0
    rate_short = float(rows.rate_short or 0) if rows else 0.0
    rate_long = float(rows.rate_long or 0) if rows else 0.0

    if n_short < 20 or n_long < 20:
        return {
            "status": "insufficient_data",
            "n_short": n_short,
            "n_long": n_long,
            "rate_short": rate_short,
            "rate_long": rate_long,
            "decay": 0.0,
        }

    decay = rate_long - rate_short  # >0 = degradación
    status: DecayStatus
    if decay > decay_threshold:
        status = "critical"
        logger.warning(
            "model_decay.critical",
            decay=decay,
            rate_short=rate_short,
            rate_long=rate_long,
        )
    elif decay > decay_threshold * 0.5:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "n_short": n_short,
        "n_long": n_long,
        "rate_short": rate_short,
        "rate_long": rate_long,
        "decay": decay,
        "recommend_retrain": status == "critical",
    }


async def check_regime_change(
    *,
    recent_days: int = 7,
    baseline_days: int = 90,
    ks_p_threshold: float = 0.01,
) -> dict[str, float | str | int]:
    """Kolmogorov-Smirnov test: distribución (p_model - p_pinnacle_fair) recent vs baseline.

    Si KS p-value < 0.01 → regime shifted → recomienda pausar hasta investigación manual.
    """
    try:
        from scipy.stats import ks_2samp  # type: ignore[import-untyped]
    except ImportError:
        return {"status": "scipy_missing"}

    async with session_scope() as session:
        # Diffs = p_model − p_pinnacle_fair (= model edge relative a Pinnacle)
        # Aprox: usamos predictions.probability + odds_history pinnacle fair
        recent_rows = (
            await session.execute(
                text(
                    """
                    SELECT p.probability AS p_model,
                           (SELECT 1.0 / oh.odds
                            FROM odds_history oh
                            WHERE oh.match_id = p.match_id
                              AND oh.market = p.market
                              AND oh.outcome = p.outcome
                              AND oh.bookmaker = 'pinnacle'
                              AND oh.is_closing = true
                            LIMIT 1) AS p_pinnacle
                    FROM predictions p
                    WHERE p.created_at > :since
                      AND p.test_data = false
                    """
                ),
                {"since": datetime.now(tz=UTC) - timedelta(days=recent_days)},
            )
        ).all()
        baseline_rows = (
            await session.execute(
                text(
                    """
                    SELECT p.probability AS p_model,
                           (SELECT 1.0 / oh.odds
                            FROM odds_history oh
                            WHERE oh.match_id = p.match_id
                              AND oh.market = p.market
                              AND oh.outcome = p.outcome
                              AND oh.bookmaker = 'pinnacle'
                              AND oh.is_closing = true
                            LIMIT 1) AS p_pinnacle
                    FROM predictions p
                    WHERE p.created_at BETWEEN :since_baseline AND :since_recent
                      AND p.test_data = false
                    """
                ),
                {
                    "since_baseline": datetime.now(tz=UTC) - timedelta(days=baseline_days),
                    "since_recent": datetime.now(tz=UTC) - timedelta(days=recent_days),
                },
            )
        ).all()

    recent_diffs = [
        float(r.p_model) - float(r.p_pinnacle)
        for r in recent_rows
        if r.p_pinnacle is not None and r.p_model is not None
    ]
    baseline_diffs = [
        float(r.p_model) - float(r.p_pinnacle)
        for r in baseline_rows
        if r.p_pinnacle is not None and r.p_model is not None
    ]

    if len(recent_diffs) < 20 or len(baseline_diffs) < 50:
        return {
            "status": "insufficient_data",
            "n_recent": len(recent_diffs),
            "n_baseline": len(baseline_diffs),
        }

    ks_stat, p_value = ks_2samp(recent_diffs, baseline_diffs)
    shifted = p_value < ks_p_threshold
    status = "regime_shift" if shifted else "stable"

    logger.info(
        "regime_monitor.result",
        status=status,
        ks_stat=float(ks_stat),
        p_value=float(p_value),
        n_recent=len(recent_diffs),
        n_baseline=len(baseline_diffs),
    )

    return {
        "status": status,
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "n_recent": len(recent_diffs),
        "n_baseline": len(baseline_diffs),
        "recommend_pause": shifted,
        "recent_mean_diff": float(np.mean(recent_diffs)),
        "baseline_mean_diff": float(np.mean(baseline_diffs)),
    }
