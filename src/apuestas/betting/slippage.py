"""Fase 4.14 — Market impact / slippage tracker.

Apostar $500 en Caliente en un mercado pequeño mueve la línea. Pros trackean
`odds_obtained` vs `odds_displayed`. Si slippage >5% → book poco líquido →
reduce stake o skip.

API:
    record_bet_slippage(bet_id, odds_displayed, odds_obtained)
    get_avg_slippage_per_book(bookmaker, lookback_days=30)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def compute_slippage_bps(odds_displayed: float, odds_obtained: float) -> int:
    """Retorna slippage en basis points.

    bps positivos = ganaste valor (odds_obtained > displayed, raro).
    bps negativos = perdiste valor (odds_obtained < displayed, normal).
    """
    if odds_displayed <= 0:
        return 0
    return round((odds_obtained - odds_displayed) / odds_displayed * 10000)


async def record_bet_slippage(
    bet_id: int,
    *,
    odds_displayed: float,
    odds_obtained: float,
) -> dict[str, int | float]:
    """Persiste odds_displayed + slippage_bps en la bet ya creada."""
    bps = compute_slippage_bps(odds_displayed, odds_obtained)
    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE bets
                SET odds_displayed = :od,
                    slippage_bps = :bps
                WHERE id = :bid
                """
            ),
            {"bid": bet_id, "od": odds_displayed, "bps": bps},
        )
    logger.info(
        "slippage.recorded",
        bet_id=bet_id,
        displayed=odds_displayed,
        obtained=odds_obtained,
        bps=bps,
    )
    return {
        "bet_id": bet_id,
        "odds_displayed": odds_displayed,
        "odds_obtained": odds_obtained,
        "slippage_bps": bps,
    }


async def get_avg_slippage_per_book(
    bookmaker: str,
    *,
    lookback_days: int = 30,
) -> dict[str, float | int]:
    """Stats de slippage del book en ventana N días."""
    since = datetime.now(tz=UTC) - timedelta(days=lookback_days)
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT COUNT(*) AS n,
                           AVG(slippage_bps) AS avg_bps,
                           PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(slippage_bps))
                             AS p95_abs_bps
                    FROM bets
                    WHERE bookmaker = :bk
                      AND placed_at > :since
                      AND slippage_bps IS NOT NULL
                      AND test_data = false
                    """
                ),
                {"bk": bookmaker, "since": since},
            )
        ).first()

    n = int(row.n or 0) if row else 0
    avg_bps = float(row.avg_bps or 0.0) if row else 0.0
    p95_abs_bps = float(row.p95_abs_bps or 0.0) if row else 0.0

    return {
        "bookmaker": bookmaker,
        "n_bets": n,
        "avg_slippage_bps": avg_bps,
        "p95_abs_slippage_bps": p95_abs_bps,
        "warning_high_slippage": p95_abs_bps > 50,  # type: ignore[dict-item]
    }
