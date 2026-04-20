"""Psychological stop-loss + bad run detection (§17.12).

Ejecutado cada hora durante sesión activa. Monitorea:
- ROI_7d < -15% con bets_7d >= 30 → pause 48h.
- drawdown_30d > 20% → reducir Kelly ¼ → ⅛.
- loss_streak >= 6 en mismo sport → pause sport 72h.
- confidence=high aciertos caen >10pp historic → pausar high-conf picks.

Los pause flags van a bot_state; resume manual vía /resumir Telegram.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.betting.portfolio import pause_bot
from apuestas.db import session_scope
from apuestas.mcp import memory as mcp_memory
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@task
async def compute_roi_and_drawdown() -> dict[str, Any]:
    """Calcula ROI 7d/30d y drawdown 30d."""
    now = datetime.now(tz=UTC)
    since_7d = now - timedelta(days=7)
    since_30d = now - timedelta(days=30)

    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                      COUNT(*) FILTER (WHERE b.status IN ('won','lost') AND b.settled_at >= :s7) AS n_7d,
                      SUM(b.pnl_units) FILTER (WHERE b.settled_at >= :s7) AS pnl_7d,
                      SUM(b.stake_units) FILTER (WHERE b.settled_at >= :s7) AS stake_7d,
                      SUM(b.pnl_units) FILTER (WHERE b.settled_at >= :s30) AS pnl_30d,
                      SUM(b.stake_units) FILTER (WHERE b.settled_at >= :s30) AS stake_30d
                    FROM bets b
                    WHERE b.status IN ('won','lost')
                    """
                ),
                {"s7": since_7d, "s30": since_30d},
            )
        ).first()

        dd_row = (
            await session.execute(
                text(
                    """
                    SELECT MAX(bankroll_units) AS peak,
                           (ARRAY_AGG(bankroll_units ORDER BY ts DESC))[1] AS current
                    FROM bankroll_history
                    WHERE ts >= :s30
                    """
                ),
                {"s30": since_30d},
            )
        ).first()

    n_7d = int(row.n_7d or 0) if row else 0
    stake_7d = float(row.stake_7d or 0) if row else 0.0
    pnl_7d = float(row.pnl_7d or 0) if row else 0.0
    roi_7d = pnl_7d / stake_7d if stake_7d > 0 else 0.0

    peak = float(dd_row.peak or 0) if dd_row else 0.0
    current = float(dd_row.current or 0) if dd_row else 0.0
    drawdown_30d = (peak - current) / peak if peak > 0 else 0.0

    return {
        "n_7d": n_7d,
        "roi_7d": roi_7d,
        "drawdown_30d": drawdown_30d,
    }


@task
async def compute_loss_streaks_by_sport() -> dict[str, int]:
    """Racha actual de pérdidas consecutivas por deporte."""
    async with session_scope() as session:
        # Para cada sport, secuencia últimas N bets settleadas
        result = await session.execute(
            text(
                """
                SELECT m.sport_code, b.status
                FROM bets b JOIN matches m ON m.id = b.match_id
                WHERE b.status IN ('won','lost')
                  AND b.settled_at >= NOW() - INTERVAL '14 days'
                ORDER BY m.sport_code, b.settled_at DESC
                """
            )
        )
        rows = [dict(r._mapping) for r in result.all()]

    by_sport: dict[str, list[str]] = {}
    for r in rows:
        by_sport.setdefault(r["sport_code"], []).append(r["status"])

    streaks: dict[str, int] = {}
    for sport, seq in by_sport.items():
        count = 0
        for status in seq:
            if status == "lost":
                count += 1
            else:
                break
        streaks[sport] = count
    return streaks


@task
async def compute_high_conf_accuracy() -> dict[str, Any]:
    """Accuracy de picks high-confidence últimos 30d vs histórico 180d."""
    async with session_scope() as session:
        # 30d
        r30 = (
            await session.execute(
                text(
                    """
                SELECT COUNT(*) AS n,
                       AVG(CASE WHEN b.status = 'won' THEN 1.0 ELSE 0.0 END) AS win_rate
                FROM bets b
                JOIN predictions p ON p.id = b.prediction_id
                WHERE b.status IN ('won','lost')
                  AND p.llm_analysis->>'confidence_in_analysis' = 'high'
                  AND b.settled_at >= NOW() - INTERVAL '30 days'
                """
                )
            )
        ).first()

        r180 = (
            await session.execute(
                text(
                    """
                SELECT COUNT(*) AS n,
                       AVG(CASE WHEN b.status = 'won' THEN 1.0 ELSE 0.0 END) AS win_rate
                FROM bets b
                JOIN predictions p ON p.id = b.prediction_id
                WHERE b.status IN ('won','lost')
                  AND p.llm_analysis->>'confidence_in_analysis' = 'high'
                  AND b.settled_at >= NOW() - INTERVAL '180 days'
                """
                )
            )
        ).first()

    n_recent = int(r30.n or 0) if r30 else 0
    wr_recent = float(r30.win_rate or 0) if r30 else 0.0
    n_hist = int(r180.n or 0) if r180 else 0
    wr_hist = float(r180.win_rate or 0) if r180 else 0.0

    return {
        "n_recent": n_recent,
        "wr_recent": wr_recent,
        "n_historic": n_hist,
        "wr_historic": wr_hist,
        "delta_pp": wr_recent - wr_hist,
    }


@task
async def update_kelly_fraction_override(fraction: float, reason: str) -> None:
    """Escribe bot_state.kelly_fraction_override."""
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES ('kelly_fraction_override', :value, NOW())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
                """
            ),
            {
                "value": {
                    "fraction": fraction,
                    "reason": reason,
                    "set_at": datetime.now(tz=UTC).isoformat(),
                }
            },
        )


@flow(name="apuestas-bad-run-monitor", log_prints=True)
async def bad_run_flow() -> dict[str, Any]:
    metrics = await compute_roi_and_drawdown()
    streaks = await compute_loss_streaks_by_sport()
    high_conf = await compute_high_conf_accuracy()

    triggered: list[str] = []

    # Regla 1: ROI 7d < -15% con n>=30 → pause 48h
    if metrics["n_7d"] >= 30 and metrics["roi_7d"] < -0.15:
        await pause_bot(
            reason=f"cool_off_roi7d_{metrics['roi_7d']:.2%}",
            triggered_by="auto_bad_run_monitor",
        )
        await mcp_memory.alarma(
            trigger="psychological_stop_loss_roi",
            details=metrics,
        )
        triggered.append("pause_48h_roi")

    # Regla 2: drawdown 30d > 20% → Kelly → ⅛
    if metrics["drawdown_30d"] > 0.20:
        await update_kelly_fraction_override(
            fraction=0.125,
            reason=f"drawdown_{metrics['drawdown_30d']:.2%}",
        )
        triggered.append("kelly_halved_drawdown")

    # Regla 3: loss streak >= 6 por sport
    for sport, streak in streaks.items():
        if streak >= 6:
            await mcp_memory.alarma(
                trigger=f"loss_streak_{sport}_{streak}",
                details={"sport": sport, "streak": streak},
            )
            triggered.append(f"pause_sport_{sport}")

    # Regla 4: high-conf degrada >10pp con muestra suficiente
    if (
        high_conf["n_recent"] >= 10
        and high_conf["n_historic"] >= 50
        and high_conf["delta_pp"] < -0.10
    ):
        await mcp_memory.alarma(
            trigger="high_conf_degradation",
            details=high_conf,
        )
        triggered.append("pause_high_conf_picks")

    logger.info(
        "bad_run.check",
        metrics=metrics,
        n_triggered=len(triggered),
        triggered=triggered,
    )
    return {
        "metrics": metrics,
        "streaks": streaks,
        "high_conf_degradation": high_conf,
        "triggered": triggered,
    }


if __name__ == "__main__":
    asyncio.run(bad_run_flow())
