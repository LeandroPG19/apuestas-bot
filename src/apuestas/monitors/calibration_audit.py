"""Calibration audit semanal §21.4.

Computa calibration_gap por (sport, market, confidence_bucket, window_days).
Si gap > 0.05 con n>=30 → alerta crítica + flag review_status='flagged'.

Pipeline:
1. Por cada (sport, market) con ≥ N_MIN predictions settleadas en W days:
2. Binear P_model en buckets de ancho 0.05 (20 buckets).
3. Calcular mean_predicted vs mean_actual por bucket.
4. Brier realized + ECE realized agregados.
5. UPSERT en calibration_rolling.
6. Si gap >0.05 & n >= 30 → alarma + flag post_mortems bucket.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.mcp import memory as mcp_memory
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


WINDOWS = (7, 30, 90)
BUCKET_WIDTH = 0.05
GAP_ALERT_THRESHOLD = 0.05
MIN_SAMPLES_PER_BUCKET = 30


def _bucket_label(p: float) -> str:
    lo = int(p // BUCKET_WIDTH) * BUCKET_WIDTH
    hi = lo + BUCKET_WIDTH
    return f"p=[{lo:.2f},{hi:.2f})"


@task
async def fetch_settled_predictions(
    *,
    sport_code: str,
    market: str,
    window_days: int,
) -> list[dict[str, Any]]:
    since = datetime.now(tz=UTC) - timedelta(days=window_days)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT p.probability, b.status, b.pnl_units
                FROM predictions p
                JOIN bets b ON b.prediction_id = p.id
                JOIN matches m ON m.id = p.match_id
                WHERE m.sport_code = :sport
                  AND p.market = :market
                  AND b.status IN ('won', 'lost')
                  AND b.settled_at >= :since
                """
            ),
            {"sport": sport_code, "market": market, "since": since},
        )
        return [dict(r._mapping) for r in result.all()]


@task
async def compute_calibration_buckets(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Agrupa por bucket y computa métricas."""
    if not predictions:
        return []

    buckets: dict[str, list[dict[str, Any]]] = {}
    for p in predictions:
        prob = float(p["probability"])
        buckets.setdefault(_bucket_label(prob), []).append(p)

    results: list[dict[str, Any]] = []
    for label, rows in buckets.items():
        n = len(rows)
        if n == 0:
            continue
        probs = [float(r["probability"]) for r in rows]
        actual = [1 if r["status"] == "won" else 0 for r in rows]
        mean_pred = sum(probs) / n
        mean_actual = sum(actual) / n
        gap = mean_actual - mean_pred
        brier = sum((p - a) ** 2 for p, a in zip(probs, actual, strict=True)) / n

        results.append(
            {
                "bucket": label,
                "n": n,
                "mean_predicted": mean_pred,
                "mean_actual": mean_actual,
                "calibration_gap": gap,
                "brier_realized": brier,
            }
        )
    return results


@task
async def upsert_calibration_rolling(
    *,
    sport_code: str,
    market: str,
    window_days: int,
    buckets: list[dict[str, Any]],
) -> None:
    if not buckets:
        return
    async with session_scope() as session:
        for b in buckets:
            await session.execute(
                text(
                    """
                    INSERT INTO calibration_rolling
                      (sport_code, market, confidence_bucket, window_days,
                       n_predictions, mean_predicted, mean_actual,
                       calibration_gap, brier_realized, last_computed)
                    VALUES (:sport, :market, :bucket, :window,
                       :n, :mean_pred, :mean_actual, :gap, :brier, NOW())
                    ON CONFLICT (sport_code, market, confidence_bucket, window_days)
                    DO UPDATE SET
                      n_predictions = EXCLUDED.n_predictions,
                      mean_predicted = EXCLUDED.mean_predicted,
                      mean_actual = EXCLUDED.mean_actual,
                      calibration_gap = EXCLUDED.calibration_gap,
                      brier_realized = EXCLUDED.brier_realized,
                      last_computed = NOW()
                    """
                ),
                {
                    "sport": sport_code,
                    "market": market,
                    "bucket": b["bucket"],
                    "window": window_days,
                    "n": b["n"],
                    "mean_pred": Decimal(f"{b['mean_predicted']:.4f}"),
                    "mean_actual": Decimal(f"{b['mean_actual']:.4f}"),
                    "gap": Decimal(f"{b['calibration_gap']:.4f}"),
                    "brier": Decimal(f"{b['brier_realized']:.4f}"),
                },
            )


@task
async def trigger_alerts_on_miscalibration(
    *,
    sport_code: str,
    market: str,
    window_days: int,
    buckets: list[dict[str, Any]],
) -> list[str]:
    """Si gap > threshold y n >= 30 → alarma + flag."""
    triggered: list[str] = []
    for b in buckets:
        if abs(b["calibration_gap"]) > GAP_ALERT_THRESHOLD and b["n"] >= MIN_SAMPLES_PER_BUCKET:
            trigger = f"miscalibration_{sport_code}_{market}_{b['bucket']}_window{window_days}"
            logger.warning(
                "calibration_audit.alert",
                sport=sport_code,
                market=market,
                bucket=b["bucket"],
                gap=b["calibration_gap"],
                n=b["n"],
            )
            await mcp_memory.alarma(
                trigger=trigger,
                details={
                    "sport": sport_code,
                    "market": market,
                    "bucket": b["bucket"],
                    "gap": b["calibration_gap"],
                    "n": b["n"],
                    "window_days": window_days,
                },
            )
            triggered.append(trigger)
    return triggered


@flow(name="apuestas-calibration-audit", log_prints=True)
async def calibration_audit_flow() -> dict[str, Any]:
    """Domingo 03:00 UTC junto a retrain_weekly."""
    # Expandir si se añaden más sport/market combos
    sport_market_pairs = [
        ("nba", "h2h"),
        ("nba", "spread"),
        ("nba", "total"),
        ("mlb", "h2h"),
        ("mlb", "total"),
        ("nfl", "spread"),
        ("nfl", "total"),
        ("soccer", "h2h"),
        ("soccer", "total"),
    ]

    all_alerts: list[str] = []
    computed_pairs = 0

    for sport, market in sport_market_pairs:
        for window in WINDOWS:
            try:
                preds = await fetch_settled_predictions(
                    sport_code=sport, market=market, window_days=window
                )
                if len(preds) < 10:
                    continue
                buckets = await compute_calibration_buckets(preds)
                await upsert_calibration_rolling(
                    sport_code=sport,
                    market=market,
                    window_days=window,
                    buckets=buckets,
                )
                alerts = await trigger_alerts_on_miscalibration(
                    sport_code=sport,
                    market=market,
                    window_days=window,
                    buckets=buckets,
                )
                all_alerts.extend(alerts)
                computed_pairs += 1
            except Exception as exc:
                logger.warning(
                    "calibration_audit.pair_fail",
                    sport=sport,
                    market=market,
                    window=window,
                    error=str(exc),
                )

    return {
        "computed_pairs": computed_pairs,
        "n_alerts": len(all_alerts),
        "alerts": all_alerts,
    }


if __name__ == "__main__":
    asyncio.run(calibration_audit_flow())
