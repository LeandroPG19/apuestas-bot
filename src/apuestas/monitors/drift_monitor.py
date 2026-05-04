"""Drift monitor semanal §15.2 / §17.8.

Detecta cuándo un modelo en Production ya no representa la distribución
actual de features + performance CBPE sin ground truth.

Pipeline por (sport, model_version):
1. Reference: features_snapshot de últimas 200 predictions con resultado.
2. Current: features_snapshot de últimas 50 predictions recientes.
3. PSI feature-level + CBPE accuracy vs reference.
4. Si retrain_recommended → alarma + issue GitHub (vía Telegram).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.mcp import memory as mcp_memory
from apuestas.ml.drift import full_drift_report
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@task
async def fetch_predictions_for_model(
    *,
    model_name: str,
    since: datetime,
    limit: int = 500,
) -> list[dict[str, Any]]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT p.probability, p.features_snapshot, p.created_at,
                       b.status, b.pnl_units
                FROM predictions p
                LEFT JOIN bets b ON b.prediction_id = p.id
                WHERE p.model_name = :name
                  AND p.created_at >= :since
                  AND p.features_snapshot IS NOT NULL
                ORDER BY p.created_at DESC
                LIMIT :lim
                """
            ),
            {"name": model_name, "since": since, "lim": limit},
        )
        return [dict(r._mapping) for r in result.all()]


def _features_to_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray | None:
    """Convierte features_snapshot JSON → matriz numpy."""
    if not rows:
        return None
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for i, r in enumerate(rows):
        feats = r.get("features_snapshot") or {}
        if not isinstance(feats, dict):
            continue
        for j, name in enumerate(feature_names):
            val = feats.get(name)
            try:
                X[i, j] = float(val) if val is not None else 0.0
            except (TypeError, ValueError):  # fmt: skip
                X[i, j] = 0.0
    return X


@task
async def run_drift_for_model(*, model_name: str) -> dict[str, Any]:
    """Reference vs current PSI + CBPE."""
    # Current window: últimos 14 días
    current_since = datetime.now(tz=UTC) - timedelta(days=14)
    current = await fetch_predictions_for_model(
        model_name=model_name, since=current_since, limit=200
    )

    # Reference window: 14-90 días atrás
    ref_since = datetime.now(tz=UTC) - timedelta(days=90)
    ref_until = datetime.now(tz=UTC) - timedelta(days=14)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT p.probability, p.features_snapshot
                FROM predictions p
                WHERE p.model_name = :name
                  AND p.created_at BETWEEN :since AND :until
                  AND p.features_snapshot IS NOT NULL
                ORDER BY p.created_at DESC
                LIMIT 500
                """
            ),
            {"name": model_name, "since": ref_since, "until": ref_until},
        )
        reference = [dict(r._mapping) for r in result.all()]

    if len(current) < 20 or len(reference) < 50:
        return {
            "model": model_name,
            "status": "insufficient_data",
            "n_current": len(current),
            "n_reference": len(reference),
        }

    # Determinar feature_names del primer snapshot
    first_feats = current[0].get("features_snapshot", {})
    feature_names = sorted(first_feats.keys()) if isinstance(first_feats, dict) else []
    if not feature_names:
        return {"model": model_name, "status": "no_features"}

    X_ref = _features_to_matrix(reference, feature_names)
    X_cur = _features_to_matrix(current, feature_names)
    if X_ref is None or X_cur is None:
        return {"model": model_name, "status": "feature_extraction_failed"}

    probs_cur = np.asarray([float(r["probability"]) for r in current])
    probs_2d = np.vstack([1 - probs_cur, probs_cur]).T

    # Reference accuracy estimate (CBPE sobre reference)
    ref_probs = np.asarray([float(r["probability"]) for r in reference])
    ref_confidence = np.maximum(ref_probs, 1 - ref_probs).mean()

    report = full_drift_report(
        X_reference=X_ref,
        X_current=X_cur,
        feature_names=feature_names,
        predicted_probs_current=probs_2d,
        psi_critical=0.25,
        cbpe_drop_threshold=0.03,
        reference_accuracy_estimate=float(ref_confidence),
    )

    logger.info(
        "drift_monitor.model_checked",
        model=model_name,
        overall_score=report.overall_drift_score,
        severe=sum(1 for a in report.feature_alerts if a.severity == "severe_drift"),
        retrain=report.needs_retrain,
    )

    return {
        "model": model_name,
        "status": "checked",
        "overall_drift_score": report.overall_drift_score,
        "n_features_drift": report.n_features_drift,
        "cbpe": report.cbpe_estimated_accuracy,
        "needs_retrain": report.needs_retrain,
        "reasons": report.reasons,
        "top_drift_features": [
            {"feature": a.feature, "psi": a.psi, "severity": a.severity}
            for a in report.feature_alerts[:10]
        ],
    }


_SPORT_FROM_MODEL = {
    "nba_moneyline": "nba",
    "nba_ats": "nba",
    "nba_total": "nba",
    "mlb_moneyline": "mlb",
    "nfl_ats": "nfl",
    "nfl_moneyline": "nfl",
    "nhl_moneyline": "nhl",
    "tennis_moneyline": "tennis",
    "soccer_liga_mx": "soccer",
}


@task
async def alert_if_retrain_needed(
    model_name: str,
    drift_result: dict[str, Any],
    *,
    auto_retrain: bool = True,
) -> None:
    if not drift_result.get("needs_retrain"):
        return
    await mcp_memory.alarma(
        trigger=f"drift_retrain_{model_name}",
        details=drift_result,
    )
    logger.warning(
        "drift_monitor.retrain_recommended",
        model=model_name,
        reasons=drift_result.get("reasons"),
    )
    if not auto_retrain:
        return
    sport_code = _SPORT_FROM_MODEL.get(model_name)
    if not sport_code:
        logger.info("drift_monitor.auto_retrain_skip_unknown_sport", model=model_name)
        return
    try:
        from apuestas.flows.retrain_weekly import retrain_sport

        trigger_fn = getattr(retrain_sport, "fn", retrain_sport)
        retrain_res = await trigger_fn(sport_code)
        logger.info(
            "drift_monitor.auto_retrain_triggered",
            model=model_name,
            sport=sport_code,
            result=str(retrain_res)[:200],
        )
    except Exception as exc:
        logger.exception(
            "drift_monitor.auto_retrain_fail",
            model=model_name,
            error=str(exc),
        )


async def auto_degradate_drifted_model(
    model_name: str, *, brier_inflation_threshold: float = 1.10
) -> dict[str, Any]:
    """F6 hardening — degrada modelo production a shadow si Brier rolling 30d
    excede 1.10× el Brier de training (modelo drifteado a producción ruidosa).

    Defensivo: aunque el drift_monitor dispare retrain (que toma horas), F6
    saca el modelo malo de production INMEDIATAMENTE para que el detector
    no emita más picks basados en él.

    Returns: dict con `degraded`, `brier_30d`, `brier_train`, `inflation`.
    Idempotente: si el modelo ya está en shadow, no-op.
    """
    async with session_scope() as session:
        # 1. Brier de training (de model_registry_meta.performance_30d)
        meta = (
            await session.execute(
                text(
                    """
                    SELECT mlflow_run_id, sport_code, performance_30d, stage
                    FROM model_registry_meta
                    WHERE model_name = :n AND stage = 'production'
                    ORDER BY promoted_at DESC NULLS LAST LIMIT 1
                    """
                ),
                {"n": model_name},
            )
        ).first()
        if meta is None:
            return {"degraded": False, "reason": "no_production_model"}
        perf = dict(meta.performance_30d or {})
        brier_train = perf.get("holdout_brier") or perf.get("brier")
        if brier_train is None:
            return {"degraded": False, "reason": "no_brier_train"}

        # 2. Brier rolling 30d desde picks settled
        rolling = (
            await session.execute(
                text(
                    """
                    SELECT AVG((p.probability - CASE WHEN pa.outcome_result='won' THEN 1.0
                                                     WHEN pa.outcome_result='lost' THEN 0.0
                                                     ELSE NULL END) ^ 2) AS brier_30d,
                           COUNT(*) AS n
                    FROM pick_alerts pa
                    JOIN predictions p ON p.id = pa.prediction_id
                    WHERE p.model_name = :n
                      AND pa.outcome_result IN ('won','lost')
                      AND pa.placed_at > NOW() - INTERVAL '30 days'
                    """
                ),
                {"n": model_name},
            )
        ).first()
        if rolling is None or rolling.brier_30d is None or (rolling.n or 0) < 20:
            # <20 picks settled → muestra muy chica para juzgar drift
            return {
                "degraded": False,
                "reason": "insufficient_settled_picks",
                "n_settled": int(rolling.n or 0) if rolling else 0,
            }
        brier_30d = float(rolling.brier_30d)
        brier_train_f = float(brier_train)
        inflation = brier_30d / brier_train_f if brier_train_f > 0 else 0.0

        if inflation < brier_inflation_threshold:
            return {
                "degraded": False,
                "reason": "within_tolerance",
                "brier_30d": round(brier_30d, 4),
                "brier_train": round(brier_train_f, 4),
                "inflation": round(inflation, 3),
                "n_settled": int(rolling.n),
            }

        # 3. Degradar: production → shadow
        await session.execute(
            text(
                """
                UPDATE model_registry_meta
                SET stage = 'shadow'
                WHERE model_name = :n AND stage = 'production'
                """
            ),
            {"n": model_name},
        )
        await session.commit()
        logger.warning(
            "drift_monitor.auto_degradated",
            model=model_name,
            sport=meta.sport_code,
            brier_30d=round(brier_30d, 4),
            brier_train=round(brier_train_f, 4),
            inflation=round(inflation, 3),
            n_settled=int(rolling.n),
        )
        try:
            await mcp_memory.alarma(
                trigger=f"auto_degradate_{model_name}",
                details={
                    "brier_30d": round(brier_30d, 4),
                    "brier_train": round(brier_train_f, 4),
                    "inflation": round(inflation, 3),
                },
            )
        except Exception:
            pass
        return {
            "degraded": True,
            "brier_30d": round(brier_30d, 4),
            "brier_train": round(brier_train_f, 4),
            "inflation": round(inflation, 3),
            "n_settled": int(rolling.n),
        }


@flow(name="apuestas-drift-monitor", log_prints=True)
async def drift_monitor_flow() -> dict[str, Any]:
    """Semanalmente ejecuta drift sobre todos los modelos en Production."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT DISTINCT model_name
                FROM model_registry_meta
                WHERE stage = 'production'
                """
            )
        )
        models = [r[0] for r in result.all()]

    reports: dict[str, Any] = {}
    degradations: list[dict[str, Any]] = []
    for name in models:
        try:
            report = await run_drift_for_model(model_name=name)
            reports[name] = report
            await alert_if_retrain_needed(name, report)
            # F6 — auto-degradate independiente del retrain.
            # Aunque retrain tarde horas, el modelo malo NO sigue emitiendo picks.
            try:
                deg = await auto_degradate_drifted_model(name)
                if deg.get("degraded"):
                    degradations.append({"model": name, **deg})
            except Exception as exc:
                logger.exception(
                    "drift_monitor.auto_degradate_fail", model=name, error=str(exc)[:120]
                )
        except Exception as exc:
            logger.exception("drift_monitor.model_fail", model=name, error=str(exc))

    return {
        "models_checked": len(models),
        "reports": reports,
        "auto_degradations": degradations,
    }


if __name__ == "__main__":
    asyncio.run(drift_monitor_flow())
