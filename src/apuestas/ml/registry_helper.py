"""Helper común para registrar modelos en `model_registry_meta` tras training.

Evita duplicación del INSERT en cada train_{sport}.py. Debe llamarse desde
el bloque `with mlflow.start_run()` para que `mlflow.active_run()` esté disponible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def register_model_in_db(
    *,
    mlflow_run_id: str,
    model_name: str,
    sport_code: str,
    stage: str,
    metrics: dict[str, Any],
) -> None:
    """Insert/update en `model_registry_meta`.

    Idempotente via ON CONFLICT (mlflow_run_id).

    Gap 5 — KPI gate universal (plan §7.5 / Sprint 7b). Si stage='production'
    se evalúa `passes_kpi_gate` sobre pick_alerts resueltas del modelo. Si no
    pasa los 4 KPIs (Brier/BSS/ECE/HR−implied), se degrada a 'shadow' y se
    loggea warning. El caller ve el stage efectivo en el log.
    """
    effective_stage = stage
    if stage == "production":
        try:
            from apuestas.ml.kpi_gate import passes_kpi_gate

            gate = await passes_kpi_gate(model_name=model_name, sport=sport_code)
            if not gate.passes:
                logger.warning(
                    "registry.kpi_gate_failed",
                    model=model_name,
                    sport=sport_code,
                    reasons=gate.reasons,
                    n_picks=gate.n_picks,
                    action="downgrade_to_shadow",
                )
                effective_stage = "shadow"
        except Exception as exc:
            # Sin datos suficientes o error transitorio → no bloquear, dejar
            # stage original; el operator decidirá si promover manualmente.
            logger.debug("registry.kpi_gate_skip", model=model_name, error=str(exc)[:80])

    try:
        async with session_scope() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO model_registry_meta
                      (mlflow_run_id, model_name, model_version, sport_code,
                       stage, promoted_at, performance_30d)
                    VALUES
                      (:run_id, :name, :version, :sport, :stage, NOW(),
                       CAST(:perf AS jsonb))
                    ON CONFLICT (mlflow_run_id) DO UPDATE SET
                      stage = EXCLUDED.stage,
                      performance_30d = EXCLUDED.performance_30d
                    """
                ),
                {
                    "run_id": mlflow_run_id,
                    "name": model_name,
                    "version": datetime.now(tz=UTC).strftime("%Y%m%d_%H%M"),
                    "sport": sport_code,
                    "stage": effective_stage,
                    "perf": json.dumps(_numeric_only(metrics)),
                },
            )
        logger.info(
            "registry.model_registered",
            model=model_name,
            sport=sport_code,
            stage=effective_stage,
            requested_stage=stage,
            run_id=mlflow_run_id[:12],
        )
    except Exception as exc:
        logger.warning(
            "registry.register_fail",
            model=model_name,
            sport=sport_code,
            error=str(exc)[:100],
        )


def _numeric_only(metrics: dict[str, Any]) -> dict[str, float]:
    """Filtra metrics a solo valores numéricos (JSON serializable)."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):  # fmt: skip
            continue
    return out
