"""Gate de promoción por KPIs primarios (Sprint 7b).

Plan §7.5 / §9.4. Antes de promover un modelo shadow → production, valida
los 4 KPIs MVP sobre sus predicciones en pick_alerts resueltas:

  1. Brier ≤ brier_objective
  2. Brier Skill Score ≥ 0.03
  3. ECE ≤ ece_objective
  4. hit_rate − implied_rate ≥ +0.02

Los objetivos por deporte viven en `config/slo_calibration.yaml`. Si un
modelo no pasa el gate, `passes_kpi_gate()` retorna `(False, reasons)` y
el caller (retrain/shadow flow) no promueve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ml.metrics import compute_metrics
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_SLO_PATH = Path(__file__).resolve().parents[3] / "config" / "slo_calibration.yaml"
_SLO_CACHE: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class KPIGateResult:
    passes: bool
    reasons: list[str]
    n_picks: int
    brier: float
    brier_skill_score: float
    ece: float
    hit_rate_minus_implied: float
    sport: str
    model_name: str


def _load_slo() -> dict[str, dict[str, Any]]:
    global _SLO_CACHE
    if _SLO_CACHE is None:
        with _SLO_PATH.open("r", encoding="utf-8") as fh:
            _SLO_CACHE = yaml.safe_load(fh) or {}
    return _SLO_CACHE


def reset_slo_cache() -> None:
    global _SLO_CACHE
    _SLO_CACHE = None


def get_slo_for_sport(sport: str) -> dict[str, Any]:
    slo = _load_slo()
    return slo.get(sport.lower()) or slo.get("defaults") or {}


async def passes_kpi_gate(
    *,
    model_name: str,
    sport: str,
    window_days: int = 30,
    min_picks: int = 50,
) -> KPIGateResult:
    """Valida los 4 KPIs MVP contra pick_alerts resueltas del modelo.

    model_name: string almacenado en `predictions.model_name`.
    window_days: ventana de evaluación (default 30d).
    min_picks: umbral mínimo de picks resueltos para emitir veredicto.

    Returns:
        KPIGateResult con `passes` y `reasons` (lista de KPIs fallidos).
    """
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                        CASE WHEN pa.outcome_result = 'won' THEN 1 ELSE 0 END AS y,
                        p.probability AS p_model,
                        pa.odds_placed
                    FROM pick_alerts pa
                    JOIN predictions p ON p.id = pa.prediction_id
                    JOIN matches m ON m.id = pa.match_id
                    WHERE m.sport_code = :sport
                      AND p.model_name = :name
                      AND pa.outcome_result IN ('won','lost')
                      AND pa.result_settled_at >= NOW() - INTERVAL ':days days'
                    """.replace(":days", str(int(window_days)))
                ),
                {"sport": sport.lower(), "name": model_name},
            )
        ).all()

    n = len(rows)
    if n < min_picks:
        return KPIGateResult(
            passes=False,
            reasons=[f"insufficient_picks:{n}<{min_picks}"],
            n_picks=n,
            brier=float("nan"),
            brier_skill_score=float("nan"),
            ece=float("nan"),
            hit_rate_minus_implied=float("nan"),
            sport=sport,
            model_name=model_name,
        )

    y = np.array([int(r.y) for r in rows])
    p = np.array([float(r.p_model) if r.p_model is not None else 0.5 for r in rows])
    odds_arr = np.array([float(r.odds_placed) for r in rows if r.odds_placed is not None])
    avg_odds = float(odds_arr[odds_arr > 1.0].mean()) if (odds_arr > 1.0).any() else None
    metrics = compute_metrics(y, p, avg_odds=avg_odds)

    slo = get_slo_for_sport(sport)
    brier_cap = float(slo.get("brier_objective", 0.24))
    ece_cap = float(slo.get("ece_objective", 0.05))

    reasons: list[str] = []
    if metrics.brier > brier_cap:
        reasons.append(f"brier:{metrics.brier:.4f}>{brier_cap:.3f}")
    if metrics.brier_skill_score < 0.03:
        reasons.append(f"bss:{metrics.brier_skill_score:+.4f}<+0.03")
    if metrics.ece > ece_cap:
        reasons.append(f"ece:{metrics.ece:.4f}>{ece_cap:.3f}")
    if metrics.hit_rate_minus_implied < 0.02:
        reasons.append(f"hr_minus_implied:{metrics.hit_rate_minus_implied:+.3f}<+0.02")

    passes = not reasons
    logger.info(
        "kpi_gate.evaluated",
        model=model_name,
        sport=sport,
        passes=passes,
        reasons=reasons,
        n_picks=n,
    )
    return KPIGateResult(
        passes=passes,
        reasons=reasons,
        n_picks=n,
        brier=metrics.brier,
        brier_skill_score=metrics.brier_skill_score,
        ece=metrics.ece,
        hit_rate_minus_implied=metrics.hit_rate_minus_implied,
        sport=sport,
        model_name=model_name,
    )


__all__ = [
    "KPIGateResult",
    "get_slo_for_sport",
    "passes_kpi_gate",
    "reset_slo_cache",
]
