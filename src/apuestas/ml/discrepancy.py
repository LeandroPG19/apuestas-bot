"""Métricas de discrepancia para post-mortems (§21.1 paso 3).

Dado `prediction_snapshot` + `outcome_real`, calcula 7 métricas:
- prediction_error: |p_model - outcome_binary|
- calibration_miss: p_model vs empirical_rate_at_bucket
- ev_realized: pnl/stake
- ev_realized_vs_predicted
- llm_alignment_score: % factores LLM que se alinearon con realidad
- shap_attribution_check: % top-5 SHAP que explican efectivamente el outcome
- line_movement_assessment_correct: sharp|public asignado fue correcto?

Salida compacta + discrepancy_score global combinado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DiscrepancyMetrics:
    prediction_error: float
    calibration_miss: float | None
    ev_realized: float
    ev_realized_vs_predicted: float
    llm_alignment_score: float | None
    shap_attribution_check: float | None
    line_movement_assessment_correct: bool | None
    discrepancy_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_error": self.prediction_error,
            "calibration_miss": self.calibration_miss,
            "ev_realized": self.ev_realized,
            "ev_realized_vs_predicted": self.ev_realized_vs_predicted,
            "llm_alignment_score": self.llm_alignment_score,
            "shap_attribution_check": self.shap_attribution_check,
            "line_movement_assessment_correct": self.line_movement_assessment_correct,
            "discrepancy_score": self.discrepancy_score,
        }


def compute_discrepancy(
    *,
    p_model: float,
    outcome_binary: int,  # 1 si ganada, 0 si perdida
    ev_predicted: float,
    pnl_units: float,
    stake_units: float,
    llm_analysis: dict[str, Any] | None = None,
    shap_top5: list[dict[str, Any]] | None = None,
    actual_key_events: list[dict[str, Any]] | None = None,
    line_movement_assessment: str | None = None,
    actual_line_movement_was_sharp: bool | None = None,
    empirical_rate_at_bucket: float | None = None,
) -> DiscrepancyMetrics:
    """Calcula todas las métricas y el score global."""
    # 1. prediction_error
    prediction_error = abs(p_model - outcome_binary)

    # 2. calibration_miss (opcional si hay bucket data)
    calibration_miss: float | None = None
    if empirical_rate_at_bucket is not None:
        calibration_miss = p_model - empirical_rate_at_bucket

    # 3/4. EV realized
    ev_realized = pnl_units / stake_units if stake_units > 0 else 0.0
    ev_realized_vs_predicted = ev_realized - ev_predicted

    # 5. LLM alignment
    llm_alignment_score = None
    if llm_analysis is not None and actual_key_events is not None:
        llm_alignment_score = _llm_alignment(llm_analysis, actual_key_events)

    # 6. SHAP attribution
    shap_attribution = None
    if shap_top5 is not None and actual_key_events is not None:
        shap_attribution = _shap_attribution_check(shap_top5, actual_key_events)

    # 7. Line movement
    line_correct = None
    if line_movement_assessment is not None and actual_line_movement_was_sharp is not None:
        assessed_sharp = line_movement_assessment == "sharp"
        line_correct = assessed_sharp == actual_line_movement_was_sharp

    # Score global ponderado (0 = perfecto, 1 = terrible)
    # Factores ponderados: prediction_error (40%), calibration_miss (20%),
    # ev_mismatch (20%), llm+shap+line (20%).
    components: list[float] = [prediction_error * 0.40]
    if calibration_miss is not None:
        components.append(abs(calibration_miss) * 0.20)
    # Penalizar solo si ganamos MENOS de lo esperado (no si ganamos más).
    ev_shortfall = max(0.0, ev_predicted - ev_realized)
    components.append(min(ev_shortfall, 1.0) * 0.20)

    aux_score = 0.0
    aux_weight = 0.0
    if llm_alignment_score is not None:
        aux_score += (1 - llm_alignment_score) * 0.07
        aux_weight += 0.07
    if shap_attribution is not None:
        aux_score += (1 - shap_attribution) * 0.07
        aux_weight += 0.07
    if line_correct is not None:
        aux_score += (0.0 if line_correct else 1.0) * 0.06
        aux_weight += 0.06
    if aux_weight > 0:
        components.append(aux_score / aux_weight * 0.20)

    discrepancy = float(np.clip(sum(components), 0.0, 1.0))

    return DiscrepancyMetrics(
        prediction_error=prediction_error,
        calibration_miss=calibration_miss,
        ev_realized=ev_realized,
        ev_realized_vs_predicted=ev_realized_vs_predicted,
        llm_alignment_score=llm_alignment_score,
        shap_attribution_check=shap_attribution,
        line_movement_assessment_correct=line_correct,
        discrepancy_score=discrepancy,
    )


def _llm_alignment(llm_analysis: dict[str, Any], actual_events: list[dict[str, Any]]) -> float:
    """Fracción de factores LLM que aparecen reflejados en los eventos reales.

    Heurística: compara strings de 'key_injuries', 'lineup_changes',
    'contextual_factors' con descripciones de `actual_key_events`.
    """
    predicted_factors: list[str] = []
    for section in ("home_team_analysis", "away_team_analysis", "matchup_context"):
        section_data = llm_analysis.get(section, {}) if isinstance(llm_analysis, dict) else {}
        for key in ("key_injuries", "lineup_changes", "contextual_factors"):
            val = section_data.get(key, [])
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        predicted_factors.append(str(item).lower())
                    elif isinstance(item, str):
                        predicted_factors.append(item.lower())

    if not predicted_factors:
        return 0.5  # Sin factores explícitos, neutral

    actual_descriptions = [
        str(e.get("description", "")).lower() for e in actual_events if isinstance(e, dict)
    ]
    actual_text = " ".join(actual_descriptions)

    matches = 0
    for f in predicted_factors:
        # Matching por substring de palabras clave
        keywords = [w for w in f.split() if len(w) > 4]
        if any(kw in actual_text for kw in keywords):
            matches += 1

    return matches / len(predicted_factors)


def _shap_attribution_check(
    shap_top5: list[dict[str, Any]], actual_events: list[dict[str, Any]]
) -> float:
    """¿Las features SHAP top-5 están correlacionadas con los eventos que
    realmente decidieron el partido?

    Heurística simple: si la feature era de injuries/rest/travel y los eventos
    reales mencionan esos temas, se considera alineación.
    """
    if not shap_top5 or not actual_events:
        return 0.5

    actual_text = " ".join(
        str(e.get("description", "")).lower() for e in actual_events if isinstance(e, dict)
    )

    feature_keywords = {
        "rest": ["rest", "fatigue", "b2b", "back-to-back"],
        "ortg": ["offensive", "scoring", "goal", "points"],
        "drtg": ["defensive", "defense", "defensive"],
        "injury": ["injury", "out", "hurt", "injured"],
        "travel": ["travel", "road", "away", "long trip"],
        "altitude": ["altitude", "elevation", "denver", "mexico"],
    }

    matches = 0
    for feat in shap_top5:
        feat_name = str(feat.get("feature", "")).lower()
        matched = False
        for root, keywords in feature_keywords.items():
            if root in feat_name and any(kw in actual_text for kw in keywords):
                matched = True
                break
        if matched:
            matches += 1

    return matches / len(shap_top5)
