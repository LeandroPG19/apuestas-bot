"""Drift detection: NannyML CBPE + PSI simple.

§15.2 / §17.8: detectar cuándo un modelo entrenado en temporada X deja
de ser válido (cambios reglas NFL, VAR, jugador estrella traspasado, etc).

Dos detectores complementarios:
- PSI (Population Stability Index) sobre cada feature individual
- CBPE (Confidence-Based Performance Estimation) sobre métricas globales
  sin requerir ground truth — ideal en betting donde resultado tarda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DriftAlert:
    feature: str
    psi: float
    severity: str  # "stable" | "drift" | "severe_drift"


@dataclass(slots=True)
class DriftReport:
    overall_drift_score: float
    feature_alerts: list[DriftAlert]
    n_features_drift: int
    cbpe_estimated_accuracy: float | None
    cbpe_confidence_interval: tuple[float, float] | None
    needs_retrain: bool
    reasons: list[str]


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI clásico: quantile bins sobre reference, luego compara distribuciones.

    Umbrales estándar:
    - PSI < 0.1: estable
    - 0.1 <= PSI < 0.25: drift moderado
    - PSI >= 0.25: drift severo, retrain recomendado
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    bin_edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / ref_counts.sum() + epsilon
    cur_pct = cur_counts / cur_counts.sum() + epsilon

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "drift"
    return "severe_drift"


def detect_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    *,
    psi_threshold: float = 0.25,
) -> list[DriftAlert]:
    """PSI por cada feature, retorna alertas con severidad."""
    if reference.shape[1] != current.shape[1]:
        msg = f"Shape mismatch: reference {reference.shape[1]} vs current {current.shape[1]}"
        raise ValueError(msg)
    if len(feature_names) != reference.shape[1]:
        msg = f"feature_names length {len(feature_names)} vs features {reference.shape[1]}"
        raise ValueError(msg)

    alerts: list[DriftAlert] = []
    for i, name in enumerate(feature_names):
        psi = population_stability_index(reference[:, i], current[:, i])
        alerts.append(DriftAlert(feature=name, psi=psi, severity=classify_psi(psi)))
    return sorted(alerts, key=lambda a: a.psi, reverse=True)


def cbpe_accuracy_estimate(
    predicted_probs: np.ndarray,
    *,
    bootstrap_samples: int = 200,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """CBPE sin ground truth: bajo hipótesis de calibración, accuracy esperada
    es E[max(p, 1-p)]. Con bootstrap se obtiene CI 90%.

    Simplificación del paper Chow et al. 2023 de NannyML.
    Si NannyML disponible, usarlo directamente; este es fallback.
    """
    try:
        import nannyml as nml  # type: ignore[import-untyped]

        # Fallback sigue siendo útil si NannyML no configurado con reference set
        _ = nml  # explicit unused
    except ImportError:
        pass

    if predicted_probs.ndim == 2:
        p_pos = predicted_probs[:, 1]
    else:
        p_pos = predicted_probs

    confidence = np.maximum(p_pos, 1.0 - p_pos)
    point_estimate = float(np.mean(confidence))

    rng = np.random.default_rng(seed)
    n = len(confidence)
    if n == 0:
        return 0.0, (0.0, 0.0)
    boot_means = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(confidence[idx])
    lower, upper = np.quantile(boot_means, [0.05, 0.95])
    return point_estimate, (float(lower), float(upper))


def full_drift_report(
    X_reference: np.ndarray,
    X_current: np.ndarray,
    feature_names: list[str],
    *,
    predicted_probs_current: np.ndarray | None = None,
    psi_critical: float = 0.25,
    cbpe_drop_threshold: float = 0.03,
    reference_accuracy_estimate: float | None = None,
) -> DriftReport:
    """Consolida feature-level PSI + CBPE en un reporte con recomendación retrain."""
    alerts = detect_feature_drift(X_reference, X_current, feature_names, psi_threshold=psi_critical)
    severe = [a for a in alerts if a.severity == "severe_drift"]
    moderate = [a for a in alerts if a.severity == "drift"]

    overall_score = float(np.mean([a.psi for a in alerts])) if alerts else 0.0

    cbpe_point: float | None = None
    cbpe_ci: tuple[float, float] | None = None
    if predicted_probs_current is not None:
        cbpe_point, cbpe_ci = cbpe_accuracy_estimate(predicted_probs_current)

    reasons: list[str] = []
    if severe:
        reasons.append(f"severe_drift_features: {', '.join(a.feature for a in severe[:5])}")
    if len(moderate) >= max(3, len(feature_names) // 4):
        reasons.append(f"many_moderate_drift_features_{len(moderate)}")
    if (
        reference_accuracy_estimate is not None
        and cbpe_point is not None
        and cbpe_point < reference_accuracy_estimate - cbpe_drop_threshold
    ):
        reasons.append(
            f"cbpe_accuracy_drop_{cbpe_point:.3f}<{reference_accuracy_estimate:.3f}-{cbpe_drop_threshold}"
        )

    needs_retrain = bool(reasons)

    logger.info(
        "drift.report",
        overall_score=overall_score,
        severe=len(severe),
        moderate=len(moderate),
        cbpe=cbpe_point,
        retrain=needs_retrain,
    )

    return DriftReport(
        overall_drift_score=overall_score,
        feature_alerts=alerts,
        n_features_drift=len(severe) + len(moderate),
        cbpe_estimated_accuracy=cbpe_point,
        cbpe_confidence_interval=cbpe_ci,
        needs_retrain=needs_retrain,
        reasons=reasons,
    )
