"""Calibración de probabilidades: isotonic/Platt + Venn-Abers + MAPIE conformal.

Pipeline completo:
1. Entrenar modelo base (probabilístico).
2. CalibratedClassifierCV con isotonic (n>=1000/clase) o sigmoid (Platt).
3. Envolver con MapieClassifier para intervalos conformal con cobertura
   garantizada (90% default).
4. Opcional: Venn-Abers para muestras pequeñas.

Métricas primarias: Brier score, log-loss, ECE. NUNCA usar accuracy
como métrica primaria (blueprint §9 anti-pattern #1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from apuestas.obs.logging import get_logger

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

logger = get_logger(__name__)

CalibrationMethod = Literal["isotonic", "sigmoid", "venn_abers"]


class HasPredictProba(Protocol):
    """Protocol para estimadores con predict_proba."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class CalibrationMetrics:
    log_loss: float
    brier: float
    ece: float
    n_samples: int
    n_bins: int

    def is_well_calibrated(self) -> bool:
        """ECE < 0.03 es considerado bien calibrado en la literatura."""
        return self.ece < 0.03


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 15
) -> float:
    """Expected Calibration Error con bins equi-frecuentes.

    Mide la brecha media ponderada entre confianza predicha y precisión real.
    Guo et al. 2017, "On Calibration of Modern Neural Networks".
    """
    if len(y_true) != len(y_prob):
        msg = f"Shapes don't match: {y_true.shape} vs {y_prob.shape}"
        raise ValueError(msg)
    if len(y_true) == 0:
        return 0.0

    # Confidence = max prob; para binario, max(p, 1-p)
    if y_prob.ndim == 1:
        confidence = np.maximum(y_prob, 1.0 - y_prob)
        predictions = (y_prob >= 0.5).astype(int)
    else:
        confidence = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)

    correct = (predictions == y_true).astype(float)

    # Quantile-based bins evitan bins vacíos en cola de la distribución
    bin_edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)

    # Fallback: si todas las confidences son idénticas (quantiles colapsan),
    # un único bin mide el gap global entre confianza y accuracy.
    if len(bin_edges) < 2:
        bin_acc = float(correct.mean())
        bin_conf = float(confidence.mean())
        return abs(bin_acc - bin_conf)

    ece = 0.0
    total = len(y_true)
    for i in range(len(bin_edges) - 1):
        mask = (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidence[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Devuelve log-loss, Brier y ECE. Para binario espera p[clase=1]."""
    y_prob_pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
    ll = float(log_loss(y_true, np.clip(y_prob_pos, 1e-7, 1 - 1e-7)))
    brier = float(brier_score_loss(y_true, y_prob_pos))
    ece = expected_calibration_error(y_true, y_prob_pos, n_bins=n_bins)
    return CalibrationMetrics(
        log_loss=ll, brier=brier, ece=ece, n_samples=len(y_true), n_bins=n_bins
    )


def select_calibration_method(n_per_class: int) -> CalibrationMethod:
    """Decisión práctica:
    - n >= 1000/clase → isotonic (flexible, no paramétrico).
    - 100 <= n < 1000 → sigmoid/Platt (paramétrico, robusto a muestras chicas).
    - n < 100 → venn_abers (probabilidades válidas bajo intercambiabilidad).
    """
    if n_per_class >= 1000:
        return "isotonic"
    if n_per_class >= 100:
        return "sigmoid"
    return "venn_abers"


def fit_calibrated(
    base_estimator: BaseEstimator,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    method: CalibrationMethod | None = None,
    cv: int | str = 5,
) -> BaseEstimator:
    """Envuelve un estimador entrenado con calibración isotonic/sigmoid.

    Con cv='prefit' asume que `base_estimator` YA está entrenado en un
    conjunto separado del de calibración (recomendado para walk-forward).
    """
    n_per_class = min(int((y_cal == 0).sum()), int((y_cal == 1).sum()))
    # "auto" (o None) delega al selector heurístico para evitar que un caller
    # imponga isotonic con muestras chicas (overfitting → ECE alto). Ver
    # NBA shadow v20260429_1806: 371 cal samples + isotonic forzado → ECE 0.10.
    if method in (None, "auto"):
        chosen: CalibrationMethod = select_calibration_method(n_per_class)
    else:
        chosen = method

    if chosen == "venn_abers":
        va = fit_venn_abers(base_estimator, X_cal, y_cal)
        if va is not None:
            logger.info("calibrate.venn_abers.ok", n_per_class=n_per_class)
            return _VennAbersWrapper(base_estimator=base_estimator, va=va)
        logger.warning(
            "calibrate.venn_abers.fallback_to_sigmoid",
            reason="Venn-Abers requiere librería venn-abers; usamos sigmoid como fallback",
            n_per_class=n_per_class,
        )
        chosen = "sigmoid"

    logger.info("calibrate.fit", method=chosen, n_per_class=n_per_class)
    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        method=chosen,
        cv=cv,
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


class _VennAbersWrapper:
    """Wrapper sklearn-compat para VennAbers calibrator.

    Combina un estimador base + VennAbers post-hoc. predict_proba devuelve
    la probabilidad punto (media de p_lower/p_upper del VA predictor).
    """

    _estimator_type = "classifier"

    def __init__(self, base_estimator: HasPredictProba, va: object) -> None:
        self.base_estimator = base_estimator
        self.va = va
        import numpy as _np

        self.classes_ = _np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_estimator.predict_proba(X)
        if raw.ndim == 2:
            p_raw_2d = raw  # (n, 2)
        else:
            p_raw_2d = np.column_stack([1.0 - raw, raw])
        try:
            # VennAbers.predict_proba espera (n, 2); devuelve (p_prime, p0_p1)
            p_prime, _p0p1 = self.va.predict_proba(p_raw_2d)  # type: ignore[attr-defined]
            # p_prime shape (n, 2): columnas = [p(0), p(1)] calibradas
            p_arr = np.asarray(p_prime)
            if p_arr.ndim == 2 and p_arr.shape[1] == 2:
                return p_arr
            # fallback si shape distinto
            p_point = p_arr.ravel()
            return np.column_stack([1.0 - p_point, p_point])
        except Exception as exc:
            logger.warning("venn_abers.predict_fail", error=str(exc)[:100])
            return p_raw_2d

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_venn_abers(
    base_estimator: HasPredictProba,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> object | None:
    """Venn-Abers predictor usando librería `venn-abers` si disponible."""
    try:
        from venn_abers import VennAbers  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("calibrate.venn_abers.library_missing")
        return None

    va = VennAbers()
    p_cal = base_estimator.predict_proba(X_cal)
    if p_cal.ndim == 1:
        p_cal = np.column_stack([1.0 - p_cal, p_cal])
    # VennAbers.fit espera shape (n, 2) + y_cal (n,)
    va.fit(p_cal, np.asarray(y_cal))
    return va


class ConformalClassifier:
    """Wrapper conformal prediction (MAPIE-compatible) con fallback manual.

    Proporciona (p_low, p_upper) por predicción con cobertura α.
    Si MAPIE no está disponible o falla, usa Inductive Conformal Classification
    manual sobre residuos de calibración (split conformal).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._mapie: object | None = None
        self._calibration_scores: np.ndarray | None = None

    def fit(
        self,
        calibrated_estimator: HasPredictProba,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> ConformalClassifier:
        # Siempre poblamos calibration_scores manuales: mapie 1.x cambió la
        # API de predict y ya no expone intervalos sobre p_pos directamente
        # (predict_set devuelve sets de clases). El fallback manual produce
        # bandas válidas sobre la prob clase-1.
        p_cal = calibrated_estimator.predict_proba(X_cal)
        p_true = p_cal[np.arange(len(y_cal)), y_cal.astype(int)]
        scores = 1.0 - p_true
        self._calibration_scores = np.sort(scores)

        try:
            # mapie 1.x: MapieClassifier renombrado → SplitConformalClassifier.
            # cv='prefit' → flag prefit=True; method='lac' es default LAC.
            from mapie.classification import (
                SplitConformalClassifier,  # type: ignore[import-untyped]
            )

            self._mapie = SplitConformalClassifier(
                estimator=calibrated_estimator,
                confidence_level=1.0 - self.alpha,
                prefit=True,
            )
            # API nueva: conformalize() en vez de fit() para split-conformal con prefit
            self._mapie.conformalize(X_cal, y_cal)  # type: ignore[attr-defined]
            logger.info("conformal.mapie_ok", api="split_conformal_v1", n_cal=len(scores))
            return self
        except Exception as exc:
            logger.warning("conformal.mapie_failed_fallback", error=str(exc))

        logger.info("conformal.manual_fallback", n_cal=len(scores))
        return self

    def predict_intervals(
        self,
        calibrated_estimator: HasPredictProba,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (p_point, p_low, p_upper) para clase positiva."""
        p_point = calibrated_estimator.predict_proba(X)
        if p_point.ndim == 2:
            p_pos = p_point[:, 1]
        else:
            p_pos = p_point

        # mapie 1.x: predict_set() devuelve sets de clases, no intervalos
        # sobre p_pos. Usamos siempre el fallback manual con calibration_scores.
        if self._calibration_scores is None:
            return p_pos, p_pos, p_pos

        # Fallback: quantile sobre scores de calibración
        q = float(np.quantile(self._calibration_scores, 1 - self.alpha, method="higher"))
        p_low = np.clip(p_pos - q, 0.0, 1.0)
        p_up = np.clip(p_pos + q, 0.0, 1.0)
        return p_pos, p_low, p_up

    def is_confident(
        self,
        p_low: float,
        *,
        implied_prob: float,
        margin: float = 0.0,
    ) -> bool:
        """Un pick es 'confident' si el límite inferior del intervalo está
        por encima de la probabilidad implícita de la cuota + margen.

        Filtro clave para §15.1 del plan: reduce bets en zona gris.
        """
        return p_low > (implied_prob + margin)
