"""Isotonic post-hoc calibration para outputs LGBM/XGB/CatBoost stacked.

Plan §7.3 / Niculescu-Mizil & Caruana 2005 (ICML). LGBM y XGBoost out-of-
the-box producen probabilidades sobreconfiadas cerca de 0 y 1. Una capa
de isotonic regression sobre un holdout reduce ECE sin tocar accuracy.

Pipeline:
  1. Entrenar modelo base (stack LGBM + XGB + CatBoost + LogReg meta).
  2. Generar p_val_raw sobre validation holdout.
  3. Fit `IsotonicRegression(out_of_bounds='clip')` con (p_val_raw, y_val).
  4. En inference: `p_calibrated = iso.predict(p_raw)`.
  5. MAPIE conformal encima de p_calibrated.

Se persiste como artifact `isotonic.pkl` junto al modelo en MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def fit_isotonic_calibrator(
    y_val: np.ndarray,
    p_val_raw: np.ndarray,
    *,
    out_path: Path | None = None,
) -> IsotonicRegression:
    """Entrena isotonic calibrator sobre holdout.

    Args:
        y_val: labels verdaderas (0/1) del holdout.
        p_val_raw: probabilidades predichas por el modelo stacked (0..1).
        out_path: si se provee, persiste el calibrator con joblib.

    Returns:
        IsotonicRegression entrenado.
    """
    y = np.asarray(y_val, dtype=np.int64).ravel()
    p = np.asarray(p_val_raw, dtype=np.float64).ravel()
    if len(y) != len(p):
        msg = f"y_val/p_val_raw shape mismatch: {y.shape} vs {p.shape}"
        raise ValueError(msg)
    if len(y) < 50:
        logger.warning("isotonic.small_holdout", n=len(y))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(iso, out_path)
        logger.info("isotonic.saved", path=str(out_path), n=len(y))

    return iso


def load_isotonic_calibrator(path: Path) -> IsotonicRegression:
    """Carga calibrator previamente persistido."""
    iso: IsotonicRegression = joblib.load(path)
    return iso


def calibrated_predict(
    estimator: Any,
    X: np.ndarray,
    calibrator: IsotonicRegression | None,
    *,
    class_index: int = 1,
) -> np.ndarray:
    """Predice p(y=class_index) aplicando calibrador isotonic si existe.

    Args:
        estimator: debe exponer `predict_proba(X) -> (n, K)`.
        X: features.
        calibrator: isotonic entrenado (None = usa p_raw sin calibrar).
        class_index: columna de `predict_proba` a calibrar (1 = positive).

    Returns:
        Array (n,) con probabilidades calibradas ∈ [0, 1].
    """
    p_raw = np.asarray(estimator.predict_proba(X))[:, class_index].astype(np.float64)
    if calibrator is None:
        return p_raw
    return np.clip(calibrator.predict(p_raw), 1e-7, 1.0 - 1e-7)


__all__ = [
    "calibrated_predict",
    "fit_isotonic_calibrator",
    "load_isotonic_calibrator",
]
