"""SHAP interpretability: top-5 features por predicción (§17.5).

Se usa shap.TreeExplainer para modelos tree-based (LightGBM/XGBoost/CatBoost).
Para el stacker LogReg sobre OOF preds, los SHAP del booster L0 siguen siendo
la mejor aproximación narrativa (qué métricas de juego movieron la predicción).

Salida: lista de dicts [{"feature": str, "value": float, "shap": float,
"direction": "up"|"down"}] ordenada por |shap| desc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class SHAPFeature:
    feature: str
    value: float
    shap: float
    direction: str  # "up" = empuja hacia p=1, "down" hacia p=0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": float(self.value),
            "shap": float(self.shap),
            "direction": self.direction,
        }


class SHAPExplainer:
    """Wrapper perezoso sobre shap.TreeExplainer.

    Maneja:
    - _StackingWrapper (extrae L0 principal para atribución)
    - CalibratedClassifierCV (desempaca estimator base)
    - Booster LightGBM/XGBoost/CatBoost directos
    """

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self._tree_model = self._extract_tree_model(model)
        self._explainer: Any | None = None
        self._feature_name_to_idx = {n: i for i, n in enumerate(feature_names)}

    @staticmethod
    def _extract_tree_model(model: Any) -> Any:
        """Descarta wrappers hasta encontrar un estimador tree-based."""
        # CalibratedClassifierCV tiene calibrated_classifiers_ con base estimator
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            first = model.calibrated_classifiers_[0]
            est = getattr(first, "estimator", None) or getattr(first, "base_estimator", None)
            if est is not None:
                return SHAPExplainer._extract_tree_model(est)
        # StackingWrapper: usa el primer L0
        if hasattr(model, "l0_models"):
            first_key = next(iter(model.l0_models))
            return SHAPExplainer._extract_tree_model(model.l0_models[first_key])
        # Adaptadores locales
        if hasattr(model, "booster"):
            return model.booster
        return model

    def _ensure_explainer(self) -> Any:
        if self._explainer is not None:
            return self._explainer
        try:
            import shap  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("shap.library_missing")
            return None

        try:
            self._explainer = shap.TreeExplainer(self._tree_model)
            return self._explainer
        except Exception as exc:
            logger.warning("shap.tree_explainer_failed", error=str(exc))
            return None

    def explain_row(self, x: np.ndarray, *, top_k: int = 5) -> list[SHAPFeature]:
        """Devuelve top_k features que más movieron la predicción para este x."""
        explainer = self._ensure_explainer()
        if explainer is None:
            return []

        x2d = x.reshape(1, -1) if x.ndim == 1 else x
        try:
            shap_values = explainer.shap_values(x2d)
        except Exception as exc:
            logger.warning("shap.compute_failed", error=str(exc))
            return []

        # shap_values puede ser:
        # - array (n, n_features) para binario sklearn-style
        # - list [(n, n_features), (n, n_features)] para multiclass
        # Tomar clase positiva
        arr: np.ndarray
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[-1])  # última clase = positiva
        else:
            arr = np.asarray(shap_values)
        if arr.ndim == 3:  # (n, n_features, n_classes)
            arr = arr[:, :, -1]

        row_shap = arr[0]
        row_x = x2d[0]

        # Ordenar por |shap| desc
        idx_sorted = np.argsort(-np.abs(row_shap))[:top_k]
        return [
            SHAPFeature(
                feature=self.feature_names[i],
                value=float(row_x[i]),
                shap=float(row_shap[i]),
                direction="up" if row_shap[i] > 0 else "down",
            )
            for i in idx_sorted
        ]

    def explain_batch(self, X: np.ndarray, *, top_k: int = 5) -> list[list[SHAPFeature]]:
        """Vectorizado: una llamada SHAP para todo el batch."""
        explainer = self._ensure_explainer()
        if explainer is None:
            return [[] for _ in range(len(X))]

        try:
            shap_values = explainer.shap_values(X)
        except Exception as exc:
            logger.warning("shap.batch_failed", error=str(exc))
            return [[] for _ in range(len(X))]

        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[-1])
        else:
            arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, -1]

        results: list[list[SHAPFeature]] = []
        for i in range(len(X)):
            row_shap = arr[i]
            row_x = X[i]
            idx_sorted = np.argsort(-np.abs(row_shap))[:top_k]
            results.append(
                [
                    SHAPFeature(
                        feature=self.feature_names[j],
                        value=float(row_x[j]),
                        shap=float(row_shap[j]),
                        direction="up" if row_shap[j] > 0 else "down",
                    )
                    for j in idx_sorted
                ]
            )
        return results

    def feature_importance(self, X: np.ndarray) -> dict[str, float]:
        """Mean |shap| por feature sobre X (batch de background)."""
        explainer = self._ensure_explainer()
        if explainer is None:
            return {}
        try:
            shap_values = explainer.shap_values(X)
        except Exception as exc:
            logger.warning("shap.importance_failed", error=str(exc))
            return {}
        if isinstance(shap_values, list):
            arr = np.asarray(shap_values[-1])
        else:
            arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        mean_abs = np.mean(np.abs(arr), axis=0)
        return {self.feature_names[i]: float(mean_abs[i]) for i in range(len(self.feature_names))}
