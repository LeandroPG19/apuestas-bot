"""Catchall baseline model — Sprint 13 Capa 6.

Modelo sklearn-compat que **siempre** está disponible como fallback cuando
no hay modelo entrenado para (sport, market, league). Usa ÚNICAMENTE
Pinnacle de-vigged como `p_blended`. No asume skill propio; solo re-expone
el consensus del mercado sharp.

Beneficio: el detector NUNCA skipea un partido por falta de modelo. Puede
evaluar si cualquier otra casa paga mejor que Pinnacle (edge operacional
vía line shopping + book_power_ratings).

Uso:
    from apuestas.ml.catchall_baseline import CatchallBaselineModel
    model = CatchallBaselineModel()
    # X[i] = [pinnacle_fair_prob_home]
    probs = model.predict_proba(X)  # shape (n, 2)
"""

from __future__ import annotations

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


class CatchallBaselineModel:
    """Fallback universal: p_blended = p_pinnacle_fair (de-vigged).

    Feature esperado: X[:, 0] = probabilidad fair Pinnacle de la outcome.
    Si X tiene más columnas, se ignoran (compat con feature frames más
    grandes).

    Esto permite al detector correr sobre TODO partido con quotes Pinnacle
    aunque no haya modelo específico entrenado.
    """

    _estimator_type = "classifier"

    def __init__(self, name: str = "catchall_baseline", version: str = "v1") -> None:
        self.name = name
        self.version = version
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.array(["pinnacle_fair_prob"])

    def __sklearn_tags__(self) -> object:
        from sklearn.utils._tags import ClassifierTags, Tags

        return Tags(
            estimator_type="classifier",
            classifier_tags=ClassifierTags(),
            target_tags=None,
            transformer_tags=None,
            regressor_tags=None,
        )

    def fit(
        self, X: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> CatchallBaselineModel:
        """No-op fit. Este modelo NO aprende; solo expone Pinnacle fair."""
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Devuelve (n, 2) con p_home=X[:, 0] y p_away=1-X[:, 0]."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        p_home = np.clip(X[:, 0].astype(float), 0.001, 0.999)
        return np.column_stack([1.0 - p_home, p_home])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {"name": self.name, "version": self.version}

    def set_params(self, **params) -> CatchallBaselineModel:  # type: ignore[no-untyped-def]
        for k, v in params.items():
            setattr(self, k, v)
        return self


__all__ = ["CatchallBaselineModel"]
