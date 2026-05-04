"""TabPFN-v2 stacker — Sprint 11 Fase B.

Foundation model tabular pre-entrenado en 10k tareas sintéticas (Hollmann 2023,
Jan 2025 updates). Zero-shot en datasets nuevos. Ventaja: outputs naturalmente
calibrados, superior a LGBM/XGB en datasets chicos (<2000 muestras).

Paper: "From Tables to Time: TabPFN-v2 Extended to Time Series Forecasting"
(arXiv 2501.02945, NeurIPS TRL workshop Jan 2025).

Uso opt-in en `train_base.py` via env `APUESTAS_USE_TABPFN_STACKER=true`.
Fallback a LogReg si la librería no está disponible o falla.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TabPFNStacker:
    """Wrapper sklearn-compat para TabPFN como L1 stacker."""

    device: str = "cpu"  # "cuda" si disponible
    n_estimators: int = 4  # TabPFN usa ensembling interno
    _model: Any = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> TabPFNStacker:
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        self._feature_names = feature_names
        try:
            from tabpfn import TabPFNClassifier

            # TabPFN v2 espera máx ~500 features y ~3000 samples; truncar si
            # necesario (meta-dataset suele ser < 3000 por TimeSeriesSplit).
            n_max = min(len(X), 3000)
            X_fit = X[:n_max]
            y_fit = y[:n_max]
            self._model = TabPFNClassifier(
                device=self.device,
                n_estimators=self.n_estimators,
            )
            self._model.fit(X_fit, y_fit)
            logger.info(
                "tabpfn.fit.ok",
                n_features=X.shape[1],
                n_samples=n_max,
                device=self.device,
            )
            return self
        except ImportError:
            logger.warning("tabpfn.import_fail fallback=logreg")
        except Exception as exc:
            logger.warning("tabpfn.fit_fail fallback=logreg", error=str(exc)[:100])

        # Fallback: LogReg
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabPFNStacker.fit() debe invocarse antes")
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"device": self.device, "n_estimators": self.n_estimators}

    def set_params(self, **params: Any) -> TabPFNStacker:
        for k, v in params.items():
            setattr(self, k, v)
        return self


__all__ = ["TabPFNStacker"]
