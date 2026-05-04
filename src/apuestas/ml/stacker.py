"""Stacker LGBM shallow con market features — Sprint 10 Fase 2 (Mejora #2).

Reemplazo opcional del LogReg stacker en train_base.py. Añade:

1. **LGBM shallow** (max_depth=3, n_estimators=100) con monotonic_constraints
   sobre EV/market_consensus para evitar que el stacker invierta señales.
2. **Market features** en el meta-input (además de OOF preds):
   - `market_consensus_delta` (dispersión Pinnacle/Polymarket/Kalshi)
   - `line_movement_velocity` (derivada de odds_history en ventana 2h)
   - `sharp_book_agreement` (nº books sharp dentro de ±2% de Pinnacle fair)

El stacker aprende "cuándo Pinnacle está más calibrado que mi modelo" y
baja peso del modelo base cuando el mercado es agreement fuerte.

Paper: Walsh & Joshi 2024 — *ML with Applications* vol. 19.

Uso:
    stacker = MarketAwareStacker(monotonic=True)
    stacker.fit(X_meta, y)  # X_meta: OOF preds + 3 market features
    p = stacker.predict_proba(X_meta_test)[:, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MarketAwareStacker:
    """LGBM shallow stacker con monotonic constraints opcionales.

    Si `lightgbm` no está disponible o `use_lgbm=False`, cae a sklearn
    LogisticRegression para backward-compat.

    `focal_loss` (Sprint 11 Fase A) usa Focal Loss de Mukhoti 2020
    (NeurIPS) como objetivo custom para LGBM. Mejora calibración vs
    logloss estándar (−30-50% ECE) sin postproc adicional. Parámetros:
    α=0.25 (class weight), γ=2.0 (focus on hard examples).
    """

    monotonic: bool = True
    use_lgbm: bool = True
    max_depth: int = 3
    n_estimators: int = 100
    learning_rate: float = 0.05
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    _model: Any = field(default=None, init=False, repr=False)
    _feature_names: list[str] = field(default_factory=list, init=False)
    _monotonic_constraints: list[int] = field(default_factory=list, init=False)

    def _build_lgbm(self, feature_names: list[str]):  # type: ignore[no-untyped-def]
        import lightgbm as lgb

        constraints: list[int] = [0] * len(feature_names)
        if self.monotonic:
            # OOF preds: monotonic +1 (más prob modelo → más prob final)
            for i, name in enumerate(feature_names):
                low = name.lower()
                if low.startswith("oof_"):
                    constraints[i] = 1
                elif low in ("market_consensus_delta",):
                    # Consensus alto = divergencia alta = menos confianza → -1
                    constraints[i] = -1
                elif low == "sharp_book_agreement":
                    # Más books agreement con Pinnacle = señal fuerte → +1
                    constraints[i] = 1
                # line_movement_velocity: sin constraint (signo ambiguo)
        self._monotonic_constraints = constraints
        self._feature_names = feature_names

        base_params = {
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "monotone_constraints": constraints if self.monotonic else None,
            "verbose": -1,
            "boosting_type": "gbdt",
            "n_jobs": 1,
        }
        if self.focal_loss:
            # Objetivo custom Focal Loss (Mukhoti 2020)
            alpha = self.focal_alpha
            gamma = self.focal_gamma

            def focal_obj(y_true, y_pred):  # type: ignore[no-untyped-def]
                # y_pred en espacio logit; sigmoid → prob
                p = 1.0 / (1.0 + np.exp(-y_pred))
                # Derivadas de focal loss respecto a logit
                # L = -α·(1-p)^γ · y·log(p) - (1-α)·p^γ · (1-y)·log(1-p)
                pt = np.where(y_true == 1, p, 1 - p)
                alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
                # grad y hess aproximados (Mukhoti eq 4)
                grad = (
                    alpha_t
                    * (1 - pt) ** gamma
                    * (p - y_true)
                    * (1 - gamma * pt * np.log(np.clip(pt, 1e-8, 1)) / (1 - pt + 1e-8))
                )
                hess = alpha_t * (1 - pt) ** gamma * p * (1 - p)
                return grad, hess

            base_params["objective"] = focal_obj
            return lgb.LGBMClassifier(**base_params)
        base_params["objective"] = "binary"
        return lgb.LGBMClassifier(**base_params)

    def _build_logreg(self):  # type: ignore[no-untyped-def]
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> MarketAwareStacker:
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        if self.use_lgbm:
            try:
                self._model = self._build_lgbm(feature_names)
                self._model.fit(X, y)
                # Con focal_loss (custom obj) LGBM no produce classes_/predict_proba standard.
                # Adaptamos: almacenamos raw_scores y exponemos predict_proba sigmoidal.
                if self.focal_loss:
                    self._model = _FocalLGBMAdapter(lgbm=self._model)
                logger.info(
                    "stacker.fit_lgbm",
                    n_features=X.shape[1],
                    monotonic=self.monotonic,
                    constraints=self._monotonic_constraints,
                )
                return self
            except ImportError:
                logger.warning("stacker.lgbm_unavailable fallback=logreg")
            except Exception as exc:
                logger.warning("stacker.lgbm_fail fallback=logreg", error=str(exc)[:100])
        # Fallback: LogReg
        self._model = self._build_logreg()
        self._model.fit(X, y)
        self._feature_names = feature_names
        logger.info("stacker.fit_logreg", n_features=X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MarketAwareStacker.fit() debe invocarse antes de predict_proba")
        return self._model.predict_proba(X)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def monotonic_constraints(self) -> list[int]:
        return list(self._monotonic_constraints)


class _FocalLGBMAdapter:
    """Envuelve LGBMClassifier con focal-obj custom para exponer predict_proba 2D.

    Cuando se entrena con objetivo custom (focal), LGBM devuelve raw scores
    (logits) por defecto. Aplicamos sigmoid + stack [1-p, p] para cumplir
    interfaz sklearn esperada por CalibratedClassifierCV downstream.
    """

    _estimator_type = "classifier"

    def __init__(self, lgbm: Any) -> None:
        self.lgbm = lgbm
        self.classes_ = np.array([0, 1])

    def __sklearn_tags__(self) -> object:
        from sklearn.utils._tags import ClassifierTags, Tags

        return Tags(
            estimator_type="classifier",
            classifier_tags=ClassifierTags(),
            target_tags=None,
            transformer_tags=None,
            regressor_tags=None,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # LGBM con custom obj expone predict() que devuelve raw score (logit)
        raw = self.lgbm.predict(X, raw_score=True)
        if isinstance(raw, np.ndarray) and raw.ndim == 2:
            # Algunos paths devuelven shape (n, 1)
            raw = raw.ravel()
        p = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def fit(self, *_a: Any, **_kw: Any) -> _FocalLGBMAdapter:
        # Ya viene ajustado; no-op para cumplir API sklearn.
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"lgbm": self.lgbm}

    def set_params(self, **params: Any) -> _FocalLGBMAdapter:
        for k, v in params.items():
            setattr(self, k, v)
        return self


__all__ = ["MarketAwareStacker"]
