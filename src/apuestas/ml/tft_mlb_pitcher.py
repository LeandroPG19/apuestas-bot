"""Temporal Fusion Transformer MLB Pitcher ERA — Sprint 14 #142.

Basado en ScienceDirect 2025: TFT consistentemente superó XGBoost/LightGBM
en pitcher ERA prediction usando pitch-level Statcast + contextual covariates
(park, opposing lineup strength, weather).

Arquitectura TFT (Lim et al. 2021):
  - Variable selection network (static + time-varying known + observed)
  - Temporal attention heads (interpretability)
  - Quantile output (p10, p50, p90)

Dependency: pytorch-forecasting + pytorch-lightning.

Este skeleton define la interfaz; producción real requiere:
  1. Instalar pytorch-forecasting
  2. Build dataset `TimeSeriesDataSet` con encoder_length=15 games
  3. Fit TFT con lr=1e-3, epochs=30

Fallback: si torch/pytorch_forecasting no disponible, retorna None.
Modelo principal LGBM sigue funcionando.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TFTConfig:
    encoder_length: int = 15
    max_prediction_length: int = 3
    hidden_size: int = 64
    n_attention_heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    max_epochs: int = 30
    batch_size: int = 64
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


def check_dependencies() -> tuple[bool, str]:
    """Verifica torch + pytorch-forecasting instalados."""
    try:
        import pytorch_forecasting  # noqa: F401
        import torch  # noqa: F401

        return True, "OK"
    except ImportError as exc:
        return False, f"missing: {exc.name}"


class TFTPitcherEraModel:
    """Wrapper TFT. fit/predict_proba compat con sklearn interface.

    Producción: reemplazar stub con TemporalFusionTransformer real.
    """

    def __init__(self, cfg: TFTConfig | None = None):
        self.cfg = cfg or TFTConfig()
        self._fitted = False
        self._fallback = False

    def fit(self, X, y) -> TFTPitcherEraModel:
        ok, reason = check_dependencies()
        if not ok:
            logger.warning("tft.dependencies_missing", reason=reason)
            self._fallback = True
            self._fitted = True
            return self
        # Production: wire pytorch_forecasting.TemporalFusionTransformer here
        logger.info("tft.fit.stub", n=len(y))
        self._fitted = True
        return self

    def predict(self, X) -> Any:
        if self._fallback or not self._fitted:
            # Fallback: mean ERA
            import numpy as np

            return np.full(len(X), 4.0)
        logger.info("tft.predict.stub", n=len(X))
        import numpy as np

        return np.full(len(X), 4.0)

    def predict_quantiles(self, X, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> dict:
        """Devuelve quantiles por prediction. Stub retorna mean ± 1.0."""
        mean = self.predict(X)
        return {q: mean + (q - 0.5) * 2.0 for q in quantiles}


__all__ = ["TFTConfig", "TFTPitcherEraModel", "check_dependencies"]
