"""Fase 4.12 — SHAP explainability por pick persistido.

Hook tras emitir cada pick: compute top-5 SHAP feature contributors → persist
en `predictions.shap_top_features` JSONB (migración 0013 añade columna).

Permite auditar cada pick: "por qué el modelo eligió este outcome".

Uso:
    from apuestas.ml.shap_persist import persist_shap_for_prediction
    await persist_shap_for_prediction(prediction_id, model, features)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def compute_top_shap_features(
    model: Any,
    features: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    """Compute top-k SHAP contributors. Retorna lista de dicts.

    Fallback graceful: si shap library no disponible, retorna lista vacía.
    """
    try:
        import shap  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("shap.library_missing")
        return []

    if features.ndim == 1:
        features_2d = features.reshape(1, -1)
    else:
        features_2d = features

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_2d)
        # Para clasificadores binarios, shap_values es list[2] arrays
        if isinstance(shap_values, list):
            sv = shap_values[1]  # class 1 (positive)
        else:
            sv = shap_values

        # Top k por |contribution|
        abs_contributions = np.abs(sv[0])
        top_indices = np.argsort(abs_contributions)[::-1][:top_k]
        result: list[dict[str, float | str]] = []
        for idx in top_indices:
            result.append(
                {
                    "feature": feature_names[idx] if idx < len(feature_names) else f"feat_{idx}",
                    "value": float(features_2d[0, idx]),
                    "contribution": float(sv[0, idx]),
                }
            )
        return result
    except Exception as exc:
        logger.debug("shap.compute_fail", error=str(exc)[:80])
        return []


async def persist_shap_for_prediction(
    prediction_id: int,
    top_features: list[dict[str, float | str]],
) -> None:
    """Persiste SHAP top features en predictions.shap_top_features JSONB."""
    if not top_features:
        return
    import json

    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE predictions
                SET shap_top_features = :shap::jsonb
                WHERE id = :pid
                """
            ),
            {"pid": prediction_id, "shap": json.dumps(top_features)},
        )
    logger.info("shap.persisted", prediction_id=prediction_id, n=len(top_features))
