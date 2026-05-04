"""MC Dropout calibration wrapper — Sprint 14 #140 (part).

Basado en MDPI Information 17(1):56 enero 2026 "Uncertainty-Aware ML for NBA
Forecasting". Aplica dropout activo en inference, samplea N predicciones,
calcula media + std → intervalos calibrados.

Uso: wrap cualquier NN/LGBM con dropout_prob>0 para obtener uncertainty:

    from apuestas.ml.mc_dropout_nba import mc_dropout_predict_proba

    # con modelo torch o lightgbm con dropout:
    p_mean, p_lower, p_upper = mc_dropout_predict_proba(model, X, n_samples=50)

Este archivo expone helpers + integration test. Requiere modelo base con
dropout NO determinístico (LGBM `extra_trees=True` + subsample_freq o PyTorch nn.Dropout).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mc_dropout_predict_proba(
    estimator: Any, X: np.ndarray, *, n_samples: int = 50, ci: float = 0.95
) -> dict[str, np.ndarray]:
    """Sample N predicciones con dropout activo, retorna (mean, ci_low, ci_high).

    Si estimator no soporta stochastic predict, retorna mean con std=0.
    Ejemplo PyTorch: aplicar model.train() antes de predict_proba N veces.
    Ejemplo LGBM: subsample+bagging_freq sampleando N bosques.
    """
    predictions: list[np.ndarray] = []
    stochastic = hasattr(estimator, "_supports_mc_dropout") and estimator._supports_mc_dropout

    for _ in range(n_samples):
        try:
            p = estimator.predict_proba(X)
            predictions.append(np.asarray(p))
        except Exception:
            continue
        if not stochastic:
            # Determinístico: no point iterating
            break

    if not predictions:
        return {
            "mean": np.zeros(X.shape[0]),
            "ci_low": np.zeros(X.shape[0]),
            "ci_high": np.zeros(X.shape[0]),
        }

    arr = np.stack(predictions)  # (N, batch, classes)
    mean = arr.mean(axis=0)
    alpha = (1.0 - ci) / 2.0
    ci_low = np.quantile(arr, alpha, axis=0)
    ci_high = np.quantile(arr, 1.0 - alpha, axis=0)
    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}


def ece_from_mc_samples(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error sobre muestras MC."""
    if len(probs) == 0:
        return 0.0
    if probs.ndim > 1:
        probs = probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


__all__ = ["ece_from_mc_samples", "mc_dropout_predict_proba"]
