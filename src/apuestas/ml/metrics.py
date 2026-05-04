"""Métricas primarias de calidad del bot post-pivote.

Al eliminar bankroll/PnL, la única ancla objetiva para validar que los
picks emitidos son +EV reales (no ruido) es la calibración. Plan §7.2
define los KPIs obligatorios:

    log_loss_rolling_30d    ≤ 0.65 NBA, 0.68 NFL
    brier_rolling_30d       ≤ 0.22 NBA, 0.23 NFL
    brier_skill_score_30d   ≥ 0.03 (positivo = mejor que climatología)
    ece_10bins_30d          ≤ 0.05
    hit_rate − implied_rate ≥ +2 pp

Este módulo expone funciones puras; `walk_forward.py` las consume sobre
picks históricos. El ECE ya vive en `apuestas.ml.calibrate` y aquí lo
re-exportamos para tener una API única.

Referencias:
  - Brier 1950, Monthly Weather Review 78(1)
  - Gneiting & Raftery 2007, JASA 102(477) — proper scoring rules
  - Walsh & Joshi 2024, ML with Applications v19 — calibración > accuracy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apuestas.ml.calibrate import expected_calibration_error


@dataclass(frozen=True, slots=True)
class MetricsResult:
    n: int
    log_loss: float
    brier: float
    brier_skill_score: float
    ece: float
    hit_rate: float
    implied_rate: float
    hit_rate_minus_implied: float


def _clip(arr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.clip(arr, eps, 1.0 - eps)


def log_loss_binary(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """−mean(y·log p + (1−y)·log(1−p)). Nunca infinita gracias al clip."""
    y = np.asarray(y_true, dtype=np.float64)
    p = _clip(np.asarray(p_pred, dtype=np.float64))
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Brier = mean((p − y)²). ∈ [0, 1]. Menor = mejor."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(y_true: np.ndarray, p_pred: np.ndarray, *, p_climatology: float) -> float:
    """BSS = 1 − (BS_modelo / BS_climatología).

    Un BSS > 0 indica que el modelo supera al predictor climatológico
    (asignar a todos la misma tasa base). BSS ≤ 0 ⇒ el modelo no tiene
    skill. Gneiting & Raftery 2007.
    """
    bs_model = brier_score(y_true, p_pred)
    # BS de la climatología = p̄ · (1 − p̄) cuando y∈{0,1} y p_climatology fijo.
    y = np.asarray(y_true, dtype=np.float64)
    bs_clim = float(np.mean((p_climatology - y) ** 2))
    if bs_clim <= 0:
        return 0.0
    return 1.0 - (bs_model / bs_clim)


def hit_rate(y_true: np.ndarray, p_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    """Fracción de aciertos cuando la decisión es argmax (binario: p ≥ threshold)."""
    y = np.asarray(y_true, dtype=np.int8)
    pred = (np.asarray(p_pred) >= threshold).astype(np.int8)
    if len(y) == 0:
        return 0.0
    return float((pred == y).mean())


def implied_rate_from_odds(avg_odds: float) -> float:
    """Tasa implícita del mercado: 1 / odds_media. Sirve como baseline
    de skill — hit_rate debe superar esto para tener edge real.
    """
    if avg_odds <= 1.0:
        return 0.0
    return 1.0 / float(avg_odds)


def compute_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    avg_odds: float | None = None,
    p_climatology: float | None = None,
    ece_bins: int = 10,
) -> MetricsResult:
    """Calcula la tabla completa de KPIs para un batch de picks.

    Args:
        y_true: 0/1 array con el resultado real de cada pick.
        p_pred: probabilidad asignada por el modelo (0..1).
        avg_odds: media de odds tomadas (para implied_rate baseline).
        p_climatology: tasa base del deporte; si None, se usa mean(y_true).
        ece_bins: buckets para ECE (default 10 — más interpretable que 15).
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    n = len(y)
    if n == 0:
        return MetricsResult(
            n=0,
            log_loss=float("nan"),
            brier=float("nan"),
            brier_skill_score=float("nan"),
            ece=float("nan"),
            hit_rate=float("nan"),
            implied_rate=float("nan"),
            hit_rate_minus_implied=float("nan"),
        )

    p_clim = float(p_climatology) if p_climatology is not None else float(y.mean())
    ll = log_loss_binary(y, p)
    bs = brier_score(y, p)
    bss = brier_skill_score(y, p, p_climatology=p_clim)
    ece = float(expected_calibration_error(y.astype(int), p, n_bins=ece_bins))
    hr = hit_rate(y, p)
    ir = implied_rate_from_odds(avg_odds) if avg_odds is not None else p_clim
    return MetricsResult(
        n=n,
        log_loss=ll,
        brier=bs,
        brier_skill_score=bss,
        ece=ece,
        hit_rate=hr,
        implied_rate=ir,
        hit_rate_minus_implied=hr - ir,
    )


__all__ = [
    "MetricsResult",
    "brier_score",
    "brier_skill_score",
    "compute_metrics",
    "hit_rate",
    "implied_rate_from_odds",
    "log_loss_binary",
]
