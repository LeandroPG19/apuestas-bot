"""Trainer player props (§23.2).

Estrategia híbrida 3 modelos por prop:
1. Distributional regression LightGBM con objective adaptado:
   - Poisson/Tweedie para counting (NBA points, MLB Ks)
   - Gamma para continuos (NFL yards)
2. TabPFN v2.5 fallback para jugadores con <500 games
3. Monte Carlo empírico (MLB bateador, soccer DC)

Selecciona el mejor por Brier score en walk-forward. Log MLflow con
stage='shadow' por prop_code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apuestas.ml.props_distributions import (
    EmpiricalDist,
    NegBinomialDist,
    PoissonDist,
    PropDistributionProtocol,
    bootstrap_conformal_interval,
    fit_gamma,
    fit_neg_binomial,
    fit_poisson,
)
from apuestas.obs.logging import get_logger
from apuestas.schemas.props import PropDistribution, PropPrediction, get_prop

logger = get_logger(__name__)


@dataclass(slots=True)
class PropTrainingResult:
    prop_code: str
    model_type: str
    brier_holdout: float
    mae_holdout: float
    n_train: int
    n_holdout: int
    distribution_params: dict[str, float]
    feature_names: list[str]


# ═══════════════════════ Poisson/NegBin trainer ══════════════════════


def _fit_param_model(
    samples: np.ndarray,
    distribution: PropDistribution,
) -> PropDistributionProtocol:
    """Fit paramétrico según distribución declarada."""
    if distribution == PropDistribution.POISSON:
        return fit_poisson(samples)
    if distribution == PropDistribution.NEG_BINOMIAL:
        return fit_neg_binomial(samples)
    if distribution == PropDistribution.GAMMA:
        return fit_gamma(samples)
    # Default NegBin (más flexible)
    return fit_neg_binomial(samples)


def train_prop_parametric(
    *,
    prop_code: str,
    historical_samples: np.ndarray,
    holdout_samples: np.ndarray | None = None,
    holdout_lines: list[float] | None = None,
) -> PropTrainingResult:
    """Entrena distribución paramétrica simple sobre historial del jugador.

    Versión base: una distribución global. En producción se condiciona por
    features (matchup, minutes_proj, etc.) vía LightGBM (pendiente Batch H).
    """
    prop_def = get_prop(prop_code)
    dist = _fit_param_model(historical_samples, prop_def.distribution)

    # Evaluar Brier sobre holdout
    brier = 0.0
    mae = 0.0
    if holdout_samples is not None and holdout_lines and len(holdout_samples) > 0:
        brier_sum = 0.0
        mae_sum = 0.0
        for actual, line in zip(holdout_samples, holdout_lines, strict=True):
            p_over = dist.p_over(line)
            actual_over = 1 if actual > line else 0
            brier_sum += (p_over - actual_over) ** 2
            mae_sum += abs(dist.mean - float(actual))
        brier = brier_sum / len(holdout_samples)
        mae = mae_sum / len(holdout_samples)

    params: dict[str, float] = {"mean": dist.mean, "std": dist.std}
    if isinstance(dist, PoissonDist):
        params["lam"] = dist.lam
    elif isinstance(dist, NegBinomialDist):
        params["dispersion"] = dist.dispersion

    return PropTrainingResult(
        prop_code=prop_code,
        model_type=str(prop_def.distribution.value),
        brier_holdout=float(brier),
        mae_holdout=float(mae),
        n_train=len(historical_samples),
        n_holdout=len(holdout_samples) if holdout_samples is not None else 0,
        distribution_params=params,
        feature_names=[],
    )


def train_prop_tabpfn(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    prop_code: str,
) -> PropTrainingResult | None:
    """TabPFN v2.5 para samples chicos (<500). Opcional si tabpfn instalado."""
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore[import-untyped]
    except ImportError:
        logger.info("train_props.tabpfn_not_installed", prop=prop_code)
        return None

    prop_def = get_prop(prop_code)
    if prop_def.category.value == "binary":
        model = TabPFNClassifier(device="auto", random_state=42)
        model.fit(X_train, y_train.astype(int))
        preds_proba = model.predict_proba(X_holdout)[:, 1]
        brier = float(np.mean((preds_proba - y_holdout) ** 2))
        mae = float(np.mean(np.abs(preds_proba - y_holdout)))
    else:
        model = TabPFNRegressor(device="auto", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_holdout)
        mae = float(np.mean(np.abs(preds - y_holdout)))
        brier = 0.0  # no aplica

    logger.info("train_props.tabpfn_done", prop=prop_code, brier=brier, mae=mae)
    return PropTrainingResult(
        prop_code=prop_code,
        model_type="tabpfn_v2.5",
        brier_holdout=brier,
        mae_holdout=mae,
        n_train=len(X_train),
        n_holdout=len(X_holdout),
        distribution_params={},
        feature_names=[],
    )


# ═══════════════════════ Inference (predict at game-time) ════════════


def predict_prop(
    *,
    prop_code: str,
    player_id: int,
    player_name: str,
    event_id: int,
    line: float | None,
    fitted_dist: PropDistributionProtocol,
    features_snapshot: dict[str, float] | None = None,
    model_name: str = "props_v1",
    model_version: str = "shadow",
    n_samples_training: int = 0,
) -> PropPrediction:
    """Convierte distribución fitada en PropPrediction canónica."""
    prop_def = get_prop(prop_code)
    p_over = fitted_dist.p_over(line) if line is not None else None
    p_under = fitted_dist.p_under(line) if line is not None else None

    # Conformal bootstrap CI para p_over
    p_low = p_up = None
    if line is not None and not isinstance(fitted_dist, EmpiricalDist):
        try:
            _, p_low, p_up = bootstrap_conformal_interval(fitted_dist, line, n_bootstrap=200)
        except Exception:
            p_low = p_up = None

    warnings: list[str] = []
    if n_samples_training < 20:
        warnings.append("small_sample_size")
    if n_samples_training < 5:
        warnings.append("limited_weather_history")

    return PropPrediction(
        prop_code=prop_code,
        player_id=player_id,
        player_name=player_name,
        event_id=event_id,
        line=line,
        mean=fitted_dist.mean,
        std=fitted_dist.std,
        p_over=p_over,
        p_under=p_under,
        p_exact=None,
        p_over_lower=p_low,
        p_over_upper=p_up,
        distribution=prop_def.distribution,
        n_samples_training=n_samples_training,
        model_name=model_name,
        model_version=model_version,
        features_snapshot=features_snapshot,
        warnings=tuple(warnings),
    )
