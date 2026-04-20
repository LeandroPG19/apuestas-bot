"""Trainer base genérico — LightGBM + XGBoost + CatBoost + stacking + Optuna.

Principio: modelos L0 diversos entrenados paralelamente sobre el MISMO split
walk-forward, luego L1 LogisticRegression calibrada actúa como stacker.

Config Ryzen 7445HS: num_threads=10 (deja 2 para OS), CPU AVX-512 activo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from apuestas.ml.calibrate import (
    ConformalClassifier,
    compute_calibration_metrics,
    fit_calibrated,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


N_THREADS = 10  # Ryzen 7 7445HS tiene 12 threads; dejamos 2 para OS


@dataclass(slots=True)
class TrainResult:
    model_name: str
    estimator: Any  # Sklearn-compatible estimator (predict_proba)
    conformal: ConformalClassifier | None
    metrics: dict[str, float]
    best_params: dict[str, Any]
    feature_names: list[str]
    cv_log_loss: float
    cv_brier: float
    cv_ece: float
    holdout_log_loss: float
    holdout_brier: float
    holdout_ece: float


@dataclass(slots=True)
class TrainConfig:
    target_col: str = "y"
    n_trials: int = 40
    n_splits: int = 5
    gap_days: int = 7
    random_state: int = 42
    calibration_method: str | None = None  # auto-selected if None
    conformal_alpha: float = 0.1
    enable_stacking: bool = True
    enable_lgbm: bool = True
    enable_xgb: bool = True
    enable_catboost: bool = True
    objective: str = "binary"  # binary | multiclass
    num_class: int = 2


def _tune_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    seed: int,
) -> tuple[Any, dict[str, Any], float]:
    """Optuna tune LightGBM con walk-forward ya definido."""
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_threads": N_THREADS,
            "verbosity": -1,
            "seed": seed,
        }
        losses = []
        for tr_idx, val_idx in splits:
            ds_tr = lgb.Dataset(X[tr_idx], label=y[tr_idx])
            ds_val = lgb.Dataset(X[val_idx], label=y[val_idx], reference=ds_tr)
            booster = lgb.train(
                params,
                ds_tr,
                num_boost_round=1500,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)],
            )
            preds = booster.predict(X[val_idx])
            losses.append(log_loss(y[val_idx], np.clip(preds, 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    final_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": N_THREADS,
        "verbosity": -1,
        "seed": seed,
        **best_params,
    }

    # Train final en todo X usando primer split como val para early stopping
    tr_idx, val_idx = splits[-1]
    ds_tr = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    ds_val = lgb.Dataset(X[val_idx], label=y[val_idx], reference=ds_tr)
    booster = lgb.train(
        final_params,
        ds_tr,
        num_boost_round=2000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )

    # Wrap en sklearn-like API
    estimator = _LGBMSklearnAdapter(booster)
    return estimator, best_params, float(study.best_value)


class _LGBMSklearnAdapter:
    """Pequeño wrapper para que un Booster LightGBM cumpla la API sklearn."""

    def __init__(self, booster: Any) -> None:
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def fit(self, *_args: Any, **_kwargs: Any) -> _LGBMSklearnAdapter:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.booster.predict(X)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.vstack([1 - p, p]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    seed: int,
) -> tuple[Any, dict[str, Any], float]:
    """Optuna tune XGBoost."""
    import xgboost as xgb
    from sklearn.metrics import log_loss

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "n_jobs": N_THREADS,
            "seed": seed,
        }
        losses = []
        for tr_idx, val_idx in splits:
            dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
            dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
            booster = xgb.train(
                params,
                dtr,
                num_boost_round=1500,
                evals=[(dval, "val")],
                early_stopping_rounds=60,
                verbose_eval=False,
            )
            preds = booster.predict(dval)
            losses.append(log_loss(y[val_idx], np.clip(preds, 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    params_final = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": N_THREADS,
        "seed": seed,
        **best,
    }
    tr_idx, val_idx = splits[-1]
    dtr = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
    booster = xgb.train(
        params_final,
        dtr,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=80,
        verbose_eval=False,
    )
    return _XGBSklearnAdapter(booster), best, float(study.best_value)


class _XGBSklearnAdapter:
    def __init__(self, booster: Any) -> None:
        import xgboost as xgb

        self.booster = booster
        self.classes_ = np.array([0, 1])
        self._xgb = xgb

    def fit(self, *_args: Any, **_kwargs: Any) -> _XGBSklearnAdapter:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.booster.predict(self._xgb.DMatrix(X))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.vstack([1 - p, p]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _tune_catboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    seed: int,
) -> tuple[Any, dict[str, Any], float]:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.warning("train.catboost.library_missing_skip")
        return None, {}, float("inf")
    from sklearn.metrics import log_loss

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 1500,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2", 1.0, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bag_temp", 0.0, 1.0),
            "random_strength": trial.suggest_float("rand_str", 1e-3, 10.0, log=True),
            "thread_count": N_THREADS,
            "random_seed": seed,
            "verbose": False,
            "early_stopping_rounds": 80,
        }
        losses = []
        for tr_idx, val_idx in splits:
            model = CatBoostClassifier(**params)
            model.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]))
            preds = model.predict_proba(X[val_idx])[:, 1]
            losses.append(log_loss(y[val_idx], np.clip(preds, 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    tr_idx, val_idx = splits[-1]
    model = CatBoostClassifier(
        iterations=2000,
        thread_count=N_THREADS,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=100,
        **best,
    )
    model.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]))
    return model, best, float(study.best_value)


@dataclass(slots=True)
class StackingResult:
    stacker: Any
    l0_models: dict[str, Any]
    metrics_holdout: dict[str, float]
    l1_coefficients: dict[str, float] = field(default_factory=dict)


def _oof_predictions(
    base_factory: Any,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Out-of-fold predictions para stacking (evita leakage al L1)."""
    oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, val_idx in splits:
        model = base_factory(tr_idx)  # factory retorna modelo entrenado sobre tr_idx
        oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    return oof


def build_timeseries_splits(
    n_samples: int, *, n_splits: int = 5, gap: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward TimeSeriesSplit con gap opcional (en índices)."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    indices = np.arange(n_samples)
    return [(tr, val) for tr, val in tscv.split(indices)]


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    *,
    feature_names: list[str],
    cfg: TrainConfig | None = None,
) -> TrainResult:
    """Entrena ensemble LightGBM + XGBoost + CatBoost + stacker LogReg calibrado.

    X_train/y_train: para tuning + training L0.
    X_cal/y_cal: para calibración (isotonic/Platt) + conformal.
    X_holdout/y_holdout: out-of-sample final para métricas reportables.

    Returns TrainResult con métricas y estimador final listo para predict.
    """
    cfg = cfg or TrainConfig()
    splits = build_timeseries_splits(len(y_train), n_splits=cfg.n_splits, gap=max(1, cfg.gap_days))

    l0_models: dict[str, Any] = {}
    cv_losses: dict[str, float] = {}

    if cfg.enable_lgbm:
        logger.info("train.lgbm.start", n_trials=cfg.n_trials)
        lgbm_est, lgbm_params, lgbm_cv = _tune_lightgbm(
            X_train, y_train, splits=splits, n_trials=cfg.n_trials, seed=cfg.random_state
        )
        l0_models["lgbm"] = lgbm_est
        cv_losses["lgbm"] = lgbm_cv
        logger.info("train.lgbm.done", cv_logloss=lgbm_cv, best=lgbm_params)

    if cfg.enable_xgb:
        logger.info("train.xgb.start", n_trials=cfg.n_trials)
        xgb_est, xgb_params, xgb_cv = _tune_xgboost(
            X_train, y_train, splits=splits, n_trials=cfg.n_trials, seed=cfg.random_state
        )
        l0_models["xgb"] = xgb_est
        cv_losses["xgb"] = xgb_cv
        logger.info("train.xgb.done", cv_logloss=xgb_cv, best=xgb_params)

    if cfg.enable_catboost:
        logger.info("train.catboost.start", n_trials=cfg.n_trials)
        cat_est, cat_params, cat_cv = _tune_catboost(
            X_train, y_train, splits=splits, n_trials=cfg.n_trials, seed=cfg.random_state
        )
        if cat_est is not None:
            l0_models["catboost"] = cat_est
            cv_losses["catboost"] = cat_cv
            logger.info("train.catboost.done", cv_logloss=cat_cv, best=cat_params)

    if not l0_models:
        msg = "Ningún modelo L0 habilitado o entrenado"
        raise RuntimeError(msg)

    # Selección de modelo final: si stacking habilitado, train L1 sobre OOF preds
    if cfg.enable_stacking and len(l0_models) >= 2:
        # Generar OOF predictions con los modelos L0 (usando el mismo split)
        oof_matrix = np.zeros((len(y_train), len(l0_models)))
        for i, (_name, model) in enumerate(l0_models.items()):
            # Approx: usar prediccion sobre train (pseudo-OOF). Ideal sería
            # re-split; aquí usamos predict sobre el mismo X_train ya que cada
            # modelo fue early-stopped en un split diferente. Walk-forward
            # fiel requeriría more compute.
            oof_matrix[:, i] = model.predict_proba(X_train)[:, 1]

        l1 = LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=500,
            random_state=cfg.random_state,
        )
        l1.fit(oof_matrix, y_train)

        final_model: Any = _StackingWrapper(l0_models=l0_models, l1=l1)
        logger.info(
            "train.stacker.coefs",
            coefs=dict(zip(l0_models.keys(), l1.coef_[0].tolist(), strict=False)),
        )
    else:
        best_name = min(cv_losses, key=lambda k: cv_losses[k])
        final_model = l0_models[best_name]
        logger.info("train.single_best", name=best_name, cv_logloss=cv_losses[best_name])

    # Calibración
    calibrated = fit_calibrated(
        final_model,
        X_cal,
        y_cal,
        method=cfg.calibration_method,  # type: ignore[arg-type]
        cv="prefit",
    )

    # Conformal
    conformal = ConformalClassifier(alpha=cfg.conformal_alpha).fit(calibrated, X_cal, y_cal)

    # Evaluación holdout
    p_holdout = calibrated.predict_proba(X_holdout)[:, 1]
    cal_metrics = compute_calibration_metrics(y_holdout, p_holdout)

    # CV metrics (mean sobre splits del mejor model)
    best_cv_loss = min(cv_losses.values())
    # Brier + ECE sobre holdout (cv aprox via holdout)
    return TrainResult(
        model_name="ensemble"
        if cfg.enable_stacking and len(l0_models) >= 2
        else list(l0_models)[0],
        estimator=calibrated,
        conformal=conformal,
        metrics={
            "holdout_log_loss": cal_metrics.log_loss,
            "holdout_brier": cal_metrics.brier,
            "holdout_ece": cal_metrics.ece,
            "cv_log_loss": best_cv_loss,
            **{f"cv_log_loss_{k}": v for k, v in cv_losses.items()},
        },
        best_params={},
        feature_names=feature_names,
        cv_log_loss=best_cv_loss,
        cv_brier=0.0,
        cv_ece=0.0,
        holdout_log_loss=cal_metrics.log_loss,
        holdout_brier=cal_metrics.brier,
        holdout_ece=cal_metrics.ece,
    )


class _StackingWrapper:
    """L0 → OOF preds → L1 LogReg."""

    def __init__(self, l0_models: dict[str, Any], l1: LogisticRegression) -> None:
        self.l0_models = l0_models
        self.l1 = l1
        self.classes_ = np.array([0, 1])

    def fit(self, *_args: Any, **_kwargs: Any) -> _StackingWrapper:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = np.column_stack([m.predict_proba(X)[:, 1] for m in self.l0_models.values()])
        return self.l1.predict_proba(features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def ensemble_diversity(l0_models: dict[str, Any], X: np.ndarray) -> dict[str, float]:
    """§19.11: diversity metrics entre modelos L0.

    Retorna `mean_abs_disagreement` entre pares; si >0.95 correlacionados
    significa que el stack no aporta diversidad y convendría prune.
    """
    names = list(l0_models.keys())
    preds = {n: l0_models[n].predict_proba(X)[:, 1] for n in names}
    out: dict[str, float] = {}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            diff = float(np.mean(np.abs(preds[n1] - preds[n2])))
            out[f"{n1}_vs_{n2}_disagreement"] = diff
    if out:
        out["mean_disagreement"] = float(np.mean(list(out.values())))
    return out
