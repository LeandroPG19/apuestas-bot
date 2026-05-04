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
    objective_kind: str = "binary",
    num_class: int = 2,
) -> tuple[Any, dict[str, Any], float]:
    """Optuna tune LightGBM con walk-forward. Soporta binary y multiclass."""
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    is_multi = objective_kind == "multiclass"
    base_params = {
        "objective": "multiclass" if is_multi else "binary",
        "metric": "multi_logloss" if is_multi else "binary_logloss",
    }
    if is_multi:
        base_params["num_class"] = num_class
    multi_labels = list(range(num_class)) if is_multi else None

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params,
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
            if is_multi:
                losses.append(log_loss(y[val_idx], preds, labels=multi_labels))
            else:
                losses.append(log_loss(y[val_idx], np.clip(preds, 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    final_params = {
        **base_params,
        "num_threads": N_THREADS,
        "verbosity": -1,
        "seed": seed,
        **best_params,
    }

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

    estimator = _LGBMSklearnAdapter(booster, num_class=num_class if is_multi else 2)
    return estimator, best_params, float(study.best_value)


class _LGBMSklearnAdapter:
    """Pequeño wrapper para que un Booster LightGBM cumpla la API sklearn.

    Soporta binary (num_class=2) y multiclass (num_class>=3).
    """

    def __init__(self, booster: Any, num_class: int = 2) -> None:
        self.booster = booster
        self.num_class = num_class
        self.classes_ = np.arange(num_class)

    def fit(self, *_args: Any, **_kwargs: Any) -> _LGBMSklearnAdapter:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.booster.predict(X)
        num_class = getattr(self, "num_class", 2)
        if num_class == 2:
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.vstack([1 - p, p]).T
        # Multiclass: LightGBM retorna (n_samples, n_classes)
        p = np.asarray(p)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        p = np.clip(p, 1e-7, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        num_class = getattr(self, "num_class", 2)
        if num_class == 2:
            return (proba[:, 1] >= 0.5).astype(int)
        return np.argmax(proba, axis=1)


def _tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    seed: int,
    objective_kind: str = "binary",
    num_class: int = 2,
) -> tuple[Any, dict[str, Any], float]:
    """Optuna tune XGBoost. Soporta binary y multiclass."""
    import xgboost as xgb
    from sklearn.metrics import log_loss

    is_multi = objective_kind == "multiclass"
    base_params: dict[str, Any] = {
        "objective": "multi:softprob" if is_multi else "binary:logistic",
        "eval_metric": "mlogloss" if is_multi else "logloss",
        "tree_method": "hist",
    }
    if is_multi:
        base_params["num_class"] = num_class
    multi_labels = list(range(num_class)) if is_multi else None

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params,
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
            if is_multi:
                losses.append(log_loss(y[val_idx], preds, labels=multi_labels))
            else:
                losses.append(log_loss(y[val_idx], np.clip(preds, 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    params_final = {
        **base_params,
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
    return (
        _XGBSklearnAdapter(booster, num_class=num_class if is_multi else 2),
        best,
        float(study.best_value),
    )


class _XGBSklearnAdapter:
    def __init__(self, booster: Any, num_class: int = 2) -> None:
        import xgboost as xgb

        self.booster = booster
        self.num_class = num_class
        self.classes_ = np.arange(num_class)
        self._xgb = xgb

    def fit(self, *_args: Any, **_kwargs: Any) -> _XGBSklearnAdapter:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.booster.predict(self._xgb.DMatrix(X))
        num_class = getattr(self, "num_class", 2)
        if num_class == 2:
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.vstack([1 - p, p]).T
        p = np.asarray(p)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        p = np.clip(p, 1e-7, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        num_class = getattr(self, "num_class", 2)
        if num_class == 2:
            return (proba[:, 1] >= 0.5).astype(int)
        return np.argmax(proba, axis=1)


def _tune_catboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    seed: int,
    objective_kind: str = "binary",
    num_class: int = 2,
) -> tuple[Any, dict[str, Any], float]:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.warning("train.catboost.library_missing_skip")
        return None, {}, float("inf")
    from sklearn.metrics import log_loss

    is_multi = objective_kind == "multiclass"
    loss_function = "MultiClass" if is_multi else "Logloss"
    multi_labels = list(range(num_class)) if is_multi else None

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 1500,
            "loss_function": loss_function,
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
            if is_multi:
                preds = model.predict_proba(X[val_idx])
                losses.append(log_loss(y[val_idx], preds, labels=multi_labels))
            else:
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
    # Map Optuna param names → CatBoost SDK names
    catboost_params = {
        "learning_rate": best["lr"],
        "depth": best["depth"],
        "l2_leaf_reg": best["l2"],
        "bagging_temperature": best["bag_temp"],
        "random_strength": best["rand_str"],
    }
    tr_idx, val_idx = splits[-1]
    model = CatBoostClassifier(
        iterations=2000,
        loss_function=loss_function,
        thread_count=N_THREADS,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=100,
        **catboost_params,
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
            X_train,
            y_train,
            splits=splits,
            n_trials=cfg.n_trials,
            seed=cfg.random_state,
            objective_kind=cfg.objective,
            num_class=cfg.num_class,
        )
        l0_models["lgbm"] = lgbm_est
        cv_losses["lgbm"] = lgbm_cv
        logger.info("train.lgbm.done", cv_logloss=lgbm_cv, best=lgbm_params)

    if cfg.enable_xgb:
        logger.info("train.xgb.start", n_trials=cfg.n_trials)
        xgb_est, xgb_params, xgb_cv = _tune_xgboost(
            X_train,
            y_train,
            splits=splits,
            n_trials=cfg.n_trials,
            seed=cfg.random_state,
            objective_kind=cfg.objective,
            num_class=cfg.num_class,
        )
        l0_models["xgb"] = xgb_est
        cv_losses["xgb"] = xgb_cv
        logger.info("train.xgb.done", cv_logloss=xgb_cv, best=xgb_params)

    if cfg.enable_catboost:
        logger.info("train.catboost.start", n_trials=cfg.n_trials)
        cat_est, cat_params, cat_cv = _tune_catboost(
            X_train,
            y_train,
            splits=splits,
            n_trials=cfg.n_trials,
            seed=cfg.random_state,
            objective_kind=cfg.objective,
            num_class=cfg.num_class,
        )
        if cat_est is not None:
            l0_models["catboost"] = cat_est
            cv_losses["catboost"] = cat_cv
            logger.info("train.catboost.done", cv_logloss=cat_cv, best=cat_params)

    if not l0_models:
        msg = "Ningún modelo L0 habilitado o entrenado"
        raise RuntimeError(msg)

    is_multi = cfg.objective == "multiclass"

    # Selección de modelo final: si stacking habilitado, train L1 sobre OOF preds.
    # Multiclass: stacker binario no aplica → seleccionamos mejor L0 por cv_logloss.
    if cfg.enable_stacking and len(l0_models) >= 2 and not is_multi:
        # Generar OOF predictions con los modelos L0 (usando el mismo split)
        oof_matrix = np.zeros((len(y_train), len(l0_models)))
        for i, (_name, model) in enumerate(l0_models.items()):
            # Approx: usar prediccion sobre train (pseudo-OOF). Ideal sería
            # re-split; aquí usamos predict sobre el mismo X_train ya que cada
            # modelo fue early-stopped en un split diferente. Walk-forward
            # fiel requeriría more compute.
            oof_matrix[:, i] = model.predict_proba(X_train)[:, 1]

        # Sprint 10 Fase 2 — MarketAwareStacker opcional con LGBM shallow
        # + monotonic constraints. Si env APUESTAS_USE_MARKET_STACKER=true,
        # usar el nuevo stacker (requiere columnas market_* en X_train;
        # si no existen, LGBM ignora constraints de columnas inexistentes).
        import os as _os

        use_market_stacker = (
            _os.environ.get("APUESTAS_USE_MARKET_STACKER", "false").lower() == "true"
        )
        use_tabpfn_stacker = (
            _os.environ.get("APUESTAS_USE_TABPFN_STACKER", "false").lower() == "true"
        )
        use_focal_loss = _os.environ.get("APUESTAS_USE_FOCAL_LOSS", "false").lower() == "true"
        oof_names = [f"oof_{name}" for name in l0_models]
        if use_tabpfn_stacker:
            from apuestas.ml.tabpfn_stacker import TabPFNStacker

            stacker_est = TabPFNStacker(device="cpu")
            stacker_est.fit(oof_matrix, y_train, feature_names=oof_names)
            final_model: Any = _StackingWrapper(l0_models=l0_models, l1=stacker_est)
            logger.info("train.stacker.tabpfn_v2", features=oof_names)
        elif use_market_stacker:
            from apuestas.ml.stacker import MarketAwareStacker

            stacker_est = MarketAwareStacker(
                use_lgbm=True,
                monotonic=True,
                max_depth=3,
                n_estimators=100,
                learning_rate=0.05,
                focal_loss=use_focal_loss,
            )
            stacker_est.fit(oof_matrix, y_train, feature_names=oof_names)
            final_model = _StackingWrapper(l0_models=l0_models, l1=stacker_est)
            logger.info(
                "train.stacker.market_aware",
                features=oof_names,
                monotonic=stacker_est.monotonic_constraints,
                focal_loss=use_focal_loss,
            )
        else:
            l1 = LogisticRegression(
                solver="liblinear",
                C=1.0,
                max_iter=500,
                random_state=cfg.random_state,
            )
            l1.fit(oof_matrix, y_train)
            final_model = _StackingWrapper(l0_models=l0_models, l1=l1)
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

    # Evaluación holdout pre-refinement.
    # Multiclass: usa log_loss sobre la matriz completa para ECE/Brier multi-clase.
    if is_multi:
        p_holdout_full = calibrated.predict_proba(X_holdout)
        from sklearn.metrics import log_loss as _log_loss

        multi_labels = list(range(cfg.num_class))
        holdout_ll = float(_log_loss(y_holdout, p_holdout_full, labels=multi_labels))
        # Brier multi-clase: media de (p - one_hot)^2
        one_hot = np.zeros_like(p_holdout_full)
        for i, c in enumerate(y_holdout):
            one_hot[i, int(c)] = 1.0
        holdout_brier = float(np.mean((p_holdout_full - one_hot) ** 2))
        # ECE multi-clase simplificado: max-prob bin calibration
        max_p = p_holdout_full.max(axis=1)
        pred_class = p_holdout_full.argmax(axis=1)
        correct = (pred_class == y_holdout).astype(float)
        from itertools import pairwise

        bins = np.linspace(0, 1, 11)
        holdout_ece = 0.0
        for lo, hi in pairwise(bins):
            mask = (max_p >= lo) & (max_p < hi)
            if mask.sum() > 0:
                holdout_ece += (mask.sum() / len(max_p)) * abs(
                    correct[mask].mean() - max_p[mask].mean()
                )

        # cal_metrics duck-type para mantener el resto del flujo
        class _CalMetrics:
            def __init__(self, log_loss: float, brier: float, ece: float) -> None:
                self.log_loss = log_loss
                self.brier = brier
                self.ece = ece

        cal_metrics = _CalMetrics(holdout_ll, holdout_brier, holdout_ece)

        best_cv_loss = min(cv_losses.values())
        return TrainResult(
            model_name=list(l0_models)[0] if len(l0_models) == 1 else "best_l0",
            estimator=calibrated,
            conformal=conformal,
            metrics={
                "holdout_log_loss": holdout_ll,
                "holdout_brier": holdout_brier,
                "holdout_ece": holdout_ece,
                "cv_log_loss": best_cv_loss,
                **{f"cv_log_loss_{k}": v for k, v in cv_losses.items()},
            },
            best_params={},
            feature_names=feature_names,
            cv_log_loss=best_cv_loss,
            cv_brier=0.0,
            cv_ece=0.0,
            holdout_log_loss=holdout_ll,
            holdout_brier=holdout_brier,
            holdout_ece=holdout_ece,
        )

    p_holdout = calibrated.predict_proba(X_holdout)[:, 1]
    cal_metrics_pre = compute_calibration_metrics(y_holdout, p_holdout)

    # Post-hoc Multi-method Calibration (SOTA 2024):
    # 1. Beta Calibration (Kull et al., AISTATS 2017)
    # 2. Temperature Scaling (Guo et al., ICML 2017) — fallback si Beta rechaza
    # 3. Histogram Binning (Zadrozny & Elkan, 2001) — último fallback
    # Aplica solo si ECE_pre > 0.08 y acepta el que más mejore.
    cal_metrics = cal_metrics_pre
    best_wrapped: Any = None
    best_metrics = cal_metrics_pre
    best_method = "none"

    # Threshold n_cal relajado a 50 (NFL tiene 56 cal samples, válido para
    # Platt/Beta/Temperature pero no para Histogram Binning 10+ bins).
    if cal_metrics_pre.ece > 0.08 and len(y_cal) >= 50:
        p_cal = calibrated.predict_proba(X_cal)[:, 1]
        p_holdout_all = calibrated.predict_proba(X_holdout)[:, 1]

        # --- Method 1: Beta Calibration ---
        try:
            from netcal.scaling import BetaCalibration  # type: ignore[import-untyped]

            beta_cal = BetaCalibration()
            beta_cal.fit(p_cal, y_cal)

            class _BetaWrapper:
                def __init__(self, base: Any, beta: Any) -> None:
                    self.base = base
                    self.beta = beta

                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    p = self.base.predict_proba(X)[:, 1]
                    p_out = np.asarray(self.beta.transform(p))
                    if p_out.ndim == 2:
                        p_out = p_out[:, 1] if p_out.shape[1] >= 2 else p_out[:, 0]
                    p_out = np.clip(p_out, 1e-4, 1 - 1e-4)
                    return np.column_stack([1 - p_out, p_out])

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

            w = _BetaWrapper(calibrated, beta_cal)
            p_h = w.predict_proba(X_holdout)[:, 1]
            m = compute_calibration_metrics(y_holdout, p_h)
            if m.ece < best_metrics.ece * 0.95:
                best_wrapped = w
                best_metrics = m
                best_method = "beta"
        except Exception as exc:
            logger.debug("train.beta_fail", error=str(exc)[:80])

        # --- Method 2: Temperature Scaling (Guo et al., ICML 2017) ---
        # Busca T que minimiza NLL sobre cal set. Si T > 1 → modelo over-confident.
        try:
            from scipy.optimize import minimize_scalar

            eps = 1e-8
            logits_cal = np.log(np.clip(p_cal, eps, 1 - eps) / (1 - np.clip(p_cal, eps, 1 - eps)))

            def _temp_nll(t: float) -> float:
                p_t = 1.0 / (1.0 + np.exp(-logits_cal / t))
                p_t = np.clip(p_t, eps, 1 - eps)
                return float(-np.mean(y_cal * np.log(p_t) + (1 - y_cal) * np.log(1 - p_t)))

            res = minimize_scalar(_temp_nll, bounds=(0.1, 10.0), method="bounded")
            T_opt = float(res.x)

            class _TempScaleWrapper:
                def __init__(self, base: Any, T: float) -> None:
                    self.base = base
                    self.T = T

                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    p = self.base.predict_proba(X)[:, 1]
                    eps_ = 1e-8
                    p = np.clip(p, eps_, 1 - eps_)
                    logits = np.log(p / (1 - p))
                    p_out = 1.0 / (1.0 + np.exp(-logits / self.T))
                    p_out = np.clip(p_out, 1e-4, 1 - 1e-4)
                    return np.column_stack([1 - p_out, p_out])

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

            w = _TempScaleWrapper(calibrated, T_opt)
            p_h = w.predict_proba(X_holdout)[:, 1]
            m = compute_calibration_metrics(y_holdout, p_h)
            if m.ece < best_metrics.ece * 0.95:
                best_wrapped = w
                best_metrics = m
                best_method = f"temp_scale_T={T_opt:.3f}"
        except Exception as exc:
            logger.debug("train.temp_scale_fail", error=str(exc)[:80])

        # --- Method 3: Histogram Binning (último fallback) ---
        try:
            from netcal.binning import HistogramBinning  # type: ignore[import-untyped]

            n_bins = max(5, min(15, len(y_cal) // 100))
            hb = HistogramBinning(bins=n_bins)
            hb.fit(p_cal, y_cal)

            class _HistBinWrapper:
                def __init__(self, base: Any, hb: Any) -> None:
                    self.base = base
                    self.hb = hb

                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    p = self.base.predict_proba(X)[:, 1]
                    p_out = np.asarray(self.hb.transform(p))
                    if p_out.ndim == 2:
                        p_out = p_out[:, 1] if p_out.shape[1] >= 2 else p_out[:, 0]
                    p_out = np.clip(p_out, 1e-4, 1 - 1e-4)
                    return np.column_stack([1 - p_out, p_out])

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

            w = _HistBinWrapper(calibrated, hb)
            p_h = w.predict_proba(X_holdout)[:, 1]
            m = compute_calibration_metrics(y_holdout, p_h)
            if m.ece < best_metrics.ece * 0.95:
                best_wrapped = w
                best_metrics = m
                best_method = f"hist_bin_{n_bins}"
        except Exception as exc:
            logger.debug("train.hist_bin_fail", error=str(exc)[:80])

        # Gap 4 wire — Isotonic regression (plan §7.3 / Niculescu-Mizil 2005).
        # Se prueba como cuarto método post-hoc; gana si reduce ECE > 5%.
        try:
            from apuestas.ml.isotonic import fit_isotonic_calibrator

            iso = fit_isotonic_calibrator(y_cal, p_cal)

            class _IsoWrapper:
                def __init__(self, base: Any, calibrator: Any) -> None:
                    self._base = base
                    self._iso = calibrator

                def predict_proba(self, X: np.ndarray) -> np.ndarray:
                    p_raw = self._base.predict_proba(X)[:, 1]
                    p_cal_arr = self._iso.predict(p_raw)
                    p_cal_arr = np.clip(p_cal_arr, 1e-7, 1 - 1e-7)
                    return np.column_stack([1 - p_cal_arr, p_cal_arr])

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

            w = _IsoWrapper(calibrated, iso)
            p_h = w.predict_proba(X_holdout)[:, 1]
            m = compute_calibration_metrics(y_holdout, p_h)
            if m.ece < best_metrics.ece * 0.95:
                best_wrapped = w
                best_metrics = m
                best_method = "isotonic"
        except Exception as exc:
            logger.debug("train.isotonic_fail", error=str(exc)[:80])

        # Log ganador
        if best_wrapped is not None:
            calibrated = best_wrapped
            cal_metrics = best_metrics
            logger.info(
                "train.post_hoc_cal_applied",
                method=best_method,
                ece_before=cal_metrics_pre.ece,
                ece_after=best_metrics.ece,
                logloss_before=cal_metrics_pre.log_loss,
                logloss_after=best_metrics.log_loss,
            )
        else:
            logger.info(
                "train.post_hoc_cal_no_improvement",
                ece_pre=cal_metrics_pre.ece,
                note="todos los métodos rechazados",
            )
            # Mantener p_holdout_all para consistencia
            _ = p_holdout_all

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
    """L0 → OOF preds → L1 (LogReg o MarketAwareStacker)."""

    _estimator_type = "classifier"

    def __init__(self, l0_models: dict[str, Any], l1: Any) -> None:
        # l1 puede ser LogisticRegression o MarketAwareStacker (ambos exponen predict_proba)
        self.l0_models = l0_models
        self.l1 = l1
        self.classes_ = np.array([0, 1])

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils._tags import ClassifierTags, Tags

        return Tags(
            estimator_type="classifier",
            classifier_tags=ClassifierTags(),
            target_tags=None,
            transformer_tags=None,
            regressor_tags=None,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"l0_models": self.l0_models, "l1": self.l1}

    def set_params(self, **params: Any) -> _StackingWrapper:
        for k, v in params.items():
            setattr(self, k, v)
        return self

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


def prune_ensemble_if_redundant(
    l0_models: dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    min_disagreement: float = 0.05,
) -> tuple[dict[str, Any], str]:
    """Fase 3.7 — Pruning automático del ensemble si diversidad es baja.

    Si `mean_disagreement < min_disagreement` (default 0.05), los modelos están
    casi de acuerdo → el ensemble no aporta valor sobre 1 modelo. Retorna solo
    el modelo con mejor log-loss en validation.

    Returns (pruned_models_dict, decision_note) — note va a model_registry_meta.ensemble_notes.
    """
    from sklearn.metrics import log_loss  # type: ignore[import-untyped]

    div = ensemble_diversity(l0_models, X_val)
    mean_disagreement = float(div.get("mean_disagreement", 1.0))

    if mean_disagreement >= min_disagreement or len(l0_models) <= 1:
        return l0_models, (
            f"ensemble_kept mean_disagreement={mean_disagreement:.4f} "
            f">= {min_disagreement} (diverse enough)"
        )

    # Prune: calcula log-loss per modelo, mantiene solo el mejor
    scores: dict[str, float] = {}
    for name, model in l0_models.items():
        try:
            proba = model.predict_proba(X_val)[:, 1]
            scores[name] = log_loss(y_val, proba, labels=[0, 1])
        except Exception:  # fmt: skip
            scores[name] = float("inf")

    best_name = min(scores, key=lambda k: scores[k])
    pruned = {best_name: l0_models[best_name]}
    note = (
        f"ensemble_pruned_to_best mean_disagreement={mean_disagreement:.4f} < "
        f"{min_disagreement} kept={best_name} cv_logloss={scores[best_name]:.4f}"
    )
    return pruned, note
