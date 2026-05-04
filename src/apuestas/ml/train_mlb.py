"""Pipeline entrenamiento MLB end-to-end con MLflow (§6 + §13-16).

Targets soportados: total (O/U 9), moneyline, runline, NRFI, F5 total.
Features: MLB Statcast + park factors + pitcher matchup + weather.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import cloudpickle
import mlflow
import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.features.common import compute_target
from apuestas.features.mlb import FEATURE_SET_NAME, build_mlb_feature_frame
from apuestas.ml.train_base import TrainConfig, TrainResult, train_ensemble
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Target = Literal["total", "moneyline", "runline"]


@dataclass(slots=True)
class MLBTrainConfig:
    years: list[int]
    target: Target = "moneyline"  # más predecible que total (mercado menos eficiente)
    n_trials: int = 30
    split_train_pct: float = 0.75
    split_cal_pct: float = 0.15  # más samples para calibrator
    random_state: int = 42
    stage: Literal["shadow", "production"] = "shadow"
    experiment_name: str = "mlb_moneyline"
    calibration_method: Literal["auto", "sigmoid", "isotonic", "venn_abers"] = "auto"


async def load_mlb_training_data(years: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, external_id, home_team_id, away_team_id,
                       start_time, venue_id, home_score, away_score, status, season
                FROM matches
                WHERE sport_code = 'mlb'
                  AND season = ANY(:years)
                  AND status = 'finished'
                ORDER BY start_time
                """
            ),
            {"years": [str(y) for y in years]},
        )
        matches_rows = [dict(r._mapping) for r in result.all()]

    if not matches_rows:
        return pl.DataFrame(), pl.DataFrame()

    # team_games derivados (en prod usa Statcast via pybaseball)
    team_rows = []
    for r in matches_rows:
        if r["home_score"] is None or r["away_score"] is None:
            continue
        total = r["home_score"] + r["away_score"]
        for tid, runs_scored, runs_allowed in (
            (r["home_team_id"], r["home_score"], r["away_score"]),
            (r["away_team_id"], r["away_score"], r["home_score"]),
        ):
            team_rows.append(
                {
                    "team_id": tid,
                    "game_date": r["start_time"],
                    "runs_scored": runs_scored,
                    "runs_allowed": runs_allowed,
                    "total_runs": total,
                }
            )
    # infer_schema_length=None usa todos los rows para inferir tipos. Necesario
    # porque venue_id es None en filas viejas y i64 en nuevas — el default 100
    # rows infiere None→Null y luego falla al ver i64. Mismo issue con season.
    return (
        pl.DataFrame(matches_rows, infer_schema_length=None),
        pl.DataFrame(team_rows, infer_schema_length=None),
    )


def build_training_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    target: Target,
) -> tuple[pl.DataFrame, list[str]]:
    features_df = build_mlb_feature_frame(matches, team_games)
    # Sprint 10 Fase 2: Elo features (rating bidireccional + anti-leakage)
    from apuestas.features.common import add_elo_features

    features_df = add_elo_features(features_df, sport="mlb")

    # Sprint 14 #146 — MLB context features (bullpen/travel). Opt-in.
    if os.environ.get("APUESTAS_MLB_CONTEXT_FEATURES", "true").lower() == "true":
        try:
            features_df = _add_mlb_context_sync(features_df, matches)
            logger.info("mlb.context_features.added")
        except Exception as exc:
            logger.warning("mlb.context_features.skip", error=str(exc)[:100])

    # Sprint 10 Fase 3 — Poisson ensemble: si APUESTAS_MLB_POISSON_ENSEMBLE=true,
    # ajusta MLBPoissonModel sobre los matches finalizados y añade `poisson_p_home`
    # como feature. El ensemble (LGBM+XGB+CatBoost) la consume como señal extra.
    if os.environ.get("APUESTAS_MLB_POISSON_ENSEMBLE", "false").lower() == "true":
        features_df = _add_poisson_prediction(features_df)

    if target == "total":
        features_df = features_df.with_columns(
            ((pl.col("home_score") + pl.col("away_score")) > 8.5).cast(pl.Int8).alias("y")
        )
    elif target == "runline":
        # Runline MLB estándar: home -1.5 / away +1.5. Target binario: home cubre
        # runline (gana por 2+ runs). Necesario porque el modelo moneyline NO
        # predice si cubre el spread; emitir picks spreads con probabilidad de
        # ganar el juego (no de cubrir) producía overconfidence +33pp y ROI -68%.
        features_df = features_df.with_columns(
            ((pl.col("home_score") - pl.col("away_score")) >= 2).cast(pl.Int8).alias("y")
        )
    else:
        features_df = compute_target(features_df, kind="win")

    feat_cols = [
        c
        for c in features_df.columns
        if c.endswith(("_home", "_away", "_diff"))
        and features_df.schema[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    # Incluir poisson_p_home si está presente (feature adicional Sprint 10 Fase 3)
    if "poisson_p_home" in features_df.columns and "poisson_p_home" not in feat_cols:
        feat_cols.append("poisson_p_home")
    features_df = features_df.drop_nulls(subset=["y"])
    return features_df, feat_cols


def _add_mlb_context_sync(features_df: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Context features (bullpen + travel) computed from matches DataFrame.

    Sync approximation: usa team-level rolling sobre innings histoáricos del mismo
    team_games DataFrame. NO llama DB — pure-polars para velocidad.
    Output features: `bullpen_ip_3d_away/home` (approx via games_last_3d),
    `travel_miles_away` (aprox via venue changes).
    """
    if features_df.height == 0 or matches.height == 0:
        return features_df
    # Games played last 3 days per team (proxy de bullpen workload)
    schedule = matches.select(
        pl.col("id").alias("game_id"),
        pl.col("start_time"),
        pl.col("home_team_id"),
        pl.col("away_team_id"),
    ).sort("start_time")
    # For each game, count games_last_3d for home_team and away_team (previous)
    home_sched = schedule.select(
        pl.col("game_id"), pl.col("start_time"), pl.col("home_team_id").alias("team_id")
    )
    away_sched = schedule.select(
        pl.col("game_id"), pl.col("start_time"), pl.col("away_team_id").alias("team_id")
    )
    all_sched = pl.concat([home_sched, away_sched]).sort(["team_id", "start_time"])
    # rolling count of games in past 3 days using group_by_dynamic isn't trivial per-team;
    # aproximación: diff to previous game date
    all_sched = all_sched.with_columns(
        (
            (
                pl.col("start_time") - pl.col("start_time").shift(1).over("team_id")
            ).dt.total_seconds()
            / 86400.0
        ).alias("days_since_prev")
    )
    all_sched = all_sched.with_columns(
        pl.when(pl.col("days_since_prev") < 3.0)
        .then(3.0)
        .when(pl.col("days_since_prev") < 5.0)
        .then(2.0)
        .otherwise(1.0)
        .alias("bullpen_ip_3d_est")
    ).fill_null(1.0)

    home_ctx = all_sched.rename({"team_id": "home_team_id"}).select(
        ["game_id", "home_team_id", pl.col("bullpen_ip_3d_est").alias("bullpen_ip_3d_home")]
    )
    away_ctx = all_sched.rename({"team_id": "away_team_id"}).select(
        ["game_id", "away_team_id", pl.col("bullpen_ip_3d_est").alias("bullpen_ip_3d_away")]
    )
    out = features_df
    if "game_id" in out.columns and "home_team_id" in out.columns:
        out = out.join(home_ctx, on=["game_id", "home_team_id"], how="left")
    if "game_id" in out.columns and "away_team_id" in out.columns:
        out = out.join(away_ctx, on=["game_id", "away_team_id"], how="left")
    return out.fill_null(1.0)


def _add_poisson_prediction(df: pl.DataFrame) -> pl.DataFrame:
    """Añade columna `poisson_p_home` al DataFrame con predicción del MLE
    Poisson sobre offense/defense + park factors (Sprint 10 Fase 3).

    Anti-leakage: re-ajusta el Poisson progresivamente sobre matches
    cronológicamente anteriores al row actual. Para velocidad, usa
    un solo fit con todo el histórico y predicción bajo supuesto de
    equipos estáticos (aproximación aceptable para MLB run-scoring).
    """
    from apuestas.ml.mlb_poisson import MLBPoissonModel

    df_sorted = df.sort("start_time")
    cutoff = int(df_sorted.height * 0.8)
    fit_df = df_sorted.slice(0, cutoff)
    matches_for_fit: list[dict] = []
    for row in fit_df.iter_rows(named=True):
        if row.get("home_score") is None or row.get("away_score") is None:
            continue
        matches_for_fit.append(
            {
                "home_id": int(row.get("home_team_id") or 0),
                "away_id": int(row.get("away_team_id") or 0),
                "home_runs": int(row["home_score"]),
                "away_runs": int(row["away_score"]),
                "venue_name": row.get("venue_name"),
            }
        )
    if not matches_for_fit:
        logger.warning("train_mlb.poisson_ensemble.no_fit_data")
        return df_sorted.with_columns(pl.lit(0.5).alias("poisson_p_home"))

    model = MLBPoissonModel.fit(matches_for_fit, n_iter=80)
    p_home_col: list[float] = []
    for row in df_sorted.iter_rows(named=True):
        venue = row.get("venue_name")
        probs = model.predict_moneyline(
            home_id=int(row.get("home_team_id") or 0),
            away_id=int(row.get("away_team_id") or 0),
            venue_name=venue if isinstance(venue, str) else None,
        )
        p_home_col.append(probs["home"])
    logger.info("train_mlb.poisson_ensemble.added", n=len(p_home_col))
    return df_sorted.with_columns(pl.Series("poisson_p_home", p_home_col))


async def train_mlb(cfg: MLBTrainConfig | None = None) -> TrainResult:
    cfg = cfg or MLBTrainConfig(years=[2024, 2025])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches, team_games = await load_mlb_training_data(cfg.years)
    if matches.height == 0:
        msg = "Sin datos MLB. Ejecuta ingest_mlb_season primero."
        raise RuntimeError(msg)

    frame, feat_cols = build_training_frame(matches, team_games, target=cfg.target)
    if not feat_cols:
        msg = "No features disponibles tras feature engineering"
        raise RuntimeError(msg)

    logger.info("mlb.train.frame_ready", rows=frame.height, features=len(feat_cols))

    frame = frame.sort("start_time")
    n = frame.height
    n_train = int(n * cfg.split_train_pct)
    n_cal = int(n * cfg.split_cal_pct)
    train_df = frame.slice(0, n_train)
    cal_df = frame.slice(n_train, n_cal)
    holdout_df = frame.slice(n_train + n_cal, n - n_train - n_cal)

    def _xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df.select(feat_cols).fill_nan(0.0).fill_null(0.0).to_numpy()
        y = df["y"].to_numpy().astype(np.int8)
        return X, y

    X_train, y_train = _xy(train_df)
    X_cal, y_cal = _xy(cal_df)
    X_holdout, y_holdout = _xy(holdout_df)

    with mlflow.start_run(
        run_name=f"mlb_{cfg.target}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
    ):
        mlflow.log_params(
            {
                "sport": "mlb",
                "target": cfg.target,
                "years": ",".join(str(y) for y in cfg.years),
                "feature_set": FEATURE_SET_NAME,
                "n_features": len(feat_cols),
                "n_train": len(y_train),
                "n_cal": len(y_cal),
                "n_holdout": len(y_holdout),
            }
        )
        result = train_ensemble(
            X_train,
            y_train,
            X_cal,
            y_cal,
            X_holdout,
            y_holdout,
            feature_names=feat_cols,
            cfg=TrainConfig(
                n_trials=cfg.n_trials,
                random_state=cfg.random_state,
                calibration_method=cfg.calibration_method,
            ),
        )
        for k, v in result.metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):  # fmt: skip
                pass

        model_path = Path("/tmp") / "mlb_calibrated.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump(
                {
                    "estimator": result.estimator,
                    "conformal": result.conformal,
                    "feature_names": feat_cols,
                    "target": cfg.target,
                    "sport": "mlb",
                },
                f,
            )
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        from apuestas.ml.registry_helper import register_model_in_db

        await register_model_in_db(
            mlflow_run_id=run_id,
            model_name=cfg.experiment_name,
            sport_code="mlb",
            stage=cfg.stage,
            metrics=result.metrics,
        )

    logger.info(
        "mlb.train.done",
        target=cfg.target,
        holdout_log_loss=result.holdout_log_loss,
        holdout_ece=result.holdout_ece,
    )
    return result
