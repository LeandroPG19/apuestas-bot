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
    target: Target = "total"
    n_trials: int = 30
    split_train_pct: float = 0.80
    split_cal_pct: float = 0.10
    random_state: int = 42
    stage: Literal["shadow", "production"] = "shadow"
    experiment_name: str = "mlb_total"


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
    return pl.DataFrame(matches_rows), pl.DataFrame(team_rows)


def build_training_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    target: Target,
) -> tuple[pl.DataFrame, list[str]]:
    features_df = build_mlb_feature_frame(matches, team_games)
    if target == "total":
        features_df = features_df.with_columns(
            ((pl.col("home_score") + pl.col("away_score")) > 8.5).cast(pl.Int8).alias("y")
        )
    else:
        features_df = compute_target(features_df, kind="win")

    feat_cols = [
        c
        for c in features_df.columns
        if c.endswith(("_home", "_away", "_diff"))
        and features_df.schema[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    features_df = features_df.drop_nulls(subset=["y"])
    return features_df, feat_cols


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
            cfg=TrainConfig(n_trials=cfg.n_trials, random_state=cfg.random_state),
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

    logger.info(
        "mlb.train.done",
        target=cfg.target,
        holdout_log_loss=result.holdout_log_loss,
        holdout_ece=result.holdout_ece,
    )
    return result
