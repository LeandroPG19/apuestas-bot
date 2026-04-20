"""Pipeline entrenamiento NFL end-to-end (§6 + §13-16).

Targets: ats (cover spread), total, moneyline. NFL low-sample (272 games/año)
requiere TimeSeriesSplit agresivo + regularización alta.
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
from apuestas.features.nfl import FEATURE_SET_NAME, build_nfl_feature_frame
from apuestas.ml.train_base import TrainConfig, TrainResult, train_ensemble
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Target = Literal["ats", "total", "moneyline"]


@dataclass(slots=True)
class NFLTrainConfig:
    seasons: list[int]
    target: Target = "ats"
    n_trials: int = 30
    split_train_pct: float = 0.80
    split_cal_pct: float = 0.10
    random_state: int = 42
    stage: Literal["shadow", "production"] = "shadow"
    experiment_name: str = "nfl_ats"


async def load_nfl_training_data(
    seasons: list[int],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, home_team_id, away_team_id, start_time,
                       home_score, away_score, status, season
                FROM matches
                WHERE sport_code = 'nfl'
                  AND season = ANY(:seasons)
                  AND status = 'finished'
                ORDER BY start_time
                """
            ),
            {"seasons": [str(s) for s in seasons]},
        )
        matches_rows = [dict(r._mapping) for r in result.all()]

    if not matches_rows:
        return pl.DataFrame(), pl.DataFrame()

    team_rows = []
    for r in matches_rows:
        if r["home_score"] is None or r["away_score"] is None:
            continue
        margin = r["home_score"] - r["away_score"]
        for tid, win in (
            (r["home_team_id"], margin),
            (r["away_team_id"], -margin),
        ):
            team_rows.append(
                {
                    "team_id": tid,
                    "game_date": r["start_time"],
                    "win_margin": win,
                    "points_scored": r["home_score"]
                    if tid == r["home_team_id"]
                    else r["away_score"],
                    "points_allowed": r["away_score"]
                    if tid == r["home_team_id"]
                    else r["home_score"],
                }
            )
    return pl.DataFrame(matches_rows), pl.DataFrame(team_rows)


async def train_nfl(cfg: NFLTrainConfig | None = None) -> TrainResult:
    cfg = cfg or NFLTrainConfig(seasons=[2024, 2025])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches, team_games = await load_nfl_training_data(cfg.seasons)
    if matches.height == 0:
        msg = "Sin datos NFL"
        raise RuntimeError(msg)

    features_df = build_nfl_feature_frame(matches, team_games)
    features_df = compute_target(features_df, kind="win")
    feat_cols = [
        c
        for c in features_df.columns
        if c.endswith(("_home", "_away", "_diff"))
        and features_df.schema[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    features_df = features_df.drop_nulls(subset=["y"])

    n = features_df.height
    n_train = int(n * cfg.split_train_pct)
    n_cal = int(n * cfg.split_cal_pct)

    def _xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df.select(feat_cols).fill_nan(0.0).fill_null(0.0).to_numpy()
        y = df["y"].to_numpy().astype(np.int8)
        return X, y

    X_train, y_train = _xy(features_df.slice(0, n_train))
    X_cal, y_cal = _xy(features_df.slice(n_train, n_cal))
    X_holdout, y_holdout = _xy(features_df.slice(n_train + n_cal, n - n_train - n_cal))

    with mlflow.start_run(
        run_name=f"nfl_{cfg.target}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
    ):
        mlflow.log_params(
            {
                "sport": "nfl",
                "target": cfg.target,
                "seasons": ",".join(str(s) for s in cfg.seasons),
                "feature_set": FEATURE_SET_NAME,
                "n_features": len(feat_cols),
                "n_train": len(y_train),
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

        model_path = Path("/tmp") / "nfl_calibrated.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump(
                {
                    "estimator": result.estimator,
                    "conformal": result.conformal,
                    "feature_names": feat_cols,
                    "target": cfg.target,
                    "sport": "nfl",
                },
                f,
            )
        mlflow.log_artifact(str(model_path), artifact_path="model")

    logger.info("nfl.train.done", target=cfg.target, log_loss=result.holdout_log_loss)
    return result
