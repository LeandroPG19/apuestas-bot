"""Pipeline entrenamiento NHL end-to-end.

NHL usa feature set simplificado (goals_for/against rolling + rest days) porque
xG/Corsi granular aún no está poblado en histórico. Estructura análoga a
train_nfl.py. Target por default: moneyline (win probability).
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
from apuestas.features.common import (
    back_to_back_flag,
    compute_target,
    days_since_last,
    rolling_mean_prev,
)
from apuestas.ml.train_base import TrainConfig, TrainResult, train_ensemble
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

FEATURE_SET_NAME = "nhl_basic_v1"
WINDOWS = [5, 10, 20]
Target = Literal["moneyline", "total"]


@dataclass(slots=True)
class NHLTrainConfig:
    seasons: list[int] | list[str]
    target: Target = "moneyline"
    n_trials: int = 30
    split_train_pct: float = 0.80
    split_cal_pct: float = 0.10
    random_state: int = 42
    stage: Literal["shadow", "production"] = "shadow"
    experiment_name: str = "nhl_moneyline"


async def load_nhl_training_data(
    seasons: list[int] | list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, home_team_id, away_team_id, start_time,
                       home_score, away_score, status, season
                FROM matches
                WHERE sport_code = 'nhl'
                  AND season = ANY(:seasons)
                  AND status = 'finished'
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY start_time
                """
            ),
            {"seasons": [str(s) for s in seasons]},
        )
        matches_rows = [dict(r._mapping) for r in result.all()]

    if not matches_rows:
        return pl.DataFrame(), pl.DataFrame()

    team_rows: list[dict[str, object]] = []
    for r in matches_rows:
        hg = int(r["home_score"])
        ag = int(r["away_score"])
        team_rows.append(
            {
                "team_id": r["home_team_id"],
                "game_date": r["start_time"],
                "goals_for": hg,
                "goals_against": ag,
                "win": 1 if hg > ag else 0,
                "margin": hg - ag,
            }
        )
        team_rows.append(
            {
                "team_id": r["away_team_id"],
                "game_date": r["start_time"],
                "goals_for": ag,
                "goals_against": hg,
                "win": 1 if ag > hg else 0,
                "margin": ag - hg,
            }
        )
    return pl.DataFrame(matches_rows), pl.DataFrame(team_rows)


def _team_rolling_basic(team_games: pl.DataFrame) -> pl.DataFrame:
    result = team_games.sort(["team_id", "game_date"])
    for metric in ("goals_for", "goals_against", "win", "margin"):
        result = rolling_mean_prev(
            result,
            by="team_id",
            order="game_date",
            value=metric,
            windows=WINDOWS,
        )
    result = days_since_last(result, by="team_id", order="game_date", name="rest_days")
    result = back_to_back_flag(result, by="team_id", order="game_date", threshold_hours=36.0)
    return result


def build_nhl_feature_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
) -> pl.DataFrame:
    feats = _team_rolling_basic(team_games)
    base_cols = [c for c in feats.columns if any(c.endswith(f"_roll_{w}") for w in WINDOWS)]
    base_cols += ["rest_days", "back_to_back"]
    base_cols = [c for c in base_cols if c in feats.columns]

    home = feats.select(
        pl.col("team_id").alias("home_team_id"),
        pl.col("game_date").alias("start_time"),
        *[pl.col(c).alias(f"{c}_home") for c in base_cols],
    )
    away = feats.select(
        pl.col("team_id").alias("away_team_id"),
        pl.col("game_date").alias("start_time"),
        *[pl.col(c).alias(f"{c}_away") for c in base_cols],
    )
    merged = matches.join(home, on=["home_team_id", "start_time"], how="left")
    merged = merged.join(away, on=["away_team_id", "start_time"], how="left")

    for m in (
        "goals_for_roll_10",
        "goals_against_roll_10",
        "margin_roll_10",
        "win_roll_10",
    ):
        h = f"{m}_home"
        a = f"{m}_away"
        if h in merged.columns and a in merged.columns:
            merged = merged.with_columns((pl.col(h) - pl.col(a)).alias(f"{m}_diff"))

    return merged


async def train_nhl(cfg: NHLTrainConfig | None = None) -> TrainResult:
    cfg = cfg or NHLTrainConfig(seasons=["2022-23", "2023-24", "2024-25"])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches, team_games = await load_nhl_training_data(cfg.seasons)
    if matches.height == 0:
        msg = "Sin datos NHL"
        raise RuntimeError(msg)

    features_df = build_nhl_feature_frame(matches, team_games)
    features_df = compute_target(features_df, kind="win")
    feat_cols = [
        c
        for c in features_df.columns
        if c.endswith(("_home", "_away", "_diff"))
        and features_df.schema[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    features_df = features_df.drop_nulls(subset=["y"])

    n = features_df.height
    if n < 200:
        msg = f"Sample NHL insuficiente ({n} matches tras feature build)"
        raise RuntimeError(msg)

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
        run_name=f"nhl_{cfg.target}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
    ):
        mlflow.log_params(
            {
                "sport": "nhl",
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

        model_path = Path("/tmp") / "nhl_calibrated.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump(
                {
                    "estimator": result.estimator,
                    "conformal": result.conformal,
                    "feature_names": feat_cols,
                    "target": cfg.target,
                    "sport": "nhl",
                },
                f,
            )
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        from apuestas.ml.registry_helper import register_model_in_db

        await register_model_in_db(
            mlflow_run_id=run_id,
            model_name=cfg.experiment_name,
            sport_code="nhl",
            stage=cfg.stage,
            metrics=result.metrics,
        )

    logger.info("nhl.train.done", target=cfg.target, log_loss=result.holdout_log_loss)
    return result
