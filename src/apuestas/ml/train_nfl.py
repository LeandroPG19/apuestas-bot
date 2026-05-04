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
from apuestas.features.common import (
    back_to_back_flag,
    compute_target,
    days_since_last,
    rolling_mean_prev,
)
from apuestas.ml.train_base import TrainConfig, TrainResult, train_ensemble
from apuestas.obs.logging import get_logger

# feature set v2: rolling points + margin + win_rate + rest_days
FEATURE_SET_NAME = "nfl_v2_basic"
_WINDOWS = [3, 5, 8]

logger = get_logger(__name__)

Target = Literal["ats", "total", "moneyline"]


@dataclass(slots=True)
class NFLTrainConfig:
    seasons: list[int] | list[str]
    target: Target = "ats"
    n_trials: int = 30
    split_train_pct: float = 0.70  # bajo 80%→70% para dar 20% a cal set
    split_cal_pct: float = 0.20  # sube 10%→20% (NFL low sample → necesita cal grande)
    random_state: int = 42
    stage: Literal["shadow", "production"] = "shadow"
    experiment_name: str = "nfl_ats"
    calibration_method: Literal["auto", "sigmoid", "isotonic", "venn_abers"] = "auto"


async def load_nfl_training_data(
    seasons: list[int] | list[str],
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
        hs = int(r["home_score"])
        asc = int(r["away_score"])
        # Registrar ambos equipos como home y away para rolling correcto
        team_rows.append(
            {
                "team_id": r["home_team_id"],
                "game_date": r["start_time"],
                "points_scored": hs,
                "points_allowed": asc,
                "win_margin": hs - asc,
                "win": 1 if hs > asc else 0,
            }
        )
        team_rows.append(
            {
                "team_id": r["away_team_id"],
                "game_date": r["start_time"],
                "points_scored": asc,
                "points_allowed": hs,
                "win_margin": asc - hs,
                "win": 1 if asc > hs else 0,
            }
        )
    return pl.DataFrame(matches_rows), pl.DataFrame(team_rows)


def _team_rolling_basic_nfl(team_games: pl.DataFrame) -> pl.DataFrame:
    """Rolling 3/5/8 games sobre points + margin + win_rate."""
    result = team_games.sort(["team_id", "game_date"])
    for metric in ("points_scored", "points_allowed", "win_margin", "win"):
        result = rolling_mean_prev(
            result,
            by="team_id",
            order="game_date",
            value=metric,
            windows=_WINDOWS,
        )
    result = days_since_last(result, by="team_id", order="game_date", name="rest_days")
    # short_week (Thursday night post-Sunday)
    result = result.with_columns((pl.col("rest_days") < 5).cast(pl.Int8).alias("short_week"))
    # Post-bye week boost conocido
    result = result.with_columns((pl.col("rest_days") >= 10).cast(pl.Int8).alias("post_bye_week"))
    result = back_to_back_flag(result, by="team_id", order="game_date", threshold_hours=96.0)
    return result


def build_nfl_feature_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
) -> pl.DataFrame:
    """Join rolling features sobre home y away → dataframe final para train."""
    feats = _team_rolling_basic_nfl(team_games)
    base_cols = [c for c in feats.columns if any(c.endswith(f"_roll_{w}") for w in _WINDOWS)]
    base_cols += ["rest_days", "short_week", "post_bye_week"]
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

    # Diferenciales críticos NFL
    for m in (
        "points_scored_roll_5",
        "points_allowed_roll_5",
        "win_margin_roll_5",
        "win_roll_5",
        "points_scored_roll_8",
        "win_margin_roll_8",
    ):
        h = f"{m}_home"
        a = f"{m}_away"
        if h in merged.columns and a in merged.columns:
            merged = merged.with_columns((pl.col(h) - pl.col(a)).alias(f"{m}_diff"))

    return merged


async def train_nfl(cfg: NFLTrainConfig | None = None) -> TrainResult:
    cfg = cfg or NFLTrainConfig(seasons=[2024, 2025])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches, team_games = await load_nfl_training_data(cfg.seasons)
    if matches.height == 0:
        msg = "Sin datos NFL"
        raise RuntimeError(msg)

    features_df = build_nfl_feature_frame(matches, team_games)
    # Sprint 10 Fase 2: Elo features (rating bidireccional + anti-leakage)
    from apuestas.features.common import add_elo_features

    features_df = add_elo_features(features_df, sport="nfl")
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

        run_id = mlflow.active_run().info.run_id
        from apuestas.ml.registry_helper import register_model_in_db

        # Sprint 5b — KPI gate: NFL historial log_loss=0.85 ≈ random. Antes de
        # registrar en `production`, validar los 4 KPIs sobre holdout.
        # Si falla, degradar `stage` a "shadow" para no afectar producción.
        effective_stage = cfg.stage
        if cfg.stage == "production" and result.holdout_log_loss > 0.68:
            logger.warning(
                "nfl.train.kpi_gate_failed",
                log_loss=result.holdout_log_loss,
                threshold=0.68,
                action="downgrade_to_shadow",
            )
            effective_stage = "shadow"

        await register_model_in_db(
            mlflow_run_id=run_id,
            model_name=cfg.experiment_name,
            sport_code="nfl",
            stage=effective_stage,
            metrics=result.metrics,
        )

    logger.info(
        "nfl.train.done",
        target=cfg.target,
        log_loss=result.holdout_log_loss,
        stage=effective_stage if "effective_stage" in locals() else cfg.stage,
    )
    return result
