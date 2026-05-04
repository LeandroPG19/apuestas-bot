"""Pipeline entrenamiento Tennis end-to-end.

Tennis se modela como match 1-vs-1 (team_id identifica al player en DB).
Feature set inicial: Elo rating + rolling win rate + rest days + surface.
Target: moneyline (prob que el "home"/player 1 gane).

El resultado del modelo complementa el Markov chain point-by-point
(features.tennis.match_probability_bo3/bo5) que aplica sobre p_hold derivado.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cloudpickle
import mlflow
import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.features.common import compute_target, days_since_last, rolling_mean_prev
from apuestas.ml.train_base import TrainConfig, TrainResult, train_ensemble
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

FEATURE_SET_NAME = "tennis_v1"
WINDOWS = [5, 10, 20]
_INITIAL_ELO = 1500.0
_ELO_K = 32.0


@dataclass(slots=True)
class TennisTrainConfig:
    seasons: list[str]
    n_trials: int = 30
    split_train_pct: float = 0.80
    split_cal_pct: float = 0.10
    random_state: int = 42
    stage: Literal["shadow", "production"] = "shadow"
    experiment_name: str = "tennis_moneyline"


async def load_tennis_training_data(
    seasons: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, home_team_id, away_team_id, start_time,
                       home_score, away_score, status, season,
                       COALESCE(metadata->>'surface', 'unknown') AS surface
                FROM matches
                WHERE sport_code = 'tennis'
                  AND season = ANY(:seasons)
                  AND status = 'finished'
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY start_time
                """
            ),
            {"seasons": list(seasons)},
        )
        rows = [dict(r._mapping) for r in result.all()]
    if not rows:
        return pl.DataFrame(), pl.DataFrame()

    player_rows: list[dict[str, Any]] = []
    for r in rows:
        hs = int(r["home_score"])
        as_ = int(r["away_score"])
        player_rows.append(
            {
                "team_id": r["home_team_id"],
                "game_date": r["start_time"],
                "win": 1 if hs > as_ else 0,
            }
        )
        player_rows.append(
            {
                "team_id": r["away_team_id"],
                "game_date": r["start_time"],
                "win": 1 if as_ > hs else 0,
            }
        )
    return pl.DataFrame(rows), pl.DataFrame(player_rows)


def _compute_elo_history(matches: pl.DataFrame) -> dict[tuple[int, Any], float]:
    """Recorre matches en orden y devuelve Elo pre-match por (team_id, match_id)."""
    elo: dict[int, float] = defaultdict(lambda: _INITIAL_ELO)
    history: dict[tuple[int, Any], float] = {}
    for row in matches.sort("start_time").iter_rows(named=True):
        mid = row["id"]
        h = int(row["home_team_id"])
        a = int(row["away_team_id"])
        history[(h, mid)] = elo[h]
        history[(a, mid)] = elo[a]
        # Update post-match
        exp_h = 1.0 / (1.0 + 10.0 ** ((elo[a] - elo[h]) / 400.0))
        outcome_h = 1.0 if int(row["home_score"]) > int(row["away_score"]) else 0.0
        elo[h] = elo[h] + _ELO_K * (outcome_h - exp_h)
        elo[a] = elo[a] + _ELO_K * ((1.0 - outcome_h) - (1.0 - exp_h))
    return history


def _rolling_player_stats(player_games: pl.DataFrame) -> pl.DataFrame:
    result = player_games.sort(["team_id", "game_date"])
    for metric in ("win",):
        result = rolling_mean_prev(
            result,
            by="team_id",
            order="game_date",
            value=metric,
            windows=WINDOWS,
        )
    result = days_since_last(result, by="team_id", order="game_date", name="rest_days")
    return result


def build_tennis_feature_frame(
    matches: pl.DataFrame,
    player_games: pl.DataFrame,
) -> pl.DataFrame:
    elo_history = _compute_elo_history(matches)
    feats = _rolling_player_stats(player_games)
    base_cols = [c for c in feats.columns if any(c.endswith(f"_roll_{w}") for w in WINDOWS)]
    base_cols += ["rest_days"]
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

    # Add Elo pre-match per row
    elo_home = [
        elo_history.get((int(row["home_team_id"]), row["id"]), _INITIAL_ELO)
        for row in merged.iter_rows(named=True)
    ]
    elo_away = [
        elo_history.get((int(row["away_team_id"]), row["id"]), _INITIAL_ELO)
        for row in merged.iter_rows(named=True)
    ]
    merged = merged.with_columns(
        pl.Series("elo_home", elo_home, dtype=pl.Float64),
        pl.Series("elo_away", elo_away, dtype=pl.Float64),
    )
    merged = merged.with_columns(
        (pl.col("elo_home") - pl.col("elo_away")).alias("elo_diff"),
    )

    for m in ("win_roll_10", "win_roll_20", "rest_days"):
        h = f"{m}_home"
        a = f"{m}_away"
        if h in merged.columns and a in merged.columns:
            merged = merged.with_columns((pl.col(h) - pl.col(a)).alias(f"{m}_diff"))

    return merged


async def train_tennis(cfg: TennisTrainConfig | None = None) -> TrainResult:
    cfg = cfg or TennisTrainConfig(seasons=["2023", "2024", "2025"])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches, player_games = await load_tennis_training_data(cfg.seasons)
    if matches.height == 0:
        msg = "Sin datos Tennis"
        raise RuntimeError(msg)

    features_df = build_tennis_feature_frame(matches, player_games)
    features_df = compute_target(features_df, kind="win")

    # Data leakage fix: seed tennis siempre guarda ganador como "home" (~99%
    # home_wins). Augmentation via in-place swap columnas en 50% filas
    # (numpy más simple que polars rename→concat).
    rng = np.random.default_rng(cfg.random_state)
    n_orig = features_df.height
    swap_mask = rng.random(n_orig) < 0.5

    # Convertir a pandas para swap eficiente (polars es más estricto con rename+concat)
    pd_df = features_df.to_pandas()
    mask = swap_mask
    home_cols = [c for c in pd_df.columns if c.endswith("_home")]
    away_cols = [
        c.replace("_home", "_away")
        for c in home_cols
        if c.replace("_home", "_away") in pd_df.columns
    ]
    home_matched = [c.replace("_away", "_home") for c in away_cols]

    # Swap home↔away para filas masked
    for h_col, a_col in zip(home_matched, away_cols, strict=True):
        temp = pd_df.loc[mask, h_col].copy()
        pd_df.loc[mask, h_col] = pd_df.loc[mask, a_col]
        pd_df.loc[mask, a_col] = temp

    # Invertir y + diffs para filas swappeadas
    pd_df.loc[mask, "y"] = 1 - pd_df.loc[mask, "y"].astype(int)
    for c in pd_df.columns:
        if c.endswith("_diff"):
            pd_df.loc[mask, c] = -pd_df.loc[mask, c]

    features_df = pl.from_pandas(pd_df)

    feat_cols = [
        c
        for c in features_df.columns
        if c.endswith(("_home", "_away", "_diff"))
        and features_df.schema[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    features_df = features_df.drop_nulls(subset=["y"])

    n = features_df.height
    if n < 500:
        msg = f"Sample Tennis insuficiente ({n} matches tras feature build)"
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

    with mlflow.start_run(run_name=f"tennis_ml_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params(
            {
                "sport": "tennis",
                "target": "moneyline",
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

        model_path = Path("/tmp") / "tennis_calibrated.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump(
                {
                    "estimator": result.estimator,
                    "conformal": result.conformal,
                    "feature_names": feat_cols,
                    "target": "moneyline",
                    "sport": "tennis",
                },
                f,
            )
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        from apuestas.ml.registry_helper import register_model_in_db

        await register_model_in_db(
            mlflow_run_id=run_id,
            model_name=cfg.experiment_name,
            sport_code="tennis",
            stage=cfg.stage,
            metrics=result.metrics,
        )

    logger.info("tennis.train.done", log_loss=result.holdout_log_loss)
    return result
