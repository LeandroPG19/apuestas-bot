"""Pipeline de entrenamiento NBA end-to-end con MLflow logging.

Flow:
1. Extract matches + team_games (via SQLAlchemy) — últimas N temporadas.
2. Features (features.nba.build_nba_feature_frame).
3. Target (win | ats | total) según mercado.
4. Split walk-forward: 80% train / 10% cal / 10% holdout por tiempo.
5. train_ensemble con LGBM+XGB+CatBoost + stacker + calibración + conformal.
6. Log todo a MLflow: params, metrics, artifacts (reliability plot, SHAP summary).
7. Registry: alta en `model_registry_meta` stage=shadow por default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.features.common import compute_target
from apuestas.features.nba import (
    FEATURE_SET_NAME,
    build_nba_feature_frame,
    feature_columns,
    four_factors_from_box,
)
from apuestas.ml.train_base import TrainConfig, TrainResult, train_ensemble
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

Target = Literal["win", "ats", "total"]


@dataclass(slots=True)
class NBATrainConfig:
    seasons: list[str]  # e.g. ["2022-23", "2023-24"]
    target: Target = "win"
    n_trials: int = 40
    split_train_pct: float = 0.80
    split_cal_pct: float = 0.10  # 22k matches — 10% es 2200 samples suficiente isotonic
    random_state: int = 42
    stage: Literal["shadow", "production"] = "shadow"
    experiment_name: str = "nba_moneyline"
    # "auto" delega a select_calibration_method según n_per_class:
    #   n>=1000 → isotonic, 100<=n<1000 → sigmoid, n<100 → venn_abers.
    # Datasets <2k matches caen en sigmoid (más robusto para ECE).
    calibration_method: Literal["auto", "sigmoid", "isotonic", "venn_abers"] = "auto"


async def load_nba_training_data(
    seasons: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extrae (matches, team_games) para las temporadas especificadas.

    team_games tiene 2 rows por partido (home + away) con stats Four Factors
    derivados del boxscore. Si no hay boxscores aún, pipeline degrada a
    usar solo scores finales (feature set reducido).
    """
    seasons_str = [str(s) for s in seasons]

    async with session_scope() as session:
        # Matches
        m = await session.execute(
            text(
                """
                SELECT id, external_id, home_team_id, away_team_id, start_time,
                       venue_id, home_score, away_score, status, season
                FROM matches
                WHERE sport_code = 'nba'
                  AND season = ANY(:seasons)
                  AND status = 'finished'
                ORDER BY start_time
                """
            ),
            {"seasons": seasons_str},
        )
        matches_rows = [dict(r._mapping) for r in m.all()]

        if not matches_rows:
            logger.warning("nba.train.no_matches", seasons=seasons_str)
            return pl.DataFrame(), pl.DataFrame()

        # Team games derivados: 2 rows por match.
        # NOTA: en una versión con boxscores reales, se consultaría una tabla
        # boxscores_nba; aquí derivamos del score final lo mínimo necesario.
        team_rows: list[dict[str, Any]] = []
        for r in matches_rows:
            if r["home_score"] is None or r["away_score"] is None:
                continue
            margin_home = r["home_score"] - r["away_score"]
            total = r["home_score"] + r["away_score"]
            team_rows.append(
                {
                    "team_id": r["home_team_id"],
                    "game_id": r["id"],
                    "start_time": r["start_time"],
                    "is_home": True,
                    "pts": r["home_score"],
                    "win_margin": margin_home,
                    "total_points": total,
                    # Features Four Factors dummy (llenar con boxscores reales)
                    "fgm": np.nan,
                    "fga": np.nan,
                    "fg3m": np.nan,
                    "ftm": np.nan,
                    "fta": np.nan,
                    "oreb": np.nan,
                    "dreb": np.nan,
                    "tov": np.nan,
                    "ortg": np.nan,
                    "drtg": np.nan,
                }
            )
            team_rows.append(
                {
                    "team_id": r["away_team_id"],
                    "game_id": r["id"],
                    "start_time": r["start_time"],
                    "is_home": False,
                    "pts": r["away_score"],
                    "win_margin": -margin_home,
                    "total_points": total,
                    "fgm": np.nan,
                    "fga": np.nan,
                    "fg3m": np.nan,
                    "ftm": np.nan,
                    "fta": np.nan,
                    "oreb": np.nan,
                    "dreb": np.nan,
                    "tov": np.nan,
                    "ortg": np.nan,
                    "drtg": np.nan,
                }
            )

    matches_df = pl.DataFrame(matches_rows)
    team_games_df = pl.DataFrame(team_rows)
    logger.info(
        "nba.train.loaded",
        matches=matches_df.height,
        team_games=team_games_df.height,
    )
    return matches_df, team_games_df


def build_training_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    target: Target,
    boxscores: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Pipeline features → target → devuelve (DataFrame, feature_cols)."""
    if boxscores is not None and boxscores.height > 0:
        ff = four_factors_from_box(boxscores)
        team_games = team_games.join(
            ff.select(["game_id", "team_id", "efg_pct", "tov_pct", "orb_pct", "ft_rate", "pace"]),
            on=["game_id", "team_id"],
            how="left",
        )

    features_df = build_nba_feature_frame(matches, team_games)
    # Sprint 10 Fase 2: Elo features (rating bidireccional + anti-leakage)
    from apuestas.features.common import add_elo_features

    features_df = add_elo_features(features_df, sport="nba")

    # Sprint 14 #149 — context features (rest/b2b/travel/referee) opt-in.
    if __import__("os").environ.get("APUESTAS_NBA_CONTEXT_FEATURES", "true").lower() == "true":
        try:
            features_df = _add_nba_context_columns_sync(features_df, matches)
            logger.info("nba.context_features.added")
        except Exception as exc:
            logger.warning("nba.context_features.skip", error=str(exc)[:100])
    # Para target=total/ats, compute_target requiere col extra. Como no tenemos
    # líneas históricas por match (solo 25/22134 con odds_history), aplicamos
    # un threshold fijo basado en distribución NBA: total ~225 (over/under),
    # spread home -3.5 (cubre/no). Subóptimo vs línea real pero permite
    # baseline funcional. Mismo approach que train_mlb total con line=8.5.
    if target == "total":
        features_df = features_df.with_columns(
            ((pl.col("home_score") + pl.col("away_score")) > 224.5).cast(pl.Int8).alias("y")
        )
    elif target == "ats":
        # Target: home cubre -3.5 (gana por 4+). Aproximación al spread NBA típico
        # del favorito ligero. Modelo predice "p(home cubre -3.5)" — interpretar
        # downstream comparando con línea real.
        features_df = features_df.with_columns(
            ((pl.col("home_score") - pl.col("away_score")) >= 4).cast(pl.Int8).alias("y")
        )
    else:
        features_df = compute_target(
            features_df,
            kind=target,
            home_score_col="home_score",
            away_score_col="away_score",
        )

    feat_cols = feature_columns(features_df)
    # Limpiar rows sin target o con todas las features NaN (primeros juegos)
    features_df = features_df.drop_nulls(subset=["y"])
    # Drop rows con >50% features NaN
    non_null_frac = features_df.select(
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in feat_cols]).alias(
            "n_non_null"
        )
    )["n_non_null"] / len(feat_cols)
    mask = non_null_frac > 0.5
    features_df = features_df.filter(mask)

    return features_df, feat_cols


def _add_nba_context_columns_sync(features_df: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Compute rest days + b2b from matches schedule (purely from team game sequence).

    Sin call a DB — usa el `matches` DataFrame ya traído. Features:
      - days_rest_home/away, is_b2b_home/away, days_rest_diff.
    """
    if matches.height == 0 or features_df.height == 0:
        return features_df
    # Previous game per team from matches (long format)
    home_records = matches.select(
        pl.col("id").alias("game_id"),
        pl.col("home_team_id").alias("team_id"),
        pl.col("start_time").alias("game_time"),
    )
    away_records = matches.select(
        pl.col("id").alias("game_id"),
        pl.col("away_team_id").alias("team_id"),
        pl.col("start_time").alias("game_time"),
    )
    all_games = pl.concat([home_records, away_records]).sort(["team_id", "game_time"])
    all_games = all_games.with_columns(
        (pl.col("game_time") - pl.col("game_time").shift(1).over("team_id"))
        .dt.total_seconds()
        .alias("seconds_rest")
    )
    all_games = all_games.with_columns(
        (pl.col("seconds_rest") / 86400.0).alias("days_rest"),
    )
    all_games = all_games.with_columns(
        pl.when(pl.col("days_rest") < 1.2).then(1.0).otherwise(0.0).alias("is_b2b")
    )

    # Join back to features
    home_rest = all_games.rename(
        {"team_id": "home_team_id", "days_rest": "days_rest_home", "is_b2b": "is_b2b_home"}
    ).select(["game_id", "home_team_id", "days_rest_home", "is_b2b_home"])
    away_rest = all_games.rename(
        {"team_id": "away_team_id", "days_rest": "days_rest_away", "is_b2b": "is_b2b_away"}
    ).select(["game_id", "away_team_id", "days_rest_away", "is_b2b_away"])

    out = features_df
    for name in ("game_id", "home_team_id", "away_team_id"):
        if name not in out.columns and name in matches.columns:
            pass
    if "game_id" in out.columns:
        out = out.join(home_rest, on=["game_id", "home_team_id"], how="left").join(
            away_rest, on=["game_id", "away_team_id"], how="left"
        )
        out = out.with_columns(
            (pl.col("days_rest_home") - pl.col("days_rest_away")).alias("days_rest_diff")
        )
    return out.fill_null(2.0)  # default 2 días rest


def time_split(
    df: pl.DataFrame,
    *,
    split_train_pct: float,
    split_cal_pct: float,
    time_col: str = "start_time",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split temporal 3-way por percentil de fecha."""
    df_sorted = df.sort(time_col)
    n = df_sorted.height
    n_train = int(n * split_train_pct)
    n_cal = int(n * split_cal_pct)
    train = df_sorted.slice(0, n_train)
    cal = df_sorted.slice(n_train, n_cal)
    holdout = df_sorted.slice(n_train + n_cal, n - n_train - n_cal)
    return train, cal, holdout


def to_numpy_xy(
    df: pl.DataFrame, feature_cols: list[str], *, target_col: str = "y"
) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feature_cols).fill_nan(0.0).fill_null(0.0).to_numpy()
    y = df[target_col].to_numpy().astype(np.int8)
    return X, y


async def train_nba(cfg: NBATrainConfig | None = None) -> TrainResult:
    """Pipeline completo con MLflow logging + registro shadow."""
    cfg = cfg or NBATrainConfig(seasons=["2023-24"])

    import os as _os

    mlflow.set_tracking_uri(_os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches, team_games = await load_nba_training_data(cfg.seasons)
    if matches.height == 0:
        msg = "Sin datos de entrenamiento NBA. ¿Has corrido ingest_nba_season?"
        raise RuntimeError(msg)

    frame, feat_cols = build_training_frame(matches, team_games, target=cfg.target)
    logger.info("nba.train.frame_ready", rows=frame.height, features=len(feat_cols))

    train_df, cal_df, holdout_df = time_split(
        frame,
        split_train_pct=cfg.split_train_pct,
        split_cal_pct=cfg.split_cal_pct,
    )

    X_train, y_train = to_numpy_xy(train_df, feat_cols)
    X_cal, y_cal = to_numpy_xy(cal_df, feat_cols)
    X_holdout, y_holdout = to_numpy_xy(holdout_df, feat_cols)

    with mlflow.start_run(
        run_name=f"nba_{cfg.target}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
    ):
        # Log config
        mlflow.log_params(
            {
                "sport": "nba",
                "target": cfg.target,
                "seasons": ",".join(cfg.seasons),
                "feature_set": FEATURE_SET_NAME,
                "n_features": len(feat_cols),
                "n_train": len(y_train),
                "n_cal": len(y_cal),
                "n_holdout": len(y_holdout),
                "random_state": cfg.random_state,
                "n_trials": cfg.n_trials,
            }
        )
        mlflow.set_tags(
            {
                "sport": "nba",
                "market": cfg.target,
                "feature_set": FEATURE_SET_NAME,
                "calibration": "isotonic+conformal",
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
                target_col="y",
                n_trials=cfg.n_trials,
                random_state=cfg.random_state,
                conformal_alpha=0.1,
                calibration_method=cfg.calibration_method,
                enable_stacking=True,
            ),
        )

        # Log métricas
        for k, v in result.metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):  # fmt: skip
                continue

        # Log feature list como artifact
        import json

        tmp = Path("/tmp") / f"features_{cfg.experiment_name}.json"
        tmp.write_text(json.dumps(feat_cols, indent=2))
        mlflow.log_artifact(str(tmp))

        # Sanity check benchmarks del plan
        ok_logloss = result.holdout_log_loss <= 0.67
        ok_ece = result.holdout_ece < 0.03
        mlflow.set_tags(
            {
                "meets_logloss_target": str(ok_logloss),
                "meets_ece_target": str(ok_ece),
            }
        )

        # Log final model pickled
        import cloudpickle

        model_path = Path("/tmp") / "calibrated_model.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump(
                {
                    "estimator": result.estimator,
                    "conformal": result.conformal,
                    "feature_names": feat_cols,
                    "target": cfg.target,
                    "sport": "nba",
                },
                f,
            )
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id

        # Registrar en model_registry_meta (shadow stage)
        await _register_in_db(
            mlflow_run_id=run_id,
            model_name=cfg.experiment_name,
            sport_code="nba",
            stage=cfg.stage,
            metrics=result.metrics,
        )

    logger.info(
        "nba.train.done",
        holdout_log_loss=result.holdout_log_loss,
        holdout_ece=result.holdout_ece,
        meets_targets=ok_logloss and ok_ece,
    )
    return result


async def _register_in_db(
    *,
    mlflow_run_id: str,
    model_name: str,
    sport_code: str,
    stage: str,
    metrics: dict[str, float],
) -> None:
    """Insert/update en `model_registry_meta` tras cada entrenamiento."""
    import json as _json

    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO model_registry_meta
                  (mlflow_run_id, model_name, model_version, sport_code,
                   stage, promoted_at, performance_30d)
                VALUES
                  (:run_id, :name, :version, :sport, :stage, NOW(), CAST(:perf AS jsonb))
                ON CONFLICT (mlflow_run_id) DO UPDATE
                  SET stage = EXCLUDED.stage,
                      performance_30d = EXCLUDED.performance_30d
                """
            ),
            {
                "run_id": mlflow_run_id,
                "name": model_name,
                "version": datetime.now(tz=UTC).strftime("%Y%m%d_%H%M"),
                "sport": sport_code,
                "stage": stage,
                "perf": _json.dumps(metrics),
            },
        )
