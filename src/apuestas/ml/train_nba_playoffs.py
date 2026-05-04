"""NBA Playoff dedicated trainer — Sprint 14 #148.

Playoff ≠ regular season (rotación corta 7-8 jugadores, defensas ajustadas,
home advantage aumenta de ~0.58 a ~0.60+, star players juegan +minutos).

Reusa el pipeline `train_nba.build_training_frame` + `train_ensemble` aplicado
al subset playoff (matches.stage='playoff' OR fechas abr-jun).

Uso:
  python -m apuestas.ml.train_nba_playoffs
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ml.train_base import TrainConfig, train_ensemble
from apuestas.ml.train_nba import (
    _register_in_db,
    build_training_frame,
    time_split,
    to_numpy_xy,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class NBAPlayoffConfig:
    # Histórico extendido (2011-12 a 2024-25): ~2500 games esperados.
    # 2025-26 excluida (en curso). Fix overfit del shadow v20260429_1814
    # (CV 0.63 vs holdout 1.14) — más data + holdout estable temporal.
    seasons: list[str] = field(
        default_factory=lambda: [
            "2011-12",
            "2012-13",
            "2013-14",
            "2014-15",
            "2015-16",
            "2016-17",
            "2017-18",
            "2018-19",
            "2019-20",
            "2020-21",
            "2021-22",
            "2022-23",
            "2023-24",
            "2024-25",
        ]
    )
    n_trials: int = 30
    experiment_name: str = "nba_moneyline_playoff"
    stage: str = "shadow"
    target: str = "win"


async def fetch_playoff_data(seasons: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carga matches playoff + team_games derivados.

    Filtros: stage='playoff' OR (mes 4-7) — abril completo + julio (cierre
    de finales en bubble 2020). Limita por seasons explícitas para evitar
    in-flight 2025-26.
    """
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT m.id, m.external_id, m.start_time, m.home_team_id,
                           m.away_team_id, m.venue_id, m.home_score, m.away_score,
                           m.status, m.season, m.stage
                    FROM matches m
                    WHERE m.sport_code='nba'
                      AND m.status='finished'
                      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
                      AND m.season = ANY(:seasons)
                      AND (
                        m.stage='playoff'
                        OR (EXTRACT(MONTH FROM m.start_time) = 4
                            AND EXTRACT(DAY FROM m.start_time) >= 15)
                        OR EXTRACT(MONTH FROM m.start_time) IN (5, 6, 7)
                      )
                    ORDER BY m.start_time
                    """
                ),
                {"seasons": list(seasons)},
            )
        ).fetchall()

    matches_rows = [dict(r._mapping) for r in rows]
    if not matches_rows:
        return pl.DataFrame(), pl.DataFrame()

    # `stage` es NULL en muchos rows y 'playoff' en otros → polars infiere
    # bool de las primeras filas y revienta al ver el string. Lo dropeamos
    # (build_training_frame no lo usa) y dejamos solo el filtro implícito.
    for r in matches_rows:
        r.pop("stage", None)

    team_rows: list[dict[str, Any]] = []
    for r in matches_rows:
        margin_home = r["home_score"] - r["away_score"]
        total = r["home_score"] + r["away_score"]
        for is_home in (True, False):
            tid = r["home_team_id"] if is_home else r["away_team_id"]
            pts = r["home_score"] if is_home else r["away_score"]
            margin = margin_home if is_home else -margin_home
            team_rows.append(
                {
                    "team_id": tid,
                    "game_id": r["id"],
                    "start_time": r["start_time"],
                    "is_home": is_home,
                    "pts": pts,
                    "win_margin": margin,
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

    return pl.DataFrame(matches_rows), pl.DataFrame(team_rows)


async def train_nba_playoffs(cfg: NBAPlayoffConfig | None = None) -> dict:
    cfg = cfg or NBAPlayoffConfig()
    matches, team_games = await fetch_playoff_data(cfg.seasons)
    n = matches.height if hasattr(matches, "height") else 0
    logger.info("nba_playoffs.data_loaded", n=n)

    if n < 200:
        return {
            "error": "insufficient_data",
            "n_games": n,
            "recommendation": (
                "Necesario: 500+ games. Poblar matches.stage='playoff' via "
                "ingest histórico NBA o ampliar seasons en NBAPlayoffConfig."
            ),
        }

    home_wins = matches.filter(pl.col("home_score") > pl.col("away_score")).height
    home_win_rate = home_wins / n

    frame, feat_cols = build_training_frame(matches, team_games, target=cfg.target)
    logger.info("nba_playoffs.frame_ready", rows=frame.height, features=len(feat_cols))

    if frame.height < 150:
        return {
            "error": "frame_too_small_after_features",
            "rows": frame.height,
            "n_input_games": n,
        }

    train_df, cal_df, holdout_df = time_split(frame, split_train_pct=0.80, split_cal_pct=0.10)
    X_train, y_train = to_numpy_xy(train_df, feat_cols)
    X_cal, y_cal = to_numpy_xy(cal_df, feat_cols)
    X_holdout, y_holdout = to_numpy_xy(holdout_df, feat_cols)

    # MLflow logging + registro shadow
    import os as _os

    import mlflow

    mlflow.set_tracking_uri(_os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=f"nba_playoff_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params(
            {
                "sport": "nba",
                "subset": "playoff",
                "target": cfg.target,
                "seasons": ",".join(cfg.seasons),
                "n_features": len(feat_cols),
                "n_train": len(y_train),
                "n_cal": len(y_cal),
                "n_holdout": len(y_holdout),
                "n_trials": cfg.n_trials,
                "home_win_rate_playoff": round(home_win_rate, 4),
            }
        )
        mlflow.set_tags({"sport": "nba", "market": "moneyline", "context": "playoff"})

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
                random_state=42,
                conformal_alpha=0.1,
                calibration_method="auto",  # ~150 cal samples → sigmoid auto
                enable_stacking=True,
            ),
        )

        for k, v in result.metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except TypeError, ValueError:
                continue

        import cloudpickle

        model_path = Path("/tmp") / "calibrated_model_playoff.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump(
                {
                    "estimator": result.estimator,
                    "conformal": result.conformal,
                    "feature_names": feat_cols,
                    "target": cfg.target,
                    "sport": "nba",
                    "context": "playoff",
                },
                f,
            )
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id

        await _register_in_db(
            mlflow_run_id=run_id,
            model_name=cfg.experiment_name,
            sport_code="nba",
            stage=cfg.stage,
            metrics=result.metrics,
        )

    return {
        "ok": True,
        "n_playoff_games": n,
        "home_win_rate_playoff": round(home_win_rate, 4),
        "holdout_log_loss": result.holdout_log_loss,
        "holdout_ece": result.holdout_ece,
        "holdout_brier": result.metrics.get("holdout_brier"),
        "mlflow_run_id": run_id,
    }


async def _main() -> None:
    cfg = NBAPlayoffConfig()
    r = await train_nba_playoffs(cfg)
    print("NBA Playoffs trainer:")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(_main())
