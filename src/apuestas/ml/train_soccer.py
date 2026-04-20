"""Pipeline entrenamiento fútbol con Dixon-Coles (§6).

Estrategia dual:
1. Dixon-Coles (penaltyblog) para P(home_goals, away_goals) → 1X2/totals/BTTS/AH.
2. LightGBM stacker sobre residuos DC + features xG/form/Elo.

Liga MX prioritario (§22), Big-5 europea secundaria.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cloudpickle
import mlflow
import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SoccerTrainConfig:
    league_id: int
    seasons: list[str]
    n_trials: int = 20
    random_state: int = 42
    stage: str = "shadow"
    experiment_name: str = "soccer_liga_mx"
    xi_decay: float = 0.0018  # Dixon-Coles decay per day (blueprint §6)


async def load_soccer_data(league_id: int, seasons: list[str]) -> list[dict[str, Any]]:
    """Carga matches finalizados de la liga."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.home_team_id AS home_id, m.away_team_id AS away_id,
                       m.home_score AS home_goals, m.away_score AS away_goals,
                       m.start_time AS date
                FROM matches m
                WHERE m.sport_code = 'soccer'
                  AND m.league_id = :lid
                  AND m.season = ANY(:seasons)
                  AND m.status = 'finished'
                ORDER BY m.start_time
                """
            ),
            {"lid": league_id, "seasons": seasons},
        )
        return [dict(r._mapping) for r in result.all()]


def fit_dixon_coles(matches: list[dict[str, Any]], xi: float = 0.0018) -> Any:
    """Entrena Dixon-Coles con penaltyblog."""
    try:
        from penaltyblog.models import DixonColesGoalModel
    except ImportError:
        logger.error("soccer.penaltyblog_missing")
        return None

    # Preparar DataFrame
    df = pl.DataFrame(matches)
    if df.height == 0:
        return None

    home = df["home_id"].to_numpy()
    away = df["away_id"].to_numpy()
    home_goals = df["home_goals"].to_numpy()
    away_goals = df["away_goals"].to_numpy()
    # Weights exponenciales por decay
    today = datetime.now(tz=UTC)
    dates = df["date"].to_numpy()
    days_ago = np.array([(today - d).total_seconds() / 86400 for d in dates])
    weights = np.exp(-xi * days_ago)

    try:
        model = DixonColesGoalModel(home_goals, away_goals, home, away, weights)
        model.fit()
        logger.info("soccer.dc_fit_ok", n_matches=len(matches), xi=xi)
        return model
    except Exception as exc:
        logger.exception("soccer.dc_fit_failed", error=str(exc))
        return None


def evaluate_dc_model(model: Any, holdout: list[dict[str, Any]]) -> dict[str, float]:
    """Computa log-loss + Brier 1X2 sobre holdout."""
    if model is None or not holdout:
        return {}

    losses: list[float] = []
    briers: list[float] = []
    for m in holdout:
        try:
            prediction = model.predict(m["home_id"], m["away_id"])
            # penaltyblog devuelve dict con probabilities {home_win, draw, away_win}
            probs = prediction.home_draw_away
            hg = int(m["home_goals"])
            ag = int(m["away_goals"])
            if hg > ag:
                actual_idx = 0
                actual = np.array([1, 0, 0])
            elif hg == ag:
                actual_idx = 1
                actual = np.array([0, 1, 0])
            else:
                actual_idx = 2
                actual = np.array([0, 0, 1])
            p_actual = max(probs[actual_idx], 1e-7)
            losses.append(-np.log(p_actual))
            briers.append(float(np.sum((np.asarray(probs) - actual) ** 2)) / 3.0)
        except Exception as exc:
            logger.debug("soccer.dc_predict_fail", error=str(exc))
            continue

    if not losses:
        return {}
    return {
        "log_loss": float(np.mean(losses)),
        "brier": float(np.mean(briers)),
        "n_holdout": len(losses),
    }


async def train_soccer(cfg: SoccerTrainConfig | None = None) -> dict[str, Any]:
    """Pipeline Dixon-Coles con MLflow logging."""
    cfg = cfg or SoccerTrainConfig(league_id=262, seasons=["2024", "2025", "2026"])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches = await load_soccer_data(cfg.league_id, cfg.seasons)
    if len(matches) < 100:
        msg = f"Muestra insuficiente ({len(matches)} matches) para Dixon-Coles"
        raise RuntimeError(msg)

    # Walk-forward 80/20
    split_idx = int(len(matches) * 0.8)
    train_matches = matches[:split_idx]
    holdout_matches = matches[split_idx:]

    with mlflow.start_run(
        run_name=f"soccer_{cfg.league_id}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
    ):
        mlflow.log_params(
            {
                "league_id": cfg.league_id,
                "seasons": ",".join(cfg.seasons),
                "model": "dixon_coles",
                "xi_decay": cfg.xi_decay,
                "n_train": len(train_matches),
                "n_holdout": len(holdout_matches),
            }
        )

        model = fit_dixon_coles(train_matches, xi=cfg.xi_decay)
        if model is None:
            return {"ok": False, "reason": "dc_fit_failed"}

        metrics = evaluate_dc_model(model, holdout_matches)
        for k, v in metrics.items():
            if isinstance(v, int | float):
                mlflow.log_metric(k, float(v))

        # Log modelo
        model_path = Path("/tmp") / "soccer_dc.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump({"model": model, "config": cfg}, f)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        logger.info(
            "soccer.train.done",
            league=cfg.league_id,
            log_loss=metrics.get("log_loss"),
            brier=metrics.get("brier"),
        )

    return {"ok": True, "metrics": metrics}
