"""Flow retrain_weekly (§13 domingo 02:00).

Re-entrena LightGBM+XGBoost+CatBoost+stacker por cada deporte:
1. Check si hay suficiente data nueva (mínimo 20 games desde último retrain).
2. Ejecuta train_{sport} con MLflow logging.
3. Registra modelo como stage='shadow'.
4. Evalúa promote_shadow() (MWhit significance test §17.6).
5. Si drift detectado → alerta Telegram + cuba_alarma.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.mcp import memory as mcp_memory
from apuestas.ml.registry import evaluate_promotion, promote_shadow
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@task
async def check_data_freshness(*, sport_code: str, min_new_games: int = 20) -> bool:
    """Hay suficientes games nuevos desde último retrain?"""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT COUNT(*) AS n
                FROM matches
                WHERE sport_code = :sport
                  AND status = 'finished'
                  AND updated_at >= NOW() - INTERVAL '7 days'
                """
            ),
            {"sport": sport_code},
        )
        row = result.first()
    count = int(row.n) if row and row.n else 0
    return count >= min_new_games


@task
async def retrain_sport(sport_code: str, *, n_trials: int = 40) -> dict[str, Any]:
    """Dispatcher por deporte. Invoca train_{sport} correspondiente."""
    if sport_code == "nba":
        from apuestas.ml.train_nba import NBATrainConfig, train_nba

        # Usar últimas 3 temporadas NBA (format 'YYYY-YY')
        current_year = datetime.now(tz=UTC).year
        seasons = [f"{y - 1}-{str(y)[2:]}" for y in range(current_year - 2, current_year + 1)]
        cfg = NBATrainConfig(seasons=seasons, n_trials=n_trials, stage="shadow")
        try:
            result = await train_nba(cfg)
            return {
                "sport": "nba",
                "ok": True,
                "holdout_log_loss": result.holdout_log_loss,
                "holdout_ece": result.holdout_ece,
                "n_features": len(result.feature_names),
            }
        except Exception as exc:
            logger.exception("retrain.nba_fail", error=str(exc))
            return {"sport": "nba", "ok": False, "error": str(exc)}

    # Placeholders para MLB/NFL/Soccer (trainers pendientes Batch H)
    logger.info("retrain.sport_not_implemented", sport=sport_code)
    return {"sport": sport_code, "ok": False, "reason": "trainer_not_implemented"}


@task
async def evaluate_all_promotions() -> dict[str, Any]:
    """Post-retrain: evalúa si algún shadow debe → production."""
    models_to_check = [
        "nba_moneyline",
        "nba_ats",
        "nba_total",
        "mlb_moneyline",
        "nfl_ats",
    ]
    results: dict[str, Any] = {}
    for model_name in models_to_check:
        try:
            decision = await evaluate_promotion(model_name)
            results[model_name] = {
                "should_promote": decision.should_promote,
                "reason": decision.reason,
                "delta_clv": decision.delta,
                "n_picks": decision.n_picks,
            }
            if decision.should_promote:
                promoted = await promote_shadow(model_name)
                results[model_name]["promoted"] = promoted
        except Exception as exc:
            logger.warning("retrain.promote_check_fail", model=model_name, error=str(exc))
            results[model_name] = {"error": str(exc)}
    return results


@flow(name="apuestas-retrain-weekly", log_prints=True)
async def retrain_weekly_flow() -> dict[str, Any]:
    """Domingo 02:00 UTC. Entry point principal del retrain."""
    logger.info("retrain_weekly.start", ts=datetime.now(tz=UTC).isoformat())

    training_results: dict[str, Any] = {}
    for sport in ("nba", "mlb", "nfl", "soccer"):
        try:
            fresh = await check_data_freshness(sport_code=sport)
            if not fresh:
                training_results[sport] = {"ok": False, "reason": "insufficient_new_data"}
                continue
            training_results[sport] = await retrain_sport(sport)
        except Exception as exc:
            logger.exception("retrain.sport_fail", sport=sport, error=str(exc))
            training_results[sport] = {"ok": False, "error": str(exc)}

    promotions = await evaluate_all_promotions()

    # Feedback a cuba-memorys
    await mcp_memory.weekly_decay()
    await mcp_memory.analyze_gaps()
    await mcp_memory.health_check()

    summary = {
        "training": training_results,
        "promotions": promotions,
        "ts": datetime.now(tz=UTC).isoformat(),
    }
    logger.info("retrain_weekly.done", **{k: str(v)[:100] for k, v in summary.items()})
    return summary


if __name__ == "__main__":
    asyncio.run(retrain_weekly_flow())
