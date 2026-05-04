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
    """Dispatcher por deporte. Invoca train_{sport} correspondiente.

    Temporadas calculadas dinámicamente del año del sistema.
    """
    current_year = datetime.now(tz=UTC).year

    if sport_code == "nba":
        from apuestas.ml.train_nba import NBATrainConfig, train_nba

        seasons = [f"{y - 1}-{str(y)[2:]}" for y in range(current_year - 2, current_year + 1)]
        # Entrenar 3 targets como MLB: moneyline (h2h), total (O/U 224.5), ats (home -3.5).
        # Sin esto, 50% outcomes NBA skipean por no_model_for_market.
        results: dict[str, Any] = {"sport": "nba", "targets": {}}
        for tgt, exp_name in (
            ("win", "nba_moneyline"),
            ("total", "nba_total"),
            ("ats", "nba_ats"),
        ):
            cfg = NBATrainConfig(
                seasons=seasons,
                target=tgt,  # type: ignore[arg-type]
                n_trials=n_trials,
                stage="shadow",
                experiment_name=exp_name,
            )
            try:
                r = await train_nba(cfg)
                results["targets"][tgt] = {
                    "ok": True,
                    "holdout_log_loss": r.holdout_log_loss,
                    "holdout_ece": r.holdout_ece,
                    "n_features": len(r.feature_names),
                }
            except Exception as exc:
                logger.exception("retrain.nba_fail", target=tgt, error=str(exc))
                results["targets"][tgt] = {"ok": False, "error": str(exc)[:200]}
        results["ok"] = any(t.get("ok") for t in results["targets"].values())
        return results

    if sport_code == "mlb":
        from apuestas.ml.train_mlb import MLBTrainConfig, train_mlb

        seasons = list(range(current_year - 4, current_year + 1))
        results: dict[str, Any] = {"sport": "mlb", "targets": {}}
        # Entrenar 3 targets: moneyline, runline (±1.5), total (O/U 8.5).
        # Pre-2026-04-27 solo se entrenaba moneyline; emitir picks de spreads/totals
        # usando ese modelo causaba overconfidence severa (Brier 0.305 spreads).
        for tgt, exp_name in (
            ("moneyline", "mlb_moneyline"),
            ("runline", "mlb_runline"),
            ("total", "mlb_total"),
        ):
            cfg = MLBTrainConfig(
                years=seasons,
                target=tgt,  # type: ignore[arg-type]
                n_trials=n_trials,
                stage="shadow",
                experiment_name=exp_name,
            )
            try:
                r = await train_mlb(cfg)
                results["targets"][tgt] = {
                    "ok": True,
                    "holdout_log_loss": r.holdout_log_loss,
                    "holdout_ece": r.holdout_ece,
                    "n_features": len(r.feature_names),
                }
            except Exception as exc:
                logger.exception("retrain.mlb_fail", target=tgt, error=str(exc))
                results["targets"][tgt] = {"ok": False, "error": str(exc)[:200]}
        results["ok"] = any(t.get("ok") for t in results["targets"].values())
        return results

    if sport_code == "nfl":
        from apuestas.ml.train_nfl import NFLTrainConfig, train_nfl

        # NFL usa formato "YYYY-YY" (ej "2021-22") en DB.season.
        seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(current_year - 5, current_year)]
        cfg = NFLTrainConfig(seasons=seasons, n_trials=n_trials, stage="shadow")
        try:
            result = await train_nfl(cfg)
            return {
                "sport": "nfl",
                "ok": True,
                "holdout_log_loss": result.holdout_log_loss,
                "holdout_ece": result.holdout_ece,
                "n_features": len(result.feature_names),
            }
        except Exception as exc:
            logger.exception("retrain.nfl_fail", error=str(exc))
            return {"sport": "nfl", "ok": False, "error": str(exc)[:200]}

    elif sport_code == "soccer":
        from sqlalchemy import text

        from apuestas.db import session_scope
        from apuestas.ml.train_soccer import SoccerTrainConfig, train_soccer

        # Soccer requiere league_id por modelo. Detectar dinámicamente ligas con
        # >= 500 matches en la DB y entrenar una por una.
        async with session_scope() as session:
            leagues = (
                await session.execute(
                    text(
                        """
                        SELECT m.league_id, COUNT(*) AS n
                        FROM matches m
                        WHERE m.sport_code = 'soccer'
                          AND m.status = 'finished'
                          AND m.league_id IS NOT NULL
                        GROUP BY m.league_id
                        HAVING COUNT(*) >= 500
                        ORDER BY n DESC
                        LIMIT 5
                        """
                    )
                )
            ).all()

            # Descubrir formato real de season en la DB (soccer usa "YYYY-YYYY"
            # típicamente, otros feeds podrían usar "YYYY" solo).
            seasons_rows = (
                await session.execute(
                    text(
                        """
                        SELECT DISTINCT season
                        FROM matches
                        WHERE sport_code = 'soccer'
                          AND status = 'finished'
                          AND season IS NOT NULL
                        ORDER BY season DESC
                        LIMIT 10
                        """
                    )
                )
            ).all()

        available_seasons = [row.season for row in seasons_rows if row.season]
        if not available_seasons:
            return {"sport": "soccer", "ok": False, "reason": "no_seasons_in_db"}

        # Usar últimas 5 seasons descubiertas (respeta formato real de DB).
        seasons_str = available_seasons[:5]

        results_by_league: dict[int, Any] = {}
        for row in leagues:
            league_id = int(row.league_id)
            cfg = SoccerTrainConfig(
                league_id=league_id,
                seasons=seasons_str,
                n_trials=n_trials,
                stage="shadow",
                experiment_name=f"soccer_league_{league_id}",
            )
            try:
                r = await train_soccer(cfg)
                results_by_league[league_id] = {k: v for k, v in r.items() if not callable(v)}
            except Exception as exc:
                logger.exception("retrain.soccer_league_fail", league_id=league_id)
                results_by_league[league_id] = {"error": str(exc)[:200]}
        if not results_by_league:
            return {"sport": "soccer", "ok": False, "reason": "no_leagues_with_data"}
        return {
            "sport": "soccer",
            "ok": True,
            "leagues": results_by_league,
            "seasons_used": seasons_str,
        }

    elif sport_code == "nhl":
        from sqlalchemy import text

        from apuestas.db import session_scope
        from apuestas.ml.train_nhl import NHLTrainConfig, train_nhl

        # Discover seasons reales (puede ser "YYYY-YY" en histórico).
        async with session_scope() as session:
            season_rows = (
                await session.execute(
                    text(
                        """
                        SELECT DISTINCT season
                        FROM matches
                        WHERE sport_code = 'nhl'
                          AND status = 'finished'
                          AND season IS NOT NULL
                        ORDER BY season DESC
                        LIMIT 8
                        """
                    )
                )
            ).all()
        seasons = [r.season for r in season_rows if r.season]
        if not seasons:
            return {"sport": "nhl", "ok": False, "reason": "no_seasons_in_db"}
        cfg = NHLTrainConfig(seasons=seasons, n_trials=n_trials, stage="shadow")
        try:
            result = await train_nhl(cfg)
            return {
                "sport": "nhl",
                "ok": True,
                "holdout_log_loss": result.holdout_log_loss,
                "holdout_ece": result.holdout_ece,
                "n_features": len(result.feature_names),
                "seasons_used": seasons,
            }
        except Exception as exc:
            logger.exception("retrain.nhl_fail", error=str(exc))
            return {"sport": "nhl", "ok": False, "error": str(exc)[:200]}

    elif sport_code == "tennis":
        from sqlalchemy import text

        from apuestas.db import session_scope
        from apuestas.ml.train_tennis import TennisTrainConfig, train_tennis

        async with session_scope() as session:
            season_rows = (
                await session.execute(
                    text(
                        """
                        SELECT DISTINCT season
                        FROM matches
                        WHERE sport_code = 'tennis'
                          AND status = 'finished'
                          AND season IS NOT NULL
                        ORDER BY season DESC
                        LIMIT 6
                        """
                    )
                )
            ).all()
        seasons = [r.season for r in season_rows if r.season]
        if not seasons:
            return {"sport": "tennis", "ok": False, "reason": "no_seasons_in_db"}
        cfg = TennisTrainConfig(seasons=seasons, n_trials=n_trials, stage="shadow")
        try:
            result = await train_tennis(cfg)
            return {
                "sport": "tennis",
                "ok": True,
                "holdout_log_loss": result.holdout_log_loss,
                "holdout_ece": result.holdout_ece,
                "n_features": len(result.feature_names),
                "seasons_used": seasons,
            }
        except Exception as exc:
            logger.exception("retrain.tennis_fail", error=str(exc))
            return {"sport": "tennis", "ok": False, "error": str(exc)[:200]}

    logger.info("retrain.sport_unknown", sport=sport_code)
    return {"sport": sport_code, "ok": False, "reason": "unknown_sport"}


@task
async def evaluate_all_promotions() -> dict[str, Any]:
    """Post-retrain: evalúa si algún shadow debe → production."""
    models_to_check = [
        "nba_moneyline",
        "nba_ats",
        "nba_total",
        "mlb_moneyline",
        "nfl_ats",
        "nhl_moneyline",
        "tennis_moneyline",
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
    for sport in ("nba", "mlb", "nfl", "soccer", "nhl", "tennis"):
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
    import argparse

    parser = argparse.ArgumentParser(description="Retrain sport(s)")
    parser.add_argument(
        "--sport",
        default=None,
        help="Single sport code (nba/mlb/nfl/nhl/soccer/tennis). Sin arg corre todo.",
    )
    parser.add_argument("--n-trials", type=int, default=40, help="Optuna trials per stage")
    args = parser.parse_args()

    if args.sport:
        trigger_fn = getattr(retrain_sport, "fn", retrain_sport)
        result = asyncio.run(trigger_fn(args.sport, n_trials=args.n_trials))
        print(f"{args.sport} RESULT: {result}")
    else:
        result = asyncio.run(retrain_weekly_flow())
        print(f"FULL RESULT: {result}")
