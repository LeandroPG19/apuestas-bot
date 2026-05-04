"""Retrain event-driven ante drift (Deuda 6 / plan §9.2).

Listener de Prefect events `concept_drift.{sport}.{market}` emitidos por
`monitors.concept_drift.BrierDriftMonitor`. Cada evento dispara retrain
del deporte correspondiente con `warm_start=True`. Cooldown 24h está en
BrierDriftMonitor; aquí sólo orquestamos el retrain.

También expone `trigger_retrain_manual(sport)` para `scripts/trigger_nfl_retrain.sh`
o cualquier otro disparo ad-hoc.
"""

from __future__ import annotations

import asyncio
from typing import Any

from prefect import flow

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


_RETRAIN_DISPATCH: dict[str, Any] = {}


def _lazy_dispatch() -> dict[str, Any]:
    """Carga las funciones train_* bajo demanda para evitar imports caros."""
    if _RETRAIN_DISPATCH:
        return _RETRAIN_DISPATCH
    from apuestas.ml.train_mlb import train_mlb
    from apuestas.ml.train_nba import train_nba
    from apuestas.ml.train_nfl import train_nfl
    from apuestas.ml.train_soccer import train_soccer

    _RETRAIN_DISPATCH["nba"] = train_nba
    _RETRAIN_DISPATCH["nfl"] = train_nfl
    _RETRAIN_DISPATCH["mlb"] = train_mlb
    _RETRAIN_DISPATCH["soccer"] = train_soccer
    return _RETRAIN_DISPATCH


@flow(name="apuestas-retrain-on-drift", log_prints=True)
async def retrain_on_drift_flow(*, sport: str, market: str = "h2h") -> dict[str, Any]:
    """Trigger retrain ante drift event.

    Registrado en Prefect con event trigger:
        prefect deployment run retrain_on_drift --event "concept_drift.*"
    """
    dispatch = _lazy_dispatch()
    trainer = dispatch.get(sport.lower())
    if trainer is None:
        logger.warning("retrain_on_drift.unsupported_sport", sport=sport)
        return {"skipped": True, "sport": sport, "reason": "no_trainer"}

    logger.info("retrain_on_drift.start", sport=sport, market=market)
    try:
        result = await trainer()
        logger.info(
            "retrain_on_drift.done",
            sport=sport,
            market=market,
            log_loss=getattr(result, "holdout_log_loss", None),
        )
        return {
            "sport": sport,
            "market": market,
            "log_loss": float(getattr(result, "holdout_log_loss", 0) or 0),
        }
    except Exception as exc:
        logger.warning("retrain_on_drift.fail", sport=sport, error=str(exc)[:120])
        return {"sport": sport, "market": market, "error": str(exc)[:200]}


async def trigger_retrain_manual(sport: str) -> dict[str, Any]:
    """Disparo manual desde CLI/Telegram. Bypass del Prefect event system."""
    return await retrain_on_drift_flow(sport=sport)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", required=True)
    parser.add_argument("--market", default="h2h")
    args = parser.parse_args()
    result = asyncio.run(retrain_on_drift_flow(sport=args.sport, market=args.market))
    print(result)
