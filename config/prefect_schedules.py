"""Prefect schedules centralizados (Gap 7 / Sprint §14).

Cada flow del bot tiene un cron aquí. Se aplica via `apuestas.scripts.deploy`
(futuro) que hace `Deployment.build_from_flow().apply()`. En modo on-demand
los schedules quedan pausados; el usuario hace `apuestas go` para arrancarlos.
"""

from __future__ import annotations

try:
    from prefect.client.schemas.schedules import CronSchedule
except ImportError:  # pragma: no cover
    CronSchedule = None  # type: ignore[assignment,misc]


# Cron strings (UTC). Los escogí así:
#  - catchup cada 30 min (balance entre frescura y créditos API)
#  - deep_analysis cada 60 min (LLM latencia + RAG cost)
#  - steam_watcher cada 120 min (mercado intra-día)
#  - alert_cleanup cada 6 h
#  - live_scores cada 15 min durante ventana 12h-04h UTC
#  - drift_monitor 03:00 diario (post-settle del día)
#  - rag_preembed 02:00 diario (off-peak GPU)
#
# IMPORTANTE — modo on-demand vs Prefect cron:
#   El bot opera en modo on-demand vía `_auto_analysis_loop` (Telegram /auto_on)
#   que orquesta catchup/deep_analysis/CLV cada 360 min de forma idempotente.
#   Si los Prefect schedules abajo se aplican CON el auto_loop activo se produce
#   doble (o triple) gasto de OddsAPI: catchup_flow corre 48× día por cron + 4×
#   día por auto_loop + 4× día por deep_analysis interno = 56× catchup/día.
#   Por eso los crones que tocan OddsAPI quedan vacíos por default. Solo el
#   trabajo realmente complementario (drift_monitor, rag_preembed) sigue cron.
SCHEDULES: dict[str, str] = {
    # Schedules ON-DEMAND (los gestiona auto_loop) — vacíos para no doble-llamar
    # OddsAPI. Si un día se opera 24/7 sin auto_loop, restaurar valores comentados.
    # "apuestas-catchup": "*/30 * * * *",
    # "apuestas-deep-analysis": "0 * * * *",
    # "apuestas-capture-closing-lines": "*/10 * * * *",
    # "apuestas-live-scores": "*/15 12-23 * * *",
    # "apuestas-steam-watcher": "0 */2 * * *",
    # "apuestas-alert-cleanup": "0 */6 * * *",
    # Schedules independientes (no consumen OddsAPI) — pueden quedar activos
    "apuestas-drift-monitor": "0 3 * * *",
    "apuestas-rag-preembed": "0 2 * * *",
    "apuestas-enrich-features": "0 3 * * *",
}


def build_schedule(flow_name: str) -> object | None:
    """Devuelve CronSchedule para el flow o None si Prefect no disponible."""
    if CronSchedule is None:
        return None
    cron = SCHEDULES.get(flow_name)
    if cron is None:
        return None
    return CronSchedule(cron=cron, timezone="UTC")


__all__ = ["SCHEDULES", "build_schedule"]
