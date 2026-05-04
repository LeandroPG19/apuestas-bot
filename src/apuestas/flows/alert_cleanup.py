"""TTL/expiration de pick_alerts abandonadas (plan §5.4).

Si un match termina pero `live_scores` falla persistentemente (rate-limit,
bug, fuente caída), su `pick_alerts.outcome_result` queda NULL y bloquea
el unique index parcial `uq_pick_alerts_identity`, impidiendo nuevas
emisiones legítimas para la misma identidad en futuros partidos.

Este flow marca como `expired` las alertas cuyos matches ya deberían
haber acabado hace más de `ttl_hours`. Schedule Prefect: cada 6 horas.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from prefect import flow
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def expire_stale_alerts(*, ttl_hours: int = 96) -> int:
    """Marca `outcome_result='expired'` a alertas huérfanas.

    Criterio: `pick_alerts.outcome_result` NULL/pending Y el match empezó
    hace más de `ttl_hours` horas. Un partido normal dura ≤5 h; 96 h
    (4 días) da margen generoso para backfill y eventos suspendidos.

    Retorna: número de alertas expiradas.
    """
    threshold = datetime.now(tz=UTC) - timedelta(hours=ttl_hours)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                UPDATE pick_alerts pa
                SET outcome_result = 'expired',
                    result_settled_at = now(),
                    notes = COALESCE(pa.notes, '')
                            || ' [auto-expired: match_end_time + ttl exceeded]'
                FROM matches m
                WHERE pa.match_id = m.id
                  AND (pa.outcome_result IS NULL OR pa.outcome_result = 'pending')
                  AND m.start_time < :threshold
                """
            ),
            {"threshold": threshold},
        )
        count = int(result.rowcount or 0)
    logger.info("alert_cleanup.expired", n=count, ttl_hours=ttl_hours)
    return count


async def cancel_orphan_matches(*, min_age_hours: int = 24) -> int:
    """Cancela matches sin cobertura de odds tras N horas desde su creación.

    Pinnacle scraper trae miles de matches de ligas menores (Cyprus B-League,
    Marroquí, Greek U21, Maktown Flyers...) que ningún book real cubre.
    Estos matches "fantasma" se acumulan en `matches.status='scheduled'` y
    aunque el detector los filtra (Fix #22, min_books=5), siguen ocupando
    espacio y degradan queries. Marcar como cancelled tras 24h sin odds
    los excluye del pipeline sin perder data ya ingresada.

    Criterio: `status='scheduled'` Y `start_time` próximas 7 días Y CERO
    rows en `odds_history` jamás (no solo recientes — si nunca tuvo odds
    es match huérfano). Min age 24h evita race con ingesters lentos.

    Retorna: número de matches cancelados.
    """
    threshold = datetime.now(tz=UTC) - timedelta(hours=min_age_hours)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                UPDATE matches m
                SET status = 'cancelled'
                WHERE m.status = 'scheduled'
                  AND m.start_time > NOW() - INTERVAL '7 days'
                  AND m.start_time < NOW() + INTERVAL '7 days'
                  AND NOT EXISTS (
                      SELECT 1 FROM odds_history oh WHERE oh.match_id = m.id LIMIT 1
                  )
                  AND EXISTS (
                      -- Solo si el match ya tiene >24h de existencia (evita race)
                      SELECT 1 FROM matches m2
                      WHERE m2.id = m.id
                  )
                """
            ),
            {"threshold": threshold},
        )
        count = int(result.rowcount or 0)
    logger.info("alert_cleanup.orphan_matches_cancelled", n=count)
    return count


@flow(name="apuestas-alert-cleanup", log_prints=True)
async def alert_cleanup_flow(*, ttl_hours: int = 96) -> dict[str, int]:
    n_alerts = await expire_stale_alerts(ttl_hours=ttl_hours)
    n_orphans = await cancel_orphan_matches()
    return {"expired_alerts": n_alerts, "cancelled_orphans": n_orphans, "ttl_hours": ttl_hours}


if __name__ == "__main__":
    import asyncio

    asyncio.run(alert_cleanup_flow())
