"""Worker LISTEN/NOTIFY para auto-settle tras trigger PG.

Arquitectura:
1. `migrations/0006` añadió trigger AFTER UPDATE que hace `pg_notify` en
   canal `apuestas_match_finished` cuando un match pasa a finished.
2. Este worker mantiene una conexión asyncpg con LISTEN al canal.
3. Al recibir notify, encola match_id y procesa via settle_bets_flow.
4. Fallback: cada 60s polea `settlement_queue` para matches pending.

Arranque:
    python -m apuestas.flows.settle_worker
o vía systemd-user timer (recomendado para no depender de TUI activa).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os

import asyncpg

from apuestas.config import get_settings
from apuestas.flows.settle_bets import settle_bets_flow
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)

_processed: set[int] = set()  # evita re-procesar mismo match en la misma sesión
_processed_lock = asyncio.Lock()


async def _pg_dsn() -> str:
    settings = get_settings()
    sync = str(settings.database.sync_url)
    return sync.replace("postgresql+psycopg://", "postgresql://")


async def _process_match(match_id: int) -> None:
    async with _processed_lock:
        if match_id in _processed:
            return
        _processed.add(match_id)
    logger.info("settle_worker.processing", match_id=match_id)
    try:
        result = await settle_bets_flow(trigger_post_mortem=True)
        logger.info("settle_worker.done", match_id=match_id, result=result)
    except Exception as exc:
        logger.exception("settle_worker.flow_failed", match_id=match_id, error=str(exc))
        async with _processed_lock:
            _processed.discard(match_id)  # permitir retry


async def _handle_notify(conn: asyncpg.Connection, _pid: int, channel: str, payload: str) -> None:
    try:
        data = json.loads(payload)
        match_id = int(data["match_id"])
        logger.info(
            "settle_worker.notify_received",
            channel=channel,
            match_id=match_id,
            home_score=data.get("home_score"),
            away_score=data.get("away_score"),
        )
        asyncio.create_task(_process_match(match_id))
        # Marcar queue como processing
        await conn.execute(
            "UPDATE settlement_queue SET status='processing' WHERE match_id=$1 AND status='pending'",
            match_id,
        )
    except Exception as exc:
        logger.warning("settle_worker.notify_parse_fail", payload=payload, error=str(exc))


async def _poll_pending_queue(conn: asyncpg.Connection) -> int:
    """Fallback: procesa pendientes en settlement_queue cada 60s."""
    rows = await conn.fetch(
        "SELECT id, match_id FROM settlement_queue WHERE status='pending' "
        "ORDER BY created_at ASC LIMIT 20"
    )
    for row in rows:
        asyncio.create_task(_process_match(int(row["match_id"])))
        await conn.execute(
            "UPDATE settlement_queue SET status='processing' WHERE id=$1",
            row["id"],
        )
    return len(rows)


async def run(*, poll_interval: int = 60) -> None:
    """Arranca LISTEN permanente + poll cada N segundos como fallback."""
    configure_logging()
    dsn = await _pg_dsn()
    logger.info("settle_worker.starting", dsn=dsn.split("@")[-1])

    conn = await asyncpg.connect(dsn)
    await conn.add_listener("apuestas_match_finished", _handle_notify)
    logger.info("settle_worker.listening", channel="apuestas_match_finished")

    # Drenar pendientes de arranque
    drained = await _poll_pending_queue(conn)
    if drained:
        logger.info("settle_worker.startup_drain", matches=drained)

    try:
        while True:
            await asyncio.sleep(poll_interval)
            try:
                n = await _poll_pending_queue(conn)
                if n:
                    logger.debug("settle_worker.poll_drain", matches=n)
            except Exception as exc:
                logger.warning("settle_worker.poll_fail", error=str(exc))
    except asyncio.CancelledError:
        logger.info("settle_worker.shutdown")
    finally:
        with contextlib.suppress(Exception):
            await conn.remove_listener("apuestas_match_finished", _handle_notify)
        await conn.close()


def main() -> None:
    _ = os
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
