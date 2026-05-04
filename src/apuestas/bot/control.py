"""Control de ciclo del bot (pausa/reanuda) sin dependencia de bankroll.

Persistencia en la tabla `bot_state` (key/value JSONB-as-text). Se usa
desde Telegram (`/pausar`, `/resumir`), TUI y flows para respetar el
estado global.

Antes vivía en `betting/portfolio.py`, pero al demoler el subsistema
bankroll estas utilidades pasan aquí donde son la responsabilidad
primaria.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def is_bot_paused() -> tuple[bool, str | None]:
    """Devuelve `(paused, reason)` leyendo `bot_state.key='paused'`."""
    async with session_scope() as session:
        result = await session.execute(text("SELECT value FROM bot_state WHERE key = 'paused'"))
        row = result.first()
    if row is None:
        return False, None
    try:
        payload = json.loads(row.value) if row.value else {}
    except (TypeError, ValueError):
        payload = {}
    return bool(payload.get("paused", False)), payload.get("reason")


async def pause_bot(*, reason: str, triggered_by: str = "manual") -> None:
    """Pausa el bot. El motivo se persiste como JSON en `bot_state`."""
    payload = {
        "paused": True,
        "reason": reason,
        "paused_at": datetime.now(tz=UTC).isoformat(),
        "triggered_by": triggered_by,
    }
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES ('paused', :value, NOW())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
                """
            ),
            {"value": json.dumps(payload)},
        )
    logger.warning("bot.paused", reason=reason, triggered_by=triggered_by)


async def resume_bot() -> None:
    payload = {"paused": False, "reason": None, "paused_at": None}
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES ('paused', :value, NOW())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
                """
            ),
            {"value": json.dumps(payload)},
        )
    logger.info("bot.resumed")
