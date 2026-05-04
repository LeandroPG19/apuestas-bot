"""Flow proactivo de detección de steam moves + pricing errors (Sprint 4e).

Corre `odds_spike.run_all_detectors()` y, para cada `SpikeAlert` detectado,
emite notificación al canal Telegram de steam si el alert no fue notificado
antes (dedupe vía `bot_state.steam_notified_{hash}` con TTL 6h).

Frecuencia sugerida: cada 2 horas (Prefect schedule). En on-demand el
usuario puede invocarlo con `python -m apuestas.flows.steam_watcher`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime

from prefect import flow
from sqlalchemy import text

from apuestas.betting.odds_spike import SpikeAlert, run_all_detectors
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def _alert_hash(alert: SpikeAlert) -> str:
    payload = f"{alert.match_id}:{alert.market}:{alert.outcome}:{alert.bookmaker}:{alert.tag}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


async def _already_notified(session: object, hash_: str) -> bool:
    row = (
        await session.execute(  # type: ignore[attr-defined]
            text("SELECT value FROM bot_state WHERE key = :k"),
            {"k": f"steam_notified_{hash_}"},
        )
    ).first()
    return row is not None


async def _mark_notified(session: object, hash_: str) -> None:
    payload = json.dumps({"notified_at": datetime.now(tz=UTC).isoformat(), "ttl_hours": 6})
    await session.execute(  # type: ignore[attr-defined]
        text(
            """
            INSERT INTO bot_state (key, value, updated_at)
            VALUES (:k, :v, now())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """
        ),
        {"k": f"steam_notified_{hash_}", "v": payload},
    )


def _format_alert(alert: SpikeAlert) -> str:
    icon = {
        "steam_move": "🚂",
        "pricing_error": "💥",
        "soft_line": "🪶",
    }.get(alert.tag, "⚠️")
    return (
        f"{icon} <b>{alert.tag.upper()}</b> · match_id={alert.match_id}\n"
        f"Market: <code>{alert.market}</code> · Outcome: <code>{alert.outcome}</code>\n"
        f"Book: <b>{alert.bookmaker}</b>\n"
        f"Odds: <code>{alert.odds_before:.2f}</code> → "
        f"<code>{alert.odds_after:.2f}</code>  "
        f"(<b>{alert.pct_move:+.1%}</b>)"
    )


async def _notify_channel(alerts: list[SpikeAlert]) -> int:
    from apuestas.config import get_settings

    settings = get_settings()
    token = settings.apis.telegram_bot_token
    channel_id = settings.apis.telegram_channel_id
    if token is None or channel_id is None:
        logger.info("steam_watcher.no_channel_configured")
        return 0

    from telegram import Bot
    from telegram.constants import ParseMode

    bot = Bot(token=token.get_secret_value())
    target: str | int = int(channel_id) if channel_id.lstrip("-").isdigit() else channel_id

    sent = 0
    async with session_scope() as session:
        for alert in alerts:
            h = _alert_hash(alert)
            if await _already_notified(session, h):
                continue
            try:
                await bot.send_message(
                    chat_id=target,
                    text=_format_alert(alert),
                    parse_mode=ParseMode.HTML,
                )
                await _mark_notified(session, h)
                sent += 1
                await asyncio.sleep(0.25)  # batching rate-limit
            except Exception as exc:
                logger.warning("steam_watcher.send_fail", error=str(exc)[:100])
        await session.commit()
    return sent


@flow(name="apuestas-steam-watcher", log_prints=True)
async def steam_watcher_flow() -> dict[str, int]:
    if os.environ.get("ENABLE_STEAM_WATCHER", "true").lower() != "true":
        logger.info("steam_watcher.disabled_by_env")
        return {"skipped": 1, "alerts": 0, "notified": 0}

    alerts = await run_all_detectors()
    logger.info("steam_watcher.alerts_raw", n=len(alerts))
    if not alerts:
        return {"alerts": 0, "notified": 0}
    notified = await _notify_channel(alerts)
    logger.info("steam_watcher.done", alerts=len(alerts), notified=notified)
    return {"alerts": len(alerts), "notified": notified}


if __name__ == "__main__":
    asyncio.run(steam_watcher_flow())
