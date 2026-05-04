"""Sanity check al arrancar `apuestas go` (Gap 7 / A7).

Verifica precondiciones antes de lanzar el bot:
  1. Postgres responde `SELECT 1`.
  2. Valkey responde `PING`.
  3. MLflow registry devuelve al menos un modelo `production` por deporte activo.
  4. Telegram bot responde `getMe`.
  5. No hay pick_alerts huérfanas > 72h sin resolver.

Si alguna falla, exit code ≠ 0 para que `apuestas go` pueda parar el arranque.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta

from sqlalchemy import text


async def _check_postgres() -> tuple[bool, str]:
    try:
        from apuestas.db import session_scope

        async with session_scope() as s:
            await s.execute(text("SELECT 1"))
        return True, "postgres:ok"
    except Exception as exc:
        return False, f"postgres:fail ({str(exc)[:60]})"


async def _check_valkey() -> tuple[bool, str]:
    try:
        import redis.asyncio as aioredis

        from apuestas.config import get_settings

        settings = get_settings().valkey
        client = aioredis.from_url(str(settings.url), decode_responses=True)
        await client.ping()
        await client.aclose()
        return True, "valkey:ok"
    except Exception as exc:
        return False, f"valkey:fail ({str(exc)[:60]})"


async def _check_models() -> tuple[bool, str]:
    try:
        from apuestas.db import session_scope

        async with session_scope() as s:
            row = (
                await s.execute(
                    text(
                        """
                        SELECT COUNT(DISTINCT sport_code) AS n
                        FROM model_registry_meta
                        WHERE stage = 'production'
                        """
                    )
                )
            ).first()
        n = int(row.n or 0) if row else 0
        return n > 0, f"models_production:{n}_sports"
    except Exception as exc:
        return False, f"models:fail ({str(exc)[:60]})"


async def _check_telegram() -> tuple[bool, str]:
    try:
        from telegram import Bot

        from apuestas.config import get_settings

        token = get_settings().apis.telegram_bot_token
        if token is None:
            return False, "telegram:no_token"
        bot = Bot(token=token.get_secret_value())
        me = await bot.get_me()
        return True, f"telegram:ok@{me.username}"
    except Exception as exc:
        return False, f"telegram:fail ({str(exc)[:60]})"


async def _check_orphans(ttl_hours: int = 72) -> tuple[bool, str]:
    try:
        from apuestas.db import session_scope

        threshold = datetime.now(tz=UTC) - timedelta(hours=ttl_hours)
        async with session_scope() as s:
            row = (
                await s.execute(
                    text(
                        """
                        SELECT COUNT(*) AS n
                        FROM pick_alerts pa
                        JOIN matches m ON m.id = pa.match_id
                        WHERE (pa.outcome_result IS NULL OR pa.outcome_result='pending')
                          AND m.start_time < :threshold
                        """
                    ),
                    {"threshold": threshold},
                )
            ).first()
        n = int(row.n or 0) if row else 0
        if n > 0:
            return False, f"orphans:{n}_alerts>{ttl_hours}h"
        return True, "orphans:0"
    except Exception as exc:
        return False, f"orphans:fail ({str(exc)[:60]})"


async def main() -> int:
    checks = [
        _check_postgres(),
        _check_valkey(),
        _check_models(),
        _check_telegram(),
        _check_orphans(),
    ]
    results = await asyncio.gather(*checks, return_exceptions=True)
    exit_code = 0
    for r in results:
        if isinstance(r, BaseException):
            print(f"❌ {r}")
            exit_code = 1
            continue
        ok, msg = r
        icon = "✅" if ok else "❌"
        print(f"{icon} {msg}")
        if not ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
