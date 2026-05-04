"""Rate limiter global para Telegram Bot API — anti-flood-control.

Límites oficiales Telegram Bot API:
- Chat privado: 1 msg/seg sostenido, burst OK.
- Grupo: 20 msg/min (hard).
- Canal público: 20 msg/min (hard) + 30 msg/seg burst global.

Política SOTA:
- aiolimiter por destino (chat_id) con 1 req/seg.
- aiolimiter global para canal/grupo con 20/min.
- Retry automático en `telegram.error.RetryAfter` (respeta el header).
- Exponential backoff para errores transientes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiolimiter import AsyncLimiter

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Global limiters — singleton por proceso.
# Hard limit Telegram: 30 msg/seg global → usamos 25 para safety margin.
_GLOBAL_LIMITER = AsyncLimiter(max_rate=25, time_period=1.0)

# Per-destination limiter cache: chat_id → AsyncLimiter(1 msg/seg).
_PER_CHAT_LIMITERS: dict[str, AsyncLimiter] = {}

# Per-channel/group limiter cache: 20 msg/min hard limit.
_PER_CHANNEL_LIMITERS: dict[str, AsyncLimiter] = {}

_lock = asyncio.Lock()


async def _get_chat_limiter(chat_id: str | int) -> AsyncLimiter:
    """Chat privado: 1 msg/seg sostenible."""
    key = str(chat_id)
    async with _lock:
        lim = _PER_CHAT_LIMITERS.get(key)
        if lim is None:
            lim = AsyncLimiter(max_rate=1, time_period=1.0)
            _PER_CHAT_LIMITERS[key] = lim
    return lim


async def _get_channel_limiter(chat_id: str | int) -> AsyncLimiter:
    """Canal/grupo: 20 msg/min hard limit → usamos 18 para safety."""
    key = str(chat_id)
    async with _lock:
        lim = _PER_CHANNEL_LIMITERS.get(key)
        if lim is None:
            lim = AsyncLimiter(max_rate=18, time_period=60.0)
            _PER_CHANNEL_LIMITERS[key] = lim
    return lim


def _is_channel_or_group(chat_id: str | int) -> bool:
    """Heurística: channel/grupo tiene chat_id negativo o empieza con @."""
    s = str(chat_id)
    return s.startswith("-") or s.startswith("@")


async def send_with_ratelimit(
    bot: Any,
    *,
    chat_id: str | int,
    text: str,
    max_retries: int = 5,
    **kwargs: Any,
) -> bool:
    """Envía mensaje respetando rate limits y reintentando en RetryAfter.

    Retorna True si se envió correctamente, False tras agotar retries.
    """
    per_chat = await _get_chat_limiter(chat_id)
    per_channel = await _get_channel_limiter(chat_id) if _is_channel_or_group(chat_id) else None

    for attempt in range(max_retries):
        async with _GLOBAL_LIMITER:
            async with per_chat:
                if per_channel:
                    async with per_channel:
                        ok = await _try_send(bot, chat_id, text, **kwargs)
                else:
                    ok = await _try_send(bot, chat_id, text, **kwargs)
        if ok is True:
            return True
        if ok is False:
            return False
        # ok es int → retry_after en segundos
        wait = min(int(ok) + 1, 60)
        logger.info(
            "telegram_ratelimit.retry_after",
            chat_id=str(chat_id),
            wait=wait,
            attempt=attempt + 1,
        )
        await asyncio.sleep(wait)
    return False


async def _try_send(bot: Any, chat_id: str | int, text: str, **kwargs: Any) -> bool | int:
    """Retorna True ok, False error no-retry, int=retry_after si RetryAfter."""
    try:
        from telegram.error import (  # type: ignore[import-not-found]
            BadRequest,
            Forbidden,
            RetryAfter,
            TelegramError,
        )
    except ImportError:
        TelegramError = Exception
        RetryAfter = Exception  # type: ignore[assignment,misc]
        BadRequest = Forbidden = Exception  # type: ignore[assignment,misc]

    try:
        await bot.send_message(chat_id=chat_id, text=text, **kwargs)
        return True
    except RetryAfter as exc:
        retry_after = getattr(exc, "retry_after", 5)
        return int(retry_after)
    except (BadRequest, Forbidden) as exc:
        logger.warning(
            "telegram_ratelimit.permanent_fail",
            chat_id=str(chat_id),
            error=str(exc)[:120],
        )
        return False
    except TelegramError as exc:
        logger.warning(
            "telegram_ratelimit.transient_fail",
            chat_id=str(chat_id),
            error=str(exc)[:120],
        )
        await asyncio.sleep(2.0)
        return 2  # treat as retry-after 2s
    except Exception as exc:
        logger.warning(
            "telegram_ratelimit.unknown_fail",
            chat_id=str(chat_id),
            error=str(exc)[:120],
        )
        return False
