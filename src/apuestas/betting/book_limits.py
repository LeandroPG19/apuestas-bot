"""Fase 1.3 — Book limits tracking dinámico.

Observa cuánto stake el book acepta vs rechaza por cuenta del usuario. Cuando
el book restringe cuenta ganadora (común en DK/FD/MGM post 2-4 semanas), el
bot detecta transición `full → restricted → closed` y reduce o omite picks
de ese book.

API principal:
    record_bet_placed(bookmaker, stake_requested, stake_accepted, ts)
    record_bet_rejected(bookmaker, stake_requested, reason, ts)
    get_limit_status(bookmaker) -> {status, max_stake, last_rejected_stake}
    adjusted_kelly_cap(bookmaker, kelly_frac) -> kelly_frac_ajustado
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

LimitStatus = Literal["full", "restricted", "closed"]


async def get_limit_status(bookmaker: str) -> dict[str, Any]:
    """Retorna `{status, max_accepted, last_rejected, notes}` del book."""
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT bookmaker, limit_status, max_accepted_stake_usd,
                           last_rejected_stake_usd, last_rejected_at, notes,
                           updated_at
                    FROM book_limits_per_user
                    WHERE bookmaker = :bk
                    """
                ),
                {"bk": bookmaker},
            )
        ).first()
    if row is None:
        # Book desconocido → asume full
        return {
            "bookmaker": bookmaker,
            "status": "full",
            "max_accepted": None,
            "last_rejected": None,
            "last_rejected_at": None,
            "notes": [],
        }
    return {
        "bookmaker": row.bookmaker,
        "status": row.limit_status,
        "max_accepted": float(row.max_accepted_stake_usd)
        if row.max_accepted_stake_usd is not None
        else None,
        "last_rejected": float(row.last_rejected_stake_usd)
        if row.last_rejected_stake_usd is not None
        else None,
        "last_rejected_at": row.last_rejected_at,
        "notes": list(row.notes or []),
    }


async def record_bet_placed(
    bookmaker: str,
    stake_requested: float,
    stake_accepted: float,
    *,
    ts: datetime | None = None,
) -> dict[str, Any]:
    """Registra una apuesta aceptada. Si stake_accepted < stake_requested → restricted."""
    ts = ts or datetime.now(tz=UTC)
    new_status: LimitStatus
    note_type: str
    if stake_accepted < stake_requested * 0.9:
        new_status = "restricted"
        note_type = "partial_fill"
    else:
        new_status = "full"
        note_type = "full_fill"

    note = {
        "ts": ts.isoformat(),
        "type": note_type,
        "stake_requested": stake_requested,
        "stake_accepted": stake_accepted,
    }

    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO book_limits_per_user
                  (bookmaker, limit_status, max_accepted_stake_usd, notes)
                VALUES (:bk, :st, :max_s, jsonb_build_array(:note::jsonb))
                ON CONFLICT (bookmaker) DO UPDATE
                SET limit_status = EXCLUDED.limit_status,
                    max_accepted_stake_usd = GREATEST(
                        book_limits_per_user.max_accepted_stake_usd,
                        EXCLUDED.max_accepted_stake_usd
                    ),
                    notes = book_limits_per_user.notes ||
                        jsonb_build_array(:note::jsonb),
                    updated_at = now()
                """
            ),
            {
                "bk": bookmaker,
                "st": new_status,
                "max_s": stake_accepted,
                "note": _json_str(note),
            },
        )

    logger.info(
        "book_limits.bet_placed",
        bookmaker=bookmaker,
        status=new_status,
        stake_accepted=stake_accepted,
        stake_requested=stake_requested,
    )
    return await get_limit_status(bookmaker)


async def record_bet_rejected(
    bookmaker: str,
    stake_requested: float,
    reason: str,
    *,
    ts: datetime | None = None,
) -> dict[str, Any]:
    """Registra apuesta rechazada. Si reason='limit' → closed (cuenta cerrada)."""
    ts = ts or datetime.now(tz=UTC)
    new_status: LimitStatus = "closed" if "closed" in reason.lower() else "restricted"
    note = {
        "ts": ts.isoformat(),
        "type": "rejected",
        "stake_requested": stake_requested,
        "reason": reason[:200],
    }

    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO book_limits_per_user
                  (bookmaker, limit_status, last_rejected_stake_usd,
                   last_rejected_at, notes)
                VALUES (:bk, :st, :rej, :ts, jsonb_build_array(:note::jsonb))
                ON CONFLICT (bookmaker) DO UPDATE
                SET limit_status = EXCLUDED.limit_status,
                    last_rejected_stake_usd = EXCLUDED.last_rejected_stake_usd,
                    last_rejected_at = EXCLUDED.last_rejected_at,
                    notes = book_limits_per_user.notes ||
                        jsonb_build_array(:note::jsonb),
                    updated_at = now()
                """
            ),
            {
                "bk": bookmaker,
                "st": new_status,
                "rej": stake_requested,
                "ts": ts,
                "note": _json_str(note),
            },
        )

    logger.warning(
        "book_limits.bet_rejected",
        bookmaker=bookmaker,
        status=new_status,
        reason=reason[:80],
    )
    return await get_limit_status(bookmaker)


async def adjusted_kelly_cap(bookmaker: str, kelly_frac: float) -> float:
    """Ajusta Kelly por status del book.

    - full: sin cambio.
    - restricted: Kelly × 0.5 (menos stake hasta confirmar restricción).
    - closed: Kelly × 0.0 (skip pick en ese book).
    """
    status = await get_limit_status(bookmaker)
    match status["status"]:
        case "closed":
            return 0.0
        case "restricted":
            return kelly_frac * 0.5
        case _:
            return kelly_frac


def _json_str(obj: dict[str, Any]) -> str:
    """Serializa dict simple a JSON string para jsonb_build_array."""
    import json

    return json.dumps(obj, ensure_ascii=False)
