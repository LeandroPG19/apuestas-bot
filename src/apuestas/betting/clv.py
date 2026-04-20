"""Closing Line Value — capture, computo, reconciliación con bets abiertas.

Blueprint §11: CLV es la métrica real de skill (Buchdahl 2023). Se computa
post-partido contra el closing odds de Pinnacle (o fallback sharp book).

Flujo:
1. Job `capture_closing_lines` identifica bets con status=pending y
   start_time < now() - 2h → pendientes de CLV.
2. Por cada una: fetch odds_history del closing window (T-5min a T).
3. Si Pinnacle disponible, usar Pinnacle; si no, usar el promedio de
   los sharp books.
4. Update bets.closing_line + bets.clv. NUNCA sobrescribir closing_line
   si ya está asignado (audit trail inmutable).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import text

from apuestas.betting.ev import compute_clv
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ClosingLineSnapshot:
    bookmaker: str
    odds: float
    ts: datetime


@dataclass(slots=True)
class CLVResult:
    bet_id: int
    closing_line: float | None
    clv: float | None
    source_bookmaker: str | None
    reconciled_at: datetime


async def pending_clv_bets(
    *,
    hours_since_start: int = 2,
    limit: int = 100,
    max_age_days: int = 14,
) -> list[dict[str, object]]:
    """Bets con status=pending, evento ya terminado, sin CLV aún."""
    cutoff_new = datetime.now(tz=UTC) - timedelta(hours=hours_since_start)
    cutoff_old = datetime.now(tz=UTC) - timedelta(days=max_age_days)

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT b.id, b.match_id, b.market, b.outcome, b.line,
                       b.odds_placed, b.placed_at,
                       m.start_time, m.external_id
                FROM bets b
                JOIN matches m ON m.id = b.match_id
                WHERE b.closing_line IS NULL
                  AND m.start_time < :cutoff_new
                  AND m.start_time > :cutoff_old
                ORDER BY m.start_time DESC
                LIMIT :limit
                """
            ),
            {"cutoff_new": cutoff_new, "cutoff_old": cutoff_old, "limit": limit},
        )
        rows = [dict(r._mapping) for r in result.all()]
    return rows


async def get_closing_odds(
    *,
    match_id: int,
    market: str,
    outcome: str,
    line: float | None = None,
    window_minutes_before_start: int = 5,
) -> ClosingLineSnapshot | None:
    """Busca odds más cercanas al start_time del match, preferencia sharp books."""
    async with session_scope() as session:
        # Rango: T-window a T (inicio de evento)
        match_row = (
            await session.execute(
                text("SELECT start_time FROM matches WHERE id = :id"),
                {"id": match_id},
            )
        ).first()
        if match_row is None:
            return None
        start = match_row.start_time
        window_start = start - timedelta(minutes=window_minutes_before_start)

        line_filter = "AND (line = :line OR (:line IS NULL AND line IS NULL))"
        result = await session.execute(
            text(
                f"""
                SELECT bookmaker, odds, ts
                FROM odds_history
                WHERE match_id = :match_id
                  AND market = :market
                  AND outcome = :outcome
                  {line_filter}
                  AND ts BETWEEN :ws AND :start
                ORDER BY
                  CASE WHEN bookmaker = 'pinnacle' THEN 0
                       WHEN bookmaker = 'circa' THEN 1
                       WHEN bookmaker = 'betfair' THEN 2
                       WHEN bookmaker = 'bookmaker' THEN 3
                       ELSE 10 END ASC,
                  ts DESC
                LIMIT 1
                """
            ),
            {
                "match_id": match_id,
                "market": market,
                "outcome": outcome,
                "line": line,
                "ws": window_start,
                "start": start,
            },
        )
        row = result.first()

    if row is None:
        return None
    return ClosingLineSnapshot(
        bookmaker=row.bookmaker,
        odds=float(row.odds),
        ts=row.ts,
    )


async def reconcile_bet_clv(bet_row: dict[str, object]) -> CLVResult:
    """Reconcilia una bet: fetch closing → compute CLV → update bets."""
    bet_id = int(bet_row["id"])
    match_id = int(bet_row["match_id"])
    market = str(bet_row["market"])
    outcome = str(bet_row["outcome"])
    line = bet_row["line"]
    odds_placed = float(bet_row["odds_placed"])

    snapshot = await get_closing_odds(
        match_id=match_id,
        market=market,
        outcome=outcome,
        line=float(line) if line is not None else None,
    )
    if snapshot is None:
        logger.info("clv.no_closing_data", bet_id=bet_id, match_id=match_id)
        return CLVResult(
            bet_id=bet_id,
            closing_line=None,
            clv=None,
            source_bookmaker=None,
            reconciled_at=datetime.now(tz=UTC),
        )

    clv = compute_clv(odds_placed=odds_placed, closing_odds=snapshot.odds)

    async with session_scope() as session:
        # ON UPDATE idempotente: solo actualiza si closing_line aún NULL
        await session.execute(
            text(
                """
                UPDATE bets
                SET closing_line = :closing,
                    clv = :clv
                WHERE id = :bet_id AND closing_line IS NULL
                """
            ),
            {
                "bet_id": bet_id,
                "closing": Decimal(f"{snapshot.odds:.4f}"),
                "clv": Decimal(f"{clv:.6f}"),
            },
        )

    logger.info(
        "clv.reconciled",
        bet_id=bet_id,
        closing=snapshot.odds,
        clv=clv,
        source=snapshot.bookmaker,
    )
    return CLVResult(
        bet_id=bet_id,
        closing_line=snapshot.odds,
        clv=clv,
        source_bookmaker=snapshot.bookmaker,
        reconciled_at=datetime.now(tz=UTC),
    )


async def run_reconciliation(*, batch_size: int = 100) -> dict[str, int]:
    """Job batch para correr al arrancar el stack (catchup) o periódicamente."""
    pending = await pending_clv_bets(limit=batch_size)
    if not pending:
        return {"checked": 0, "reconciled": 0, "no_data": 0}

    reconciled = 0
    no_data = 0
    for row in pending:
        result = await reconcile_bet_clv(row)
        if result.clv is not None:
            reconciled += 1
        else:
            no_data += 1

    logger.info(
        "clv.batch_done",
        checked=len(pending),
        reconciled=reconciled,
        no_data=no_data,
    )
    return {
        "checked": len(pending),
        "reconciled": reconciled,
        "no_data": no_data,
    }


async def clv_summary(*, days: int = 30) -> dict[str, float]:
    """Stats agregadas de CLV de últimos N días. Dashboard §17.14."""
    since = datetime.now(tz=UTC) - timedelta(days=days)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT
                  COUNT(*) AS n,
                  AVG(clv) AS mean_clv,
                  STDDEV_POP(clv) AS std_clv,
                  AVG(CASE WHEN clv > 0 THEN 1.0 ELSE 0.0 END) AS positive_rate,
                  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY clv) AS median_clv,
                  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY clv) AS p95_clv,
                  PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY clv) AS p05_clv
                FROM bets
                WHERE clv IS NOT NULL
                  AND placed_at >= :since
                """
            ),
            {"since": since},
        )
        row = result.first()

    if row is None or row.n == 0:
        return {
            "n": 0,
            "mean_clv": 0.0,
            "std_clv": 0.0,
            "positive_rate": 0.0,
            "median_clv": 0.0,
            "p95_clv": 0.0,
            "p05_clv": 0.0,
        }
    return {
        "n": int(row.n),
        "mean_clv": float(row.mean_clv or 0.0),
        "std_clv": float(row.std_clv or 0.0),
        "positive_rate": float(row.positive_rate or 0.0),
        "median_clv": float(row.median_clv or 0.0),
        "p95_clv": float(row.p95_clv or 0.0),
        "p05_clv": float(row.p05_clv or 0.0),
    }
