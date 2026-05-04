"""Fase 4.2 — CLV consensus multi-sharp.

El bot hoy trackea CLV vs **solo Pinnacle** (`betting/clv.py::get_closing_odds`).
Los pros usan **consensus CLV** de múltiples sharp books (Pinnacle + Circa +
Betfair Exchange + Matchbook + SBOBet) → reduce varianza del CLV signal ~40%
y evita outliers (un Pinnacle mal-priced un día no contamina el CLV).

Pesos por book (basado en sharpness empírica):
  - Pinnacle: 0.50  (referencia canónica)
  - Betfair: 0.20   (exchange, implícitas de mercado)
  - Circa: 0.15     (sharp Nevada)
  - Matchbook: 0.10 (exchange backup)
  - SBOBet: 0.05    (sharp asiático)

Si falta un book en la ventana, los pesos se renormalizan entre los presentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from apuestas.betting.devig import shin
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Weights por book — suman 1.0 cuando todos presentes
SHARP_BOOK_WEIGHTS: dict[str, float] = {
    "pinnacle": 0.50,
    "betfair": 0.20,
    "betfair_exchange": 0.20,  # alias
    "circa": 0.15,
    "matchbook": 0.10,
    "sbobet": 0.05,
    "pinnacle_close": 0.50,  # alias cuando viene de CSV histórico
}


@dataclass(slots=True, frozen=True)
class ConsensusClosing:
    match_id: int
    market: str
    outcome: str
    line: float | None
    consensus_odds: float
    p_fair_consensus: float  # probability devigged consensus
    n_books_contributing: int
    books_used: list[str]
    ts: datetime


async def get_consensus_closing_odds(
    *,
    match_id: int,
    market: str,
    outcome: str,
    line: float | None = None,
    window_minutes_before_start: int = 5,
) -> ConsensusClosing | None:
    """Retorna closing consensus odds ponderado por sharpness.

    Similar a `clv.get_closing_odds` pero promedia sobre los sharp books
    disponibles en la ventana en lugar de solo Pinnacle.
    """
    async with session_scope() as session:
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

        rows = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT ON (bookmaker) bookmaker, odds, ts
                    FROM odds_history
                    WHERE match_id = :mid
                      AND market = :mk
                      AND outcome = :oc
                      AND (line = :line OR (:line IS NULL AND line IS NULL))
                      AND ts BETWEEN :ws AND :start
                      AND bookmaker = ANY(:books)
                    ORDER BY bookmaker, ts DESC
                    """
                ),
                {
                    "mid": match_id,
                    "mk": market,
                    "oc": outcome,
                    "line": line,
                    "ws": window_start,
                    "start": start,
                    "books": list(SHARP_BOOK_WEIGHTS.keys()),
                },
            )
        ).all()

    if not rows:
        return None

    # Weighted average of implied probabilities (NOT odds — probabilities
    # are additive when weighted, odds are not).
    total_weight = 0.0
    weighted_prob = 0.0
    books_used: list[str] = []
    for r in rows:
        weight = SHARP_BOOK_WEIGHTS.get(r.bookmaker, 0.0)
        if weight == 0:
            continue
        implied_prob = 1.0 / float(r.odds)
        weighted_prob += weight * implied_prob
        total_weight += weight
        books_used.append(r.bookmaker)

    if total_weight < 1e-9:
        return None

    p_consensus_raw = weighted_prob / total_weight
    consensus_odds = 1.0 / p_consensus_raw if p_consensus_raw > 0 else float("inf")

    # Devig: intentar Shin con el par de outcomes si disponibles
    # Para simplicidad aquí asumimos outcome binario (h2h NBA/tennis).
    try:
        # Fetch opposite outcome odds
        async with session_scope() as session:
            opposite_row = (
                await session.execute(
                    text(
                        """
                        SELECT bookmaker, odds FROM odds_history
                        WHERE match_id = :mid AND market = :mk AND outcome != :oc
                          AND (line = :line OR (:line IS NULL AND line IS NULL))
                          AND ts BETWEEN :ws AND :start
                          AND bookmaker = ANY(:books)
                        ORDER BY ts DESC LIMIT 1
                        """
                    ),
                    {
                        "mid": match_id,
                        "mk": market,
                        "oc": outcome,
                        "line": line,
                        "ws": window_start,
                        "start": start,
                        "books": list(SHARP_BOOK_WEIGHTS.keys()),
                    },
                )
            ).first()

        if opposite_row is not None:
            # Devig Shin sobre el par (target, opposite)
            opp_odds = float(opposite_row.odds)
            pair_probs = shin([consensus_odds, opp_odds])
            p_fair = float(pair_probs[0])
        else:
            p_fair = p_consensus_raw
    except Exception as exc:
        logger.debug("clv_consensus.shin_fail", error=str(exc)[:80])
        p_fair = p_consensus_raw

    return ConsensusClosing(
        match_id=match_id,
        market=market,
        outcome=outcome,
        line=line,
        consensus_odds=consensus_odds,
        p_fair_consensus=p_fair,
        n_books_contributing=len(books_used),
        books_used=books_used,
        ts=start,
    )


async def compute_clv_vs_consensus(
    bet_id: int,
) -> dict[str, Any] | None:
    """Calcula CLV de una bet vs consensus closing (no solo Pinnacle).

    Reemplazo más robusto que `reconcile_bet_clv` original que usa solo Pinnacle.
    """
    async with session_scope() as session:
        bet = (
            await session.execute(
                text(
                    """
                    SELECT b.id, b.match_id, b.market, b.outcome, b.line,
                           b.odds_placed, m.start_time
                    FROM bets b
                    JOIN matches m ON m.id = b.match_id
                    WHERE b.id = :bid
                    """
                ),
                {"bid": bet_id},
            )
        ).first()

    if bet is None:
        return None

    closing = await get_consensus_closing_odds(
        match_id=bet.match_id,
        market=bet.market,
        outcome=bet.outcome,
        line=float(bet.line) if bet.line is not None else None,
    )
    if closing is None:
        return None

    odds_placed = float(bet.odds_placed)
    # CLV = (p_fair_closing × odds_placed) - 1 = fair EV implícita del pick
    clv_consensus = closing.p_fair_consensus * odds_placed - 1.0

    return {
        "bet_id": bet_id,
        "clv_consensus": clv_consensus,
        "consensus_odds": closing.consensus_odds,
        "p_fair_consensus": closing.p_fair_consensus,
        "books_used": closing.books_used,
        "n_books": closing.n_books_contributing,
    }
