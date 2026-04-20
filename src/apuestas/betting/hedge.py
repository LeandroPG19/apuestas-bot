"""Hedging y cash-out rules — sugerencias, NO ejecución automática.

Blueprint §17.13: el bot sugiere hedges cuando:
- Bet pregame +EV y la línea se movió fuerte a favor → lock-in profit via
  bet en lado opuesto en otra casa.
- Parlay 3+ legs con N-1 acertadas → hedge en última leg para profit
  garantizado.

Sólo sugiere; el usuario decide. Registra decisiones en decision_log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.betting.ev import compute_clv
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class HedgeSuggestion:
    bet_id: int
    original_stake: float
    original_odds: float
    original_outcome: str
    opposite_outcome: str
    opposite_odds: float
    opposite_bookmaker: str
    # Cantidades calculadas
    hedge_stake: float
    guaranteed_profit_if_original_wins: float
    guaranteed_profit_if_original_loses: float
    min_profit_locked: float
    confidence: str  # "strong"|"moderate"|"weak"
    reason: str


def compute_hedge_stake(
    *,
    original_stake: float,
    original_odds: float,
    opposite_odds: float,
    objective: str = "equal",
) -> tuple[float, float, float]:
    """Calcula stake hedge y profit en ambos escenarios.

    Si objective='equal': iguala payout en ambos lados (profit seguro).
    Si objective='profit_lock': hedge parcial preservando upside.

    Returns: (hedge_stake, profit_if_original_wins, profit_if_original_loses)
    """
    if objective == "equal":
        # Total payout original si gana = stake * odds_original
        # Total payout hedge si gana = hedge_stake * odds_opposite
        # Ecualizar - stake inicial y hedge_stake
        # P_win_orig = stake * (odds_orig - 1) - hedge_stake
        # P_win_hedge = hedge_stake * (odds_opp - 1) - stake
        # Iguales: stake * (odds_orig - 1) - hedge = hedge * (odds_opp - 1) - stake
        # → stake * odds_orig = hedge * odds_opp
        hedge_stake = original_stake * original_odds / opposite_odds
    else:
        # Profit lock parcial: solo hedge 50%
        hedge_stake = original_stake * 0.5

    profit_if_original = original_stake * (original_odds - 1) - hedge_stake
    profit_if_opposite = hedge_stake * (opposite_odds - 1) - original_stake

    return hedge_stake, profit_if_original, profit_if_opposite


def compute_parlay_hedge(
    *,
    legs_completed: int,
    total_legs: int,
    parlay_stake: float,
    accumulated_multiplier: float,
    remaining_odds: float,
    hedge_book_opposite_odds: float,
) -> HedgeSuggestion | None:
    """Para parlays con N-1 legs acertadas: hedge en última leg.

    Solo sugerido si legs_completed = total_legs - 1 (solo falta una).
    """
    if legs_completed != total_legs - 1:
        return None
    # Payout si parlay completa = parlay_stake * accumulated_multiplier * remaining_odds
    potential_win = parlay_stake * accumulated_multiplier * remaining_odds
    # Hedge en lado opuesto de última leg
    # Iguala pagos: hedge * hedge_book_opposite_odds = potential_win - hedge
    hedge_stake = potential_win / (1 + hedge_book_opposite_odds)
    profit_if_hit = potential_win - hedge_stake - parlay_stake
    profit_if_miss = hedge_stake * (hedge_book_opposite_odds - 1) - parlay_stake

    min_profit = min(profit_if_hit, profit_if_miss)
    if min_profit <= 0:
        return None

    return HedgeSuggestion(
        bet_id=0,
        original_stake=parlay_stake,
        original_odds=accumulated_multiplier * remaining_odds,
        original_outcome="parlay_final_leg",
        opposite_outcome="parlay_final_leg_opposite",
        opposite_odds=hedge_book_opposite_odds,
        opposite_bookmaker="tbd",
        hedge_stake=hedge_stake,
        guaranteed_profit_if_original_wins=profit_if_hit,
        guaranteed_profit_if_original_loses=profit_if_miss,
        min_profit_locked=min_profit,
        confidence="strong" if min_profit / parlay_stake > 0.5 else "moderate",
        reason=f"parlay_{total_legs}legs_{legs_completed}_hit_hedge_final",
    )


async def find_hedge_candidates(
    *,
    min_clv_improvement: float = 0.05,
    min_time_before_start_hours: float = 0.5,
) -> list[HedgeSuggestion]:
    """Escanea bets pending buscando oportunidades de hedge.

    Criterio: bet tomada con CLV estimado (vs Pinnacle actual) > umbral,
    y el evento aún no empieza.
    """
    now = datetime.now(tz=UTC)
    min_start = now + timedelta(hours=min_time_before_start_hours)

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT b.id, b.match_id, b.market, b.outcome, b.odds_placed,
                       b.stake_units, m.start_time
                FROM bets b
                JOIN matches m ON m.id = b.match_id
                WHERE b.status = 'pending'
                  AND b.is_paper = false
                  AND m.start_time > :min_start
                ORDER BY m.start_time ASC
                LIMIT 50
                """
            ),
            {"min_start": min_start},
        )
        pending = [dict(r._mapping) for r in result.all()]

    if not pending:
        return []

    suggestions: list[HedgeSuggestion] = []
    for bet in pending:
        current_pinnacle = await _fetch_current_sharp_odds(
            match_id=int(bet["match_id"]),
            market=str(bet["market"]),
            outcome=str(bet["outcome"]),
        )
        if current_pinnacle is None:
            continue

        # CLV implícito actual
        clv_now = compute_clv(
            odds_placed=float(bet["odds_placed"]),
            closing_odds=current_pinnacle,
        )
        if clv_now < min_clv_improvement:
            continue

        # Buscar cuota opposite en soft books
        opposite_odds, opposite_book = await _fetch_opposite_outcome_odds(
            match_id=int(bet["match_id"]),
            market=str(bet["market"]),
            outcome=str(bet["outcome"]),
        )
        if opposite_odds is None or opposite_book is None:
            continue

        hedge_stake, p_orig, p_opp = compute_hedge_stake(
            original_stake=float(bet["stake_units"]),
            original_odds=float(bet["odds_placed"]),
            opposite_odds=opposite_odds,
            objective="equal",
        )
        min_profit = min(p_orig, p_opp)
        if min_profit <= 0:
            continue

        suggestions.append(
            HedgeSuggestion(
                bet_id=int(bet["id"]),
                original_stake=float(bet["stake_units"]),
                original_odds=float(bet["odds_placed"]),
                original_outcome=str(bet["outcome"]),
                opposite_outcome=_opposite_outcome(str(bet["outcome"]), str(bet["market"])),
                opposite_odds=opposite_odds,
                opposite_bookmaker=opposite_book,
                hedge_stake=hedge_stake,
                guaranteed_profit_if_original_wins=p_orig,
                guaranteed_profit_if_original_loses=p_opp,
                min_profit_locked=min_profit,
                confidence="strong" if clv_now > 0.10 else "moderate",
                reason=f"pregame_line_moved_clv_{clv_now:.2%}",
            )
        )

    logger.info("hedge.candidates_found", n=len(suggestions))
    return suggestions


async def _fetch_current_sharp_odds(*, match_id: int, market: str, outcome: str) -> float | None:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT odds FROM odds_history
                WHERE match_id = :match_id
                  AND market = :market
                  AND outcome = :outcome
                  AND bookmaker IN ('pinnacle', 'circa', 'betfair')
                  AND ts >= NOW() - INTERVAL '30 minutes'
                ORDER BY ts DESC
                LIMIT 1
                """
            ),
            {"match_id": match_id, "market": market, "outcome": outcome},
        )
        row = result.first()
    return float(row.odds) if row else None


async def _fetch_opposite_outcome_odds(
    *, match_id: int, market: str, outcome: str
) -> tuple[float | None, str | None]:
    opp = _opposite_outcome(outcome, market)
    if opp is None:
        return None, None
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT bookmaker, odds FROM odds_history
                WHERE match_id = :match_id
                  AND market = :market
                  AND outcome = :outcome
                  AND bookmaker NOT IN ('pinnacle', 'circa', 'betfair')
                  AND ts >= NOW() - INTERVAL '15 minutes'
                ORDER BY odds DESC
                LIMIT 1
                """
            ),
            {"match_id": match_id, "market": market, "outcome": opp},
        )
        row = result.first()
    if row is None:
        return None, None
    return float(row.odds), str(row.bookmaker)


def _opposite_outcome(outcome: str, market: str) -> str | None:
    """Mapeo simple de outcomes opuestos por mercado."""
    mapping = {
        "home": "away",
        "away": "home",
        "over": "under",
        "under": "over",
        "yes": "no",
        "no": "yes",
    }
    low = outcome.lower()
    if low in mapping:
        return mapping[low]
    # 1X2: sin opposite simple; hedge parcial en el draw
    return None
