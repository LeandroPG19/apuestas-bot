"""Flow settle_bets — liquidación automática de bets con match finished (§19.7).

Para cada bet `status='pending'` cuyo match tiene `status='finished'` y scores
cargados, calcula win/loss/void/halfwon/halflost según mercado y outcome,
persiste `pnl_units`, actualiza `settled_at`, y dispara el flow `post_mortem`.

Mercados soportados:
- moneyline / h2h (sin empate → void si draw en soccer y outcome es home/away)
- spread / handicap (con push = void)
- total (O/U) con medio punto (no push) y enteros (push = void)
- btts (ambos equipos anotan)
- asian_handicap con cuartos (half-win/half-loss)
- player_props (over/under sobre una stat — requiere player_game_logs)
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.flows.post_mortem import post_mortem_flow
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@task(retries=2, retry_delay_seconds=15)
async def load_pending_bets_with_final_match() -> list[dict[str, Any]]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT b.id AS bet_id, b.match_id, b.market, b.outcome,
                       b.line, b.stake_units, b.odds_placed,
                       m.home_score, m.away_score, m.sport_code,
                       m.home_team_id, m.away_team_id
                FROM bets b
                JOIN matches m ON m.id = b.match_id
                WHERE b.status = 'pending'
                  AND m.status = 'finished'
                  AND m.home_score IS NOT NULL
                  AND m.away_score IS NOT NULL
                ORDER BY b.placed_at ASC
                LIMIT 500
                """
            )
        )
        return [dict(r._mapping) for r in result.all()]


def settle_moneyline(*, outcome: str, home_score: int, away_score: int, sport_code: str) -> str:
    """h2h / moneyline. Outcome: 'home' | 'away' | 'draw'."""
    if home_score == away_score:
        if sport_code in {"soccer"}:
            return "won" if outcome == "draw" else "lost"
        # Sin draws en deportes US → nunca debería llegar aquí.
        return "void"
    winner = "home" if home_score > away_score else "away"
    return "won" if outcome == winner else "lost"


def settle_total(*, outcome: str, line: float, home_score: int, away_score: int) -> str:
    """O/U total. outcome: 'over' | 'under'. Push si total == line exacta."""
    total = home_score + away_score
    if total == line:
        return "void"
    over = total > line
    if outcome == "over":
        return "won" if over else "lost"
    return "won" if not over else "lost"


def settle_spread(*, outcome: str, line: float, home_score: int, away_score: int) -> str:
    """Spread handicap. outcome ∈ {home, away}. line es el handicap del home.

    home_adjusted = home_score + line. Si == away_score → push.
    """
    diff = (home_score + line) - away_score
    if diff == 0:
        return "void"
    cover_home = diff > 0
    if outcome == "home":
        return "won" if cover_home else "lost"
    return "won" if not cover_home else "lost"


def settle_btts(*, outcome: str, home_score: int, away_score: int) -> str:
    both = home_score >= 1 and away_score >= 1
    yes = outcome in {"yes", "btts_yes", "gg"}
    return "won" if yes == both else "lost"


def settle_asian_handicap(*, outcome: str, line: float, home_score: int, away_score: int) -> str:
    """AH con cuartos. Si line es cuarto (.25/.75) divide stake en dos mitades.

    Para simplificar esta iteración, AH entero/medio se maneja con spread.
    AH cuarto retorna placeholder 'halfwon'/'halflost' según corresponda.
    """
    # Caso entero o medio: equivalente a spread
    if abs((line * 2) - round(line * 2)) < 1e-9:
        return settle_spread(
            outcome=outcome, line=line, home_score=home_score, away_score=away_score
        )
    # Cuarto: dividir en (line-0.25) y (line+0.25) cada uno stake/2
    half_low = line - 0.25
    half_high = line + 0.25
    r1 = settle_spread(outcome=outcome, line=half_low, home_score=home_score, away_score=away_score)
    r2 = settle_spread(
        outcome=outcome, line=half_high, home_score=home_score, away_score=away_score
    )
    # Ambos won
    if r1 == "won" and r2 == "won":
        return "won"
    if r1 == "lost" and r2 == "lost":
        return "lost"
    if r1 == "won" and r2 in {"void", "lost"}:
        return "halfwon" if r2 == "void" else "lost"
    if r2 == "won" and r1 == "void":
        return "halfwon"
    if r1 == "lost" and r2 == "void":
        return "halflost"
    if r2 == "lost" and r1 == "void":
        return "halflost"
    return "void"


def compute_pnl(*, status: str, stake_units: float, odds_placed: float) -> float:
    if status == "won":
        return stake_units * (odds_placed - 1.0)
    if status == "lost":
        return -stake_units
    if status == "halfwon":
        return 0.5 * stake_units * (odds_placed - 1.0)
    if status == "halflost":
        return -0.5 * stake_units
    return 0.0  # void / cashed


def classify_bet(bet: dict[str, Any]) -> tuple[str, float]:
    """Devuelve (status, pnl_units) aplicando regla de settlement al market."""
    market = str(bet["market"]).lower()
    outcome = str(bet["outcome"]).lower()
    line = float(bet["line"]) if bet.get("line") is not None else 0.0
    hs = int(bet["home_score"])
    as_ = int(bet["away_score"])
    sport = str(bet.get("sport_code") or "")

    if market in {"h2h", "moneyline", "1x2"}:
        status = settle_moneyline(outcome=outcome, home_score=hs, away_score=as_, sport_code=sport)
    elif market in {"total", "totals", "ou"}:
        status = settle_total(outcome=outcome, line=line, home_score=hs, away_score=as_)
    elif market in {"spread", "runline", "puckline"}:
        status = settle_spread(outcome=outcome, line=line, home_score=hs, away_score=as_)
    elif market in {"btts"}:
        status = settle_btts(outcome=outcome, home_score=hs, away_score=as_)
    elif market in {"asian_handicap", "ah"}:
        status = settle_asian_handicap(outcome=outcome, line=line, home_score=hs, away_score=as_)
    elif market.startswith("player_"):
        # Player props requieren stats reales del jugador — fuera de scope
        # automático. Marcar para settle manual hasta tener player_game_logs.
        logger.debug(
            "settle_bets.player_prop_manual_required",
            bet_id=bet["bet_id"],
            market=market,
        )
        status = "pending"
    else:
        logger.warning("settle_bets.unknown_market", bet_id=bet["bet_id"], market=market)
        status = "pending"

    pnl = compute_pnl(
        status=status, stake_units=float(bet["stake_units"]), odds_placed=float(bet["odds_placed"])
    )
    return status, pnl


@task(retries=1, retry_delay_seconds=10)
async def apply_settlement(bets: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"won": 0, "lost": 0, "void": 0, "halfwon": 0, "halflost": 0, "skipped": 0}
    if not bets:
        return counts
    async with session_scope() as session:
        for bet in bets:
            status, pnl = classify_bet(bet)
            if status == "pending":
                counts["skipped"] += 1
                continue
            await session.execute(
                text(
                    """
                    UPDATE bets
                    SET status = :st,
                        pnl_units = :pnl,
                        settled_at = NOW()
                    WHERE id = :bid
                    """
                ),
                {"bid": bet["bet_id"], "st": status, "pnl": Decimal(str(round(pnl, 4)))},
            )
            counts[status] = counts.get(status, 0) + 1
    return counts


@task(retries=1, retry_delay_seconds=15)
async def update_bankroll_from_settled() -> float:
    """Suma pnl de bets settled últimos 5 min a bankroll_history."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT COALESCE(SUM(pnl_units), 0) AS delta
                FROM bets
                WHERE settled_at >= NOW() - INTERVAL '5 minutes'
                  AND status IN ('won','lost','halfwon','halflost')
                """
            )
        )
        row = result.first()
        delta = float(row[0]) if row else 0.0

        if delta == 0.0:
            return 0.0

        await session.execute(
            text(
                """
                INSERT INTO bankroll_history
                    (ts, is_paper, bankroll_units, delta_units, event, notes)
                SELECT NOW(),
                       true,
                       COALESCE(
                           (SELECT bankroll_units FROM bankroll_history
                            ORDER BY ts DESC LIMIT 1),
                           100.0
                       ) + :d,
                       :d,
                       'bet_settled',
                       'settle_bets_flow'
                """
            ),
            {"d": delta},
        )
    return delta


@flow(name="apuestas-settle-bets", log_prints=True)
async def settle_bets_flow(*, trigger_post_mortem: bool = True) -> dict[str, Any]:
    bets = await load_pending_bets_with_final_match()
    logger.info("settle_bets.start", pending_settleable=len(bets))
    if not bets:
        return {"candidates": 0, "counts": {}, "pnl_delta": 0.0}

    counts = await apply_settlement(bets)
    delta = await update_bankroll_from_settled()
    logger.info("settle_bets.done", **counts, pnl_delta=delta)

    post_mortem_result: dict[str, int] | None = None
    if trigger_post_mortem:
        try:
            post_mortem_result = await post_mortem_flow(batch_size=len(bets))
        except Exception as exc:
            logger.warning("settle_bets.postmortem_fail", error=str(exc))

    return {
        "candidates": len(bets),
        "counts": counts,
        "pnl_delta": delta,
        "post_mortem": post_mortem_result,
    }


if __name__ == "__main__":
    asyncio.run(settle_bets_flow())
