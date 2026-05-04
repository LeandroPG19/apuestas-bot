"""Steam move detector — detecta cuando Pinnacle mueve línea significativamente.

Los sharps mueven Pinnacle primero (muy líquido, respuesta rápida). Los soft
books tardan 5-15 min en ajustar. Ese gap temporal es EV garantizado: apuesta
soft book ANTES de que ajuste a la nueva línea de Pinnacle.

Detección:
- Compara Pinnacle odds de hace 30 min vs ahora
- Si delta implied_prob > 3% → STEAM MOVE
- Busca soft books que AÚN cotizan la línea vieja
- Emite pick con EV calculado vs nuevo fair

Uso: integrado en el auto_loop, corre cada ciclo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SteamMove:
    match_id: int
    outcome: str
    pinnacle_odds_before: float
    pinnacle_odds_now: float
    delta_implied_prob_pp: float  # puntos porcentuales
    direction: str  # "up" (odds suben = menos probable) | "down" (odds bajan)
    soft_books_behind: list[tuple[str, float]]  # books aún con odds vieja
    detected_at: datetime


async def detect_steam_moves(
    *,
    min_delta_pp: float = 3.0,
    lookback_minutes: int = 30,
) -> list[SteamMove]:
    """Escanea matches con cambio >3pp en Pinnacle fair durante lookback.

    Args:
        min_delta_pp: umbral mínimo de cambio (pp implied prob).
        lookback_minutes: ventana histórica para detectar cambio.
    """
    now = datetime.now(tz=UTC)
    cutoff = now - timedelta(minutes=lookback_minutes)
    moves: list[SteamMove] = []

    async with session_scope() as s:
        # Matches con odds recientes Pinnacle (≤ 5 min) Y hace ≥ 20min
        r = await s.execute(
            text(
                """
                WITH recent AS (
                    SELECT DISTINCT ON (match_id, outcome)
                        match_id, outcome, odds, ts
                    FROM odds_history
                    WHERE bookmaker = 'pinnacle'
                      AND market = 'h2h'
                      AND ts > NOW() - INTERVAL '5 minutes'
                    ORDER BY match_id, outcome, ts DESC
                ),
                old AS (
                    SELECT DISTINCT ON (match_id, outcome)
                        match_id, outcome, odds, ts
                    FROM odds_history
                    WHERE bookmaker = 'pinnacle'
                      AND market = 'h2h'
                      AND ts BETWEEN :cutoff AND NOW() - INTERVAL '15 minutes'
                    ORDER BY match_id, outcome, ts DESC
                )
                SELECT r.match_id, r.outcome,
                       r.odds AS now_odds, o.odds AS old_odds
                FROM recent r
                JOIN old o ON o.match_id = r.match_id AND o.outcome = r.outcome
                WHERE ABS((1.0 / r.odds) - (1.0 / o.odds)) > :min_delta
                  AND r.match_id IN (
                    SELECT id FROM matches
                    WHERE status = 'scheduled'
                      AND start_time > NOW()
                      AND start_time < NOW() + INTERVAL '48 hours'
                  )
                """
            ),
            {"cutoff": cutoff, "min_delta": min_delta_pp / 100.0},
        )
        rows = r.all()

    for row in rows:
        # Buscar soft books que AÚN cotizan cerca de la odds vieja
        async with session_scope() as s:
            sr = await s.execute(
                text(
                    """
                    SELECT DISTINCT ON (bookmaker) bookmaker, odds
                    FROM odds_history
                    WHERE match_id = :mid AND outcome = :oc AND market = 'h2h'
                      AND bookmaker NOT IN ('pinnacle','circa','bookmaker','betfair','betfair_ex_eu')
                      AND ts > NOW() - INTERVAL '10 minutes'
                    ORDER BY bookmaker, ts DESC
                    """
                ),
                {"mid": row.match_id, "oc": row.outcome},
            )
            soft_rows = sr.all()

        old_implied = 1.0 / float(row.old_odds)
        new_implied = 1.0 / float(row.now_odds)
        delta = new_implied - old_implied  # positivo = más probable ahora

        # Soft books con odds "vieja" (≈ old_odds, mejor que new_implied)
        books_behind: list[tuple[str, float]] = []
        for sb in soft_rows:
            soft_odds = float(sb.odds)
            soft_implied = 1.0 / soft_odds
            # Si soft book está más cerca de old que de new → aún no ajustó
            if abs(soft_implied - old_implied) < abs(soft_implied - new_implied):
                books_behind.append((sb.bookmaker, soft_odds))

        if books_behind:
            moves.append(
                SteamMove(
                    match_id=row.match_id,
                    outcome=row.outcome,
                    pinnacle_odds_before=float(row.old_odds),
                    pinnacle_odds_now=float(row.now_odds),
                    delta_implied_prob_pp=delta * 100,
                    direction="down" if delta > 0 else "up",
                    soft_books_behind=books_behind,
                    detected_at=now,
                )
            )

    logger.info("steam_moves.scan_done", moves_detected=len(moves))
    return moves


async def format_steam_move(move: SteamMove) -> dict[str, Any]:
    """Formatea un steam move para enviar a Telegram."""
    async with session_scope() as s:
        r = (
            await s.execute(
                text(
                    """
                    SELECT m.sport_code, m.start_time,
                           COALESCE(th.name,'?') AS home,
                           COALESCE(ta.name,'?') AS away
                    FROM matches m
                    LEFT JOIN teams th ON th.id = m.home_team_id
                    LEFT JOIN teams ta ON ta.id = m.away_team_id
                    WHERE m.id = :mid
                    """
                ),
                {"mid": move.match_id},
            )
        ).first()
    if r is None:
        return {}
    return {
        "home": r.home,
        "away": r.away,
        "sport": r.sport_code,
        "outcome": move.outcome,
        "delta_pp": move.delta_implied_prob_pp,
        "direction": move.direction,
        "pinnacle_before": move.pinnacle_odds_before,
        "pinnacle_now": move.pinnacle_odds_now,
        "books_behind": move.soft_books_behind,
        "start_time": r.start_time,
    }
