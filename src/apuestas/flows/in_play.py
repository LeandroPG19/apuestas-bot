"""Fase 3.8 — Live betting in-play skeleton.

60-70% del edge de los top pros viene del live (Voulgaris, Benter, Bloom).
Este módulo deja el skeleton listo para activar; activación real tras validar
el modelo pre-game con backtest Sharpe≥1.2 (Fase 1.5).

Arquitectura mínima:
  1. Poll Pinnacle `/live/odds` o Sofascore live endpoint cada 30s.
  2. Para cada match activo: compute `p_live_adjusted` interpolando
     entre `p_pregame` y `p_final_ajustado_por_score_actual`.
  3. Detecta overreactions del mercado (red card → under odds bajan 30% en 5s)
     vs impacto real (under odds deberían bajar solo 8-15%).
  4. Emite pick live vía `send_pick_to_telegram` con etiqueta 🔴 LIVE.

Este es un **skeleton** (~4h). Implementación full (~20h+) queda post-MVP
tras validar el modelo pregame.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

LIVE_POLL_INTERVAL_SECONDS = 30


@dataclass(slots=True)
class LiveGameState:
    match_id: int
    sport_code: str
    home_score: int
    away_score: int
    time_elapsed_min: float  # minuto del partido (0-90 soccer, 0-48 NBA)
    total_time_min: float
    incidents: list[dict[str, Any]]  # goals, red cards, injuries in-game
    p_pregame: dict[str, float]  # {outcome: p_pregame}
    current_odds: dict[str, dict[str, float]]  # {book: {outcome: odds_live}}


def interpolate_live_probability(
    p_pregame: float,
    current_score_diff: int,
    time_elapsed_pct: float,
    *,
    sport_code: str,
) -> float:
    """Estimación simple de p_live interpolando pregame vs end-state.

    Para outcomes h2h "home_win":
      p_live = α(t) · p_based_on_current_score + (1 − α(t)) · p_pregame
      α(t) = time_elapsed_pct² (peso aumenta cuadráticamente con el tiempo).

    `current_score_diff` = home_score - away_score.
    """
    alpha = min(1.0, max(0.0, time_elapsed_pct**2))

    # Estimación "current-state probability"
    # Deportes con scoring alto (NBA) → diff importa más
    # Deportes con scoring bajo (soccer) → diff es casi determinante tras el 70%
    if sport_code == "soccer":
        if current_score_diff > 1:
            p_from_state = 0.85
        elif current_score_diff == 1:
            p_from_state = 0.65
        elif current_score_diff == 0:
            p_from_state = 0.35  # empate o visitante win
        else:
            p_from_state = 0.10
    elif sport_code in ("nba", "nhl"):
        # NBA: lead de 10+ es casi seguro; lead de 5 es 70%
        p_from_state = 0.5 + 0.05 * current_score_diff
    else:
        p_from_state = p_pregame

    p_from_state = max(0.02, min(0.98, p_from_state))

    return alpha * p_from_state + (1 - alpha) * p_pregame


async def fetch_live_matches() -> list[int]:
    """Fetch match_ids con status='live' en la DB (actualizado por live_scores worker)."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id FROM matches
                    WHERE status = 'live'
                       OR (start_time < now() AND start_time > now() - interval '4 hours'
                           AND status = 'scheduled')
                    """
                )
            )
        ).all()
    return [r.id for r in rows]


async def poll_live_state(match_id: int) -> LiveGameState | None:
    """Fetch match state + odds + pregame predictions desde DB + Sofascore.

    Hidratación:
      - matches + scores actuales desde DB
      - pregame `p_model` desde predictions (último model_name)
      - current_odds desde odds_history (último minuto por bookmaker)
      - incidents desde Sofascore cuando sofascore_event_id está linkeado
    """
    async with session_scope() as session:
        match = (
            await session.execute(
                text(
                    """
                    SELECT m.id, m.sport_code, m.home_score, m.away_score,
                           m.start_time, m.metadata ->> 'sofascore_event_id' AS sfid
                    FROM matches m
                    WHERE m.id = :mid
                    """
                ),
                {"mid": match_id},
            )
        ).first()

    if match is None:
        return None

    total_time = 90.0 if match.sport_code == "soccer" else 48.0
    time_elapsed = min(
        total_time,
        (datetime.now(tz=UTC) - match.start_time).total_seconds() / 60.0,
    )

    # 1. Load pregame p_model desde predictions (más reciente por outcome)
    p_pregame: dict[str, float] = {}
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT outcome, AVG(probability) AS p
                FROM predictions
                WHERE match_id = :mid
                GROUP BY outcome
                """
            ),
            {"mid": match_id},
        )
        for row in r.all():
            p_pregame[str(row.outcome)] = float(row.p)

    # 2. Load current_odds desde odds_history (última hora, agrupado por book)
    current_odds: dict[str, dict[str, float]] = {}
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT DISTINCT ON (bookmaker, outcome)
                    bookmaker, outcome, odds
                FROM odds_history
                WHERE match_id = :mid
                  AND market = 'h2h'
                  AND ts > NOW() - INTERVAL '10 minutes'
                ORDER BY bookmaker, outcome, ts DESC
                """
            ),
            {"mid": match_id},
        )
        for row in r.all():
            current_odds.setdefault(str(row.bookmaker), {})[str(row.outcome)] = float(row.odds)

    # 3. Incidents desde Sofascore (opcional)
    incidents: list[dict[str, Any]] = []
    if match.sfid:
        try:
            from apuestas.ingest.sofascore_scraper import fetch_event_incidents

            raw_incidents = await fetch_event_incidents(int(match.sfid))
            incidents = list(raw_incidents)[:20]  # últimos 20
        except Exception as exc:
            logger.debug("in_play.sofascore_incidents_fail", error=str(exc)[:80])

    return LiveGameState(
        match_id=match_id,
        sport_code=match.sport_code,
        home_score=int(match.home_score or 0),
        away_score=int(match.away_score or 0),
        time_elapsed_min=time_elapsed,
        total_time_min=total_time,
        incidents=incidents,
        p_pregame=p_pregame,
        current_odds=current_odds,
    )


async def detect_live_value(state: LiveGameState) -> list[dict[str, Any]]:
    """Evalúa live value. Skeleton: compara p_live_estimate vs p_implied_por_odds.

    Retorna lista de picks live con EV. Activado solo si feature flag ENABLED.
    """
    picks: list[dict[str, Any]] = []

    if not state.p_pregame or not state.current_odds:
        return picks

    time_elapsed_pct = state.time_elapsed_min / state.total_time_min
    score_diff = state.home_score - state.away_score

    for outcome, p_pregame in state.p_pregame.items():
        p_live = interpolate_live_probability(
            p_pregame, score_diff, time_elapsed_pct, sport_code=state.sport_code
        )
        for book, odds_dict in state.current_odds.items():
            if outcome not in odds_dict:
                continue
            odds_live = odds_dict[outcome]
            ev = p_live * odds_live - 1.0
            if ev > 0.04:  # umbral live más alto que pregame (+4%)
                picks.append(
                    {
                        "match_id": state.match_id,
                        "outcome": outcome,
                        "book": book,
                        "odds_live": odds_live,
                        "p_live": p_live,
                        "ev": ev,
                        "state": {
                            "score": f"{state.home_score}-{state.away_score}",
                            "minute": state.time_elapsed_min,
                        },
                    }
                )
    return picks


async def in_play_loop(max_iterations: int | None = None) -> None:
    """Loop principal: cada 30s escanea live matches → detecta picks → notifica.

    `max_iterations` para testing. None = infinito.
    """
    iter_count = 0
    logger.info("in_play.loop_start")
    while max_iterations is None or iter_count < max_iterations:
        try:
            matches = await fetch_live_matches()
            all_picks: list[dict[str, Any]] = []
            for match_id in matches:
                state = await poll_live_state(match_id)
                if state is None:
                    continue
                picks = await detect_live_value(state)
                all_picks.extend(picks)

            if all_picks:
                logger.info("in_play.picks_detected", n=len(all_picks))
                await _notify_live_picks(all_picks)

        except Exception as exc:
            logger.warning("in_play.loop_error", error=str(exc)[:120])

        iter_count += 1
        if max_iterations is None or iter_count < max_iterations:
            await asyncio.sleep(LIVE_POLL_INTERVAL_SECONDS)


async def _notify_live_picks(picks: list[dict[str, Any]]) -> None:
    """Envía picks live a Telegram con tag 🔴 LIVE. No dedup per-call (controlado
    por `APUESTAS_LIVE_BETTING_ENABLED` flag).
    """
    import os

    if os.environ.get("APUESTAS_LIVE_BETTING_ENABLED", "false").lower() != "true":
        return
    try:
        from telegram import Bot

        from apuestas.config import get_settings

        settings = get_settings()
        token = settings.apis.telegram_bot_token
        chat_id = settings.apis.telegram_chat_id
        if token is None or chat_id is None:
            return
        bot = Bot(token=token)

        for pick in picks:
            msg = (
                f"🔴 <b>LIVE pick</b>\n"
                f"Match: {pick['match_id']} ({pick['state']['score']} @ "
                f"{pick['state']['minute']:.0f}')\n"
                f"{pick['outcome']} @ {pick['odds_live']:.2f} ({pick['book']})\n"
                f"p_live={pick['p_live']:.2f} · EV={pick['ev'] * 100:+.1f}%"
            )
            try:
                await bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
            except Exception as exc_send:
                logger.debug("in_play.send_fail", error=str(exc_send)[:80])
    except Exception as exc:
        logger.debug("in_play.notify_fail", error=str(exc)[:100])


if __name__ == "__main__":
    asyncio.run(in_play_loop())
