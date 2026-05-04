"""Watchdog: alerta si el bot está silencioso durante ventana activa (Gap 7/A6).

Cada hora (schedule externo o auto_loop Telegram), verifica:
  - ¿Hay picks emitidos en las últimas `quiet_hours`? (default 4h)
  - ¿El bot está `auto_on`? (sino, skip — el usuario lo pausó conscientemente)

Si sigue silencioso durante ventana activa, emite alerta admin Telegram.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def check_bot_silence(*, quiet_hours: int = 4) -> bool:
    """Retorna True si el bot lleva >quiet_hours sin emitir.

    Fuente del estado auto_on: `bot_state.key='auto_on'`. Si no existe o
    es false, devolvemos False (no hay silencio que reportar).
    """
    async with session_scope() as session:
        auto_on_row = (
            await session.execute(text("SELECT value FROM bot_state WHERE key = 'auto_on'"))
        ).first()
        if auto_on_row is None:
            return False
        value = auto_on_row.value or ""
        if "true" not in str(value).lower():
            return False

        threshold = datetime.now(tz=UTC) - timedelta(hours=quiet_hours)
        last_row = (
            await session.execute(
                text(
                    """
                    SELECT MAX(COALESCE(last_alert_at, placed_at)) AS last_ts
                    FROM pick_alerts
                    """
                )
            )
        ).first()
        last_ts = last_row.last_ts if last_row else None
        if last_ts is None:
            # Bot encendido sin ningún emit histórico → posible arranque reciente
            return False
        return last_ts < threshold


async def check_stuck_picks(*, hours_after_kickoff: int = 6) -> list[dict[str, object]]:
    """Picks con kickoff > N horas atrás y outcome_result aún pending/null.

    Detecta el bug del fin de semana 25-26 abr: live_scores no resolvió scores
    de partidos en ligas no cubiertas (Suiza/Noruega/Turquía/segundas), dejando
    7 picks pending eternamente. Esta función expone esa condición para alertar.

    Excluye:
      - Picks ya marcados como void/cancelled (status).
      - Picks de deportes no-emit (NHL/tennis/boxing): no vamos a re-resolver
        scores de deportes desactivados; ese watchdog solo aporta ruido.
      - Picks > 72h: si después de 3 días siguen sin score son irrecuperables,
        el alert recurrente solo molesta. Se deben marcar void manualmente.
    """
    async with session_scope() as session:
        rows = await session.execute(
            text(
                """
                SELECT pa.id, pa.match_id,
                       (SELECT name FROM teams WHERE id = m.home_team_id) AS home,
                       (SELECT name FROM teams WHERE id = m.away_team_id) AS away,
                       m.sport_code, m.start_time,
                       EXTRACT(EPOCH FROM (NOW() - m.start_time)) / 3600 AS hours_ago
                FROM pick_alerts pa
                JOIN matches m ON m.id = pa.match_id
                WHERE (pa.outcome_result IS NULL OR pa.outcome_result = 'pending')
                  AND COALESCE(pa.status, 'pending') NOT IN ('void', 'cancelled')
                  AND m.sport_code IN ('mlb', 'nba', 'soccer', 'nfl')
                  AND m.start_time < NOW() - make_interval(hours => :h)
                  AND m.start_time > NOW() - INTERVAL '72 hours'
                ORDER BY m.start_time
                """
            ),
            {"h": int(hours_after_kickoff)},
        )
        return [dict(r._mapping) for r in rows.all()]


async def auto_void_irrecoverable_picks(*, hours_threshold: int = 96) -> int:
    """Marca como void picks con kickoff > N horas atrás cuyo match sigue sin
    scores ingeridos.

    Casos típicos:
    - NBA sin `external_id_nba` (pinnacle scraper crea match sin id nativo
      → `sync_nba_scores_native` skipea → pick queda pending eternamente).
    - Soccer en ligas no cubiertas por football-data.org gratuito y con Odds
      API en budget critical.
    - UCL knockout con identity rota (Sp Braga vs Freiburg como CL en lugar
      de Europa League).

    Después de 96 h sin score es virtualmente irrecuperable; mejor void que
    seguir saturando watchdog.stuck_picks alerts.
    """
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                UPDATE pick_alerts SET
                    status = 'void',
                    outcome_result = 'void',
                    settled_at = NOW(),
                    notes = COALESCE(notes, '') ||
                            ' [auto-void: irrecoverable score >' ||
                            CAST(:h AS text) || 'h post-kickoff]'
                FROM matches m
                WHERE m.id = pick_alerts.match_id
                  AND (pick_alerts.outcome_result IS NULL
                       OR pick_alerts.outcome_result = 'pending')
                  AND COALESCE(pick_alerts.status, 'pending') NOT IN ('void', 'cancelled')
                  AND m.home_score IS NULL
                  AND m.away_score IS NULL
                  AND m.start_time < NOW() - make_interval(hours => :h)
                RETURNING pick_alerts.id
                """
            ),
            {"h": int(hours_threshold)},
        )
        rows = result.fetchall()
        n = len(rows)
        if n > 0:
            logger.warning(
                "watchdog.auto_voided_irrecoverable_picks",
                count=n,
                pick_ids=[int(r[0]) for r in rows][:10],
            )
        return n


async def run_watchdog() -> dict[str, object]:
    """Entry point. Si detecta silencio o picks atascados, notifica admin Telegram."""
    silent = await check_bot_silence()
    voided = await auto_void_irrecoverable_picks(hours_threshold=96)
    stuck = await check_stuck_picks(hours_after_kickoff=6)

    if stuck:
        logger.warning("watchdog.stuck_picks", count=len(stuck))
        try:
            from apuestas.bot.telegram import send_admin_alert

            sample = "\n".join(
                f"  • #{p['id']} {p['home']} vs {p['away']} ({p['sport_code']}, {int(p['hours_ago'])}h)"
                for p in stuck[:5]
            )
            await send_admin_alert(
                f"⚠️ <b>{len(stuck)} picks atascados</b> (kickoff &gt;6h, outcome pending):\n"
                f"{sample}\n\nRevisa <code>make live-scores</code>."
            )
        except Exception as exc:
            logger.debug("watchdog.stuck_notify_fail", error=str(exc)[:80])

    if silent:
        logger.warning("watchdog.bot_silent_over_threshold")
        try:
            from apuestas.bot.telegram import send_admin_alert

            await send_admin_alert(
                "🔕 <b>Bot silencioso</b>: no se han emitido picks en las últimas 4h. "
                "Revisa <code>/estado</code> o reinicia <code>apuestas go</code>."
            )
        except Exception as exc:
            logger.debug("watchdog.notify_fail", error=str(exc)[:80])

    return {"silent": silent, "stuck_count": len(stuck), "auto_voided": voided}


__all__ = [
    "auto_void_irrecoverable_picks",
    "check_bot_silence",
    "check_stuck_picks",
    "run_watchdog",
]
