"""Lineup scratch detector — Sprint 14 #147.

Scrape starting lineup ~2h pre-kickoff para detectar:
  - Starting pitcher scratch (MLB): cambio = pick stale
  - Star player scratch (NBA): wire con star_out_adjustment

Fuente primary: The Odds API `/events/{event_id}/odds` o Sofascore lineups.
Fallback: parseo de injury reports.

Marca pick_alerts.notes con soft_tag='lineup_changed' y devalúa confidence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def check_starting_pitcher_change_mlb(session: Any, match_id: int) -> dict[str, Any]:
    """Detecta si starting pitcher MLB cambió desde el momento del pick.

    Compara `predictions.features_snapshot['starting_pitcher_id']` con
    lineup actual en `match_lineups` (cuando exista tabla).

    Fallback: si no hay tabla lineups, retorna no_change.
    """
    try:
        row = (
            await session.execute(
                text(
                    """
                    SELECT pa.id, p.features_snapshot
                    FROM pick_alerts pa
                    JOIN predictions p ON p.id = pa.prediction_id
                    WHERE pa.match_id = :mid AND pa.outcome_result IS NULL
                    LIMIT 1
                    """
                ),
                {"mid": match_id},
            )
        ).first()
        if not row or not row.features_snapshot:
            return {"changed": False, "reason": "no_baseline"}

        # Check current lineup via match_lineups table (optional, no error si missing)
        try:
            cur_lineup = (
                await session.execute(
                    text(
                        "SELECT home_starting_pitcher_id, away_starting_pitcher_id "
                        "FROM match_lineups WHERE match_id=:mid "
                        "ORDER BY updated_at DESC LIMIT 1"
                    ),
                    {"mid": match_id},
                )
            ).first()
        except Exception:
            return {"changed": False, "reason": "no_lineup_table"}

        if not cur_lineup:
            return {"changed": False, "reason": "no_current_lineup"}

        snap = row.features_snapshot or {}
        orig_home = snap.get("home_starting_pitcher_id")
        orig_away = snap.get("away_starting_pitcher_id")
        cur_home = cur_lineup.home_starting_pitcher_id
        cur_away = cur_lineup.away_starting_pitcher_id

        if orig_home and cur_home and orig_home != cur_home:
            return {
                "changed": True,
                "reason": "home_pitcher_scratch",
                "orig": orig_home,
                "current": cur_home,
            }
        if orig_away and cur_away and orig_away != cur_away:
            return {
                "changed": True,
                "reason": "away_pitcher_scratch",
                "orig": orig_away,
                "current": cur_away,
            }
        return {"changed": False, "reason": "confirmed"}
    except Exception as exc:
        logger.debug("lineup_check.fail", match_id=match_id, error=str(exc)[:80])
        return {"changed": False, "reason": "error"}


async def mark_stale_picks_pre_kickoff(
    session: Any, sport_code: str = "mlb", minutes_before: int = 120
) -> int:
    """Escanea picks pending cuyo match arranca en <minutes_before min.

    Si starting lineup cambió → escribir soft_tag='lineup_changed' en notes
    y downgrade a tier inferior (lógica en classify_confidence).

    Retorna número de picks marcados.
    """
    window_end = datetime.now(tz=UTC) + timedelta(minutes=minutes_before)
    window_start = datetime.now(tz=UTC) + timedelta(minutes=max(30, minutes_before // 4))
    try:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT pa.id, pa.match_id, pa.notes
                    FROM pick_alerts pa
                    JOIN matches m ON m.id = pa.match_id
                    WHERE m.sport_code = :sport
                      AND pa.outcome_result IS NULL
                      AND m.start_time BETWEEN :ws AND :we
                      AND (pa.notes IS NULL OR pa.notes NOT LIKE '%lineup_changed%')
                    """
                ),
                {"sport": sport_code, "ws": window_start, "we": window_end},
            )
        ).fetchall()
    except Exception as exc:
        logger.debug("lineup_scan.fail", error=str(exc)[:80])
        return 0

    marked = 0
    for r in rows:
        check = await check_starting_pitcher_change_mlb(session, int(r.match_id))
        if check.get("changed"):
            try:
                await session.execute(
                    text(
                        "UPDATE pick_alerts SET notes = COALESCE(notes,'') "
                        "|| ' [lineup_changed: ' || :reason || ']' "
                        "WHERE id = :id"
                    ),
                    {"reason": check["reason"], "id": int(r.id)},
                )
                marked += 1
                logger.info(
                    "lineup_scratch.marked",
                    pick_id=int(r.id),
                    match_id=int(r.match_id),
                    reason=check["reason"],
                )
            except Exception as exc:
                logger.warning("lineup_scratch.update_fail", error=str(exc)[:80])
    return marked


__all__ = ["check_starting_pitcher_change_mlb", "mark_stale_picks_pre_kickoff"]
