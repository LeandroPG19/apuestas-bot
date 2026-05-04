"""Market movement features — Sprint 14 #158.

Detecta line movement pre-kickoff (Pinnacle) para identificar sharp money.
Si Pinnacle odds mueven contra el pick del bot → sharp disagreement → baja tier.
Si mueven a favor → sharp confirmation → sube tier.

Métricas:
  - line_move_6h_to_30min: (odds_30m - odds_6h) / odds_6h
  - line_move_velocity: cambios/hora últimas 3h
  - rlm_flag: Reverse Line Movement (línea mueve contra public %)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def compute_line_movement(
    session: Any,
    *,
    match_id: int,
    market: str,
    outcome: str,
    line: float | None,
    match_start: datetime,
) -> dict[str, float]:
    """Movimiento de line Pinnacle entre t-6h y t-30min pre-kickoff.

    Retorna dict con:
      - line_move_pct: (odds_30m - odds_6h) / odds_6h
      - n_updates_3h: número de cambios últimas 3h
      - has_sharp_move: 1 si movimiento > 3% en <1h (steam)
    """
    result = {"line_move_pct": 0.0, "n_updates_3h": 0.0, "has_sharp_move": 0.0}
    try:
        t_6h = match_start - timedelta(hours=6)
        t_30m = match_start - timedelta(minutes=30)
        t_3h = match_start - timedelta(hours=3)

        rows = (
            await session.execute(
                text(
                    """
                    SELECT ts, odds FROM odds_history
                    WHERE match_id=:mid AND bookmaker='pinnacle'
                      AND market=:mkt AND outcome=:out
                      AND (line=:line OR (:line IS NULL AND line IS NULL))
                      AND ts BETWEEN :t_6h AND :t_30m
                    ORDER BY ts
                    """
                ),
                {
                    "mid": match_id,
                    "mkt": market,
                    "out": outcome,
                    "line": line,
                    "t_6h": t_6h,
                    "t_30m": t_30m,
                },
            )
        ).fetchall()
        if len(rows) < 2:
            return result

        first_odds = float(rows[0].odds)
        last_odds = float(rows[-1].odds)
        if first_odds > 0:
            result["line_move_pct"] = (last_odds - first_odds) / first_odds

        # Updates last 3h + steam detection
        recent = [r for r in rows if r.ts >= t_3h]
        result["n_updates_3h"] = float(len(recent))

        # Steam: buscar cambio >3% en ventana 1h cualquiera
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                if (recent[j].ts - recent[i].ts) <= timedelta(hours=1):
                    if first_odds > 0:
                        change = abs(
                            (float(recent[j].odds) - float(recent[i].odds)) / float(recent[i].odds)
                        )
                        if change > 0.03:
                            result["has_sharp_move"] = 1.0
                            return result
                else:
                    break
        return result
    except Exception as exc:
        logger.debug("market_movement.fail", match_id=match_id, error=str(exc)[:80])
        return result


def classify_move_vs_pick(
    *, pick_outcome: str, line_move_pct: float, threshold: float = 0.02
) -> str:
    """Si pick=home y home_odds subió (line_move_pct > 0), pinnacle lo devaluó →
    sharp_disagreement. Si bajó (move<0), sharp_confirmation.

    Con line_move=0 o |move|<threshold: stable.
    """
    if abs(line_move_pct) < threshold:
        return "stable"
    if line_move_pct > 0:
        # Odds subieron → Pinnacle cree menos probable el outcome → sharp disagrees
        return "sharp_disagreement"
    return "sharp_confirmation"


__all__ = ["classify_move_vs_pick", "compute_line_movement"]
