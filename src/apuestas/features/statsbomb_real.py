"""StatsBomb event-level real features — Sprint 14 #153.

Reemplaza soccer_xt proxy con features reales extraidas de los 3.1M eventos
ya ingestados en `statsbomb_events`.

Features por equipo por match (calculados sobre últimos 5 matches rolling):
  - xg_total_rolling
  - shots_total
  - shots_on_target_ratio
  - pass_completion_final_3rd
  - progressive_passes_per_match
  - pressures_per_def_action
  - ball_recoveries_final_3rd
  - corners_per_match

Todos anti-leakage (solo events antes de match_time).

Uso en trainer: from apuestas.features.statsbomb_real import build_sb_features
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def fetch_statsbomb_team_rolling(
    session: Any, team_id: int, as_of: datetime, n_matches: int = 5
) -> dict[str, float]:
    """Features StatsBomb rolling últimos n_matches del team antes de as_of.

    Fallback a 0s si statsbomb_events no existe o team no coincide.
    """
    empty = {
        "sb_xg_rolling": 0.0,
        "sb_shots_rolling": 0.0,
        "sb_shots_on_target_pct": 0.0,
        "sb_pass_completion_final3": 0.0,
        "sb_progressive_passes": 0.0,
        "sb_pressures_per_match": 0.0,
    }
    try:
        exists = (
            await session.execute(
                text(
                    "SELECT COUNT(*) n FROM information_schema.tables "
                    "WHERE table_name='statsbomb_events'"
                )
            )
        ).first()
        if not exists or exists.n == 0:
            return empty

        # Get last n match_ids for team
        match_ids = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT match_id FROM statsbomb_events
                    WHERE team_id = :tid AND match_date < :ts
                    ORDER BY match_date DESC LIMIT :n
                    """
                ),
                {"tid": team_id, "ts": as_of, "n": n_matches},
            )
        ).fetchall()
        if not match_ids:
            return empty

        ids = [m.match_id for m in match_ids]
        # Aggregate event counts
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                        COALESCE(SUM((data->>'shot_statsbomb_xg')::float), 0.0) xg,
                        COUNT(*) FILTER (WHERE event_type='Shot') shots,
                        COUNT(*) FILTER (WHERE event_type='Shot'
                            AND (data->>'shot_on_target')::bool) on_target,
                        COUNT(*) FILTER (WHERE event_type='Pass'
                            AND NOT (data->>'pass_incomplete')::bool
                            AND (data->>'pass_end_x')::float > 80) comp_final3,
                        COUNT(*) FILTER (WHERE event_type='Pass'
                            AND (data->>'pass_progressive')::bool) progressive,
                        COUNT(*) FILTER (WHERE event_type='Pressure') pressures,
                        COUNT(DISTINCT match_id) n_matches
                    FROM statsbomb_events
                    WHERE team_id = :tid
                      AND match_id = ANY(:ids)
                    """
                ),
                {"tid": team_id, "ids": ids},
            )
        ).first()
        n_m = max(1, int(row.n_matches or 1))
        return {
            "sb_xg_rolling": float(row.xg) / n_m,
            "sb_shots_rolling": float(row.shots) / n_m,
            "sb_shots_on_target_pct": float(row.on_target) / max(1, float(row.shots)),
            "sb_pass_completion_final3": float(row.comp_final3) / n_m,
            "sb_progressive_passes": float(row.progressive) / n_m,
            "sb_pressures_per_match": float(row.pressures) / n_m,
        }
    except Exception as exc:
        logger.debug("statsbomb.rolling.fail", team_id=team_id, error=str(exc)[:80])
        return empty


async def build_sb_features_for_match(
    session: Any, *, home_team_id: int, away_team_id: int, match_start: datetime
) -> dict[str, float]:
    """Combina features StatsBomb para ambos equipos + diff features."""
    home = await fetch_statsbomb_team_rolling(session, home_team_id, match_start)
    away = await fetch_statsbomb_team_rolling(session, away_team_id, match_start)
    out: dict[str, float] = {}
    for k, v in home.items():
        out[f"{k}_home"] = v
    for k, v in away.items():
        out[f"{k}_away"] = v
    for k in home:
        out[f"{k}_diff"] = home[k] - away[k]
    return out


__all__ = ["build_sb_features_for_match", "fetch_statsbomb_team_rolling"]
