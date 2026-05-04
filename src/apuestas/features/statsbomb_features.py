"""Features agregados desde `statsbomb_events` — Sprint 12.

Convierte 4M+ eventos event-level en features team-match útiles para
trainer soccer. Agrega por (match_id, team_id):
- n_shots / n_shots_on_target
- xG total (del event type "Shot" con shot.statsbomb_xg)
- passes_completed_pct
- pressures aplicadas
- ball recoveries
- dribbles completed
- progressive_passes (passes que avanzan >30m hacia portería rival)

Estos features aproximan VAEP/xT sin implementar el modelo completo.
Sufficient para mejorar el trainer soccer cuando StatsBomb open-data
coincide con matches de DB.

Uso:
    from apuestas.features.statsbomb_features import compute_match_team_features
    feats = await compute_match_team_features(session, sb_match_id=123)
    # feats = {"home": {...}, "away": {...}}
"""

from __future__ import annotations

from typing import Any

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def compute_match_team_features(
    session: Any, sb_match_id: int
) -> dict[int, dict[str, float]]:
    """Agrega eventos StatsBomb → features por team del match.

    Retorna: {team_id: {feature_name: value}}.
    """
    from sqlalchemy import text as _text

    try:
        rows = (
            await session.execute(
                _text(
                    """
                    SELECT team_id, event_type, event_jsonb
                    FROM statsbomb_events
                    WHERE match_id = :mid AND team_id IS NOT NULL
                    """
                ),
                {"mid": sb_match_id},
            )
        ).fetchall()
    except Exception as exc:
        logger.debug("sb_features.fetch_fail", mid=sb_match_id, error=str(exc)[:100])
        return {}

    if not rows:
        return {}

    # Agregación por team_id
    by_team: dict[int, dict[str, float]] = {}
    for row in rows:
        tid = int(row.team_id)
        if tid not in by_team:
            by_team[tid] = {
                "n_events": 0,
                "n_shots": 0,
                "n_shots_on_target": 0,
                "xg_total": 0.0,
                "n_passes": 0,
                "n_passes_completed": 0,
                "n_progressive_passes": 0,
                "n_pressures": 0,
                "n_ball_recoveries": 0,
                "n_dribbles": 0,
                "n_dribbles_completed": 0,
            }
        acc = by_team[tid]
        et = str(row.event_type or "").lower()
        ev = row.event_jsonb or {}
        acc["n_events"] += 1

        if et == "shot":
            acc["n_shots"] += 1
            shot = ev.get("shot") or {}
            xg = shot.get("statsbomb_xg")
            if xg is not None:
                try:
                    acc["xg_total"] += float(xg)
                except (TypeError, ValueError):
                    pass
            outcome = (shot.get("outcome") or {}).get("name", "")
            if outcome in ("Goal", "Saved"):
                acc["n_shots_on_target"] += 1
        elif et == "pass":
            acc["n_passes"] += 1
            p = ev.get("pass") or {}
            if p.get("outcome") is None:
                # outcome None = completo (StatsBomb convention)
                acc["n_passes_completed"] += 1
            # Progressive: pass avanza >30% distancia a portería
            loc = ev.get("location") or []
            end = p.get("end_location") or []
            if len(loc) >= 2 and len(end) >= 2:
                try:
                    dist_gain = float(end[0]) - float(loc[0])
                    if dist_gain > 20:  # >20m hacia portería (campo 120m long)
                        acc["n_progressive_passes"] += 1
                except (TypeError, ValueError):
                    pass
        elif et == "pressure":
            acc["n_pressures"] += 1
        elif et == "ball recovery":
            acc["n_ball_recoveries"] += 1
        elif et == "dribble":
            acc["n_dribbles"] += 1
            d = ev.get("dribble") or {}
            if (d.get("outcome") or {}).get("name", "") == "Complete":
                acc["n_dribbles_completed"] += 1

    # Derivadas
    for acc in by_team.values():
        n_passes = max(acc["n_passes"], 1)
        acc["pass_completion_pct"] = acc["n_passes_completed"] / n_passes
        n_dribbles = max(acc["n_dribbles"], 1)
        acc["dribble_completion_pct"] = acc["n_dribbles_completed"] / n_dribbles
        n_shots = max(acc["n_shots"], 1)
        acc["shot_on_target_pct"] = acc["n_shots_on_target"] / n_shots
        acc["xg_per_shot"] = acc["xg_total"] / n_shots

    return by_team


async def compute_team_rolling_from_sb(
    session: Any, team_sb_id: int, through_match_id: int, window: int = 10
) -> dict[str, float]:
    """Features rolling últimos N matches del team en StatsBomb hasta
    `through_match_id` (anti-leakage: no incluye el match actual).

    Uso en trainer soccer: enriquece el feature frame con xG_roll_10,
    pass_completion_roll_10, progressive_passes_roll_10.
    """
    from sqlalchemy import text as _text

    try:
        rows = (
            await session.execute(
                _text(
                    """
                    SELECT DISTINCT match_id FROM statsbomb_events
                    WHERE team_id = :tid AND match_id < :thru
                    ORDER BY match_id DESC LIMIT :w
                    """
                ),
                {"tid": team_sb_id, "thru": through_match_id, "w": window},
            )
        ).fetchall()
    except Exception as exc:
        logger.debug("sb_features.rolling_fail", error=str(exc)[:100])
        return {}

    if not rows:
        return {}

    xg_vals: list[float] = []
    pass_pct_vals: list[float] = []
    progressive_vals: list[int] = []
    shots_vals: list[int] = []

    for r in rows:
        feats = await compute_match_team_features(session, int(r.match_id))
        team_feats = feats.get(team_sb_id)
        if team_feats is None:
            continue
        xg_vals.append(float(team_feats.get("xg_total", 0.0)))
        pass_pct_vals.append(float(team_feats.get("pass_completion_pct", 0.0)))
        progressive_vals.append(int(team_feats.get("n_progressive_passes", 0)))
        shots_vals.append(int(team_feats.get("n_shots", 0)))

    n = max(len(xg_vals), 1)
    return {
        "sb_xg_mean": sum(xg_vals) / n if xg_vals else 0.0,
        "sb_pass_pct_mean": sum(pass_pct_vals) / n if pass_pct_vals else 0.0,
        "sb_progressive_mean": sum(progressive_vals) / n if progressive_vals else 0.0,
        "sb_shots_mean": sum(shots_vals) / n if shots_vals else 0.0,
        "sb_n_matches": float(n),
    }


__all__ = [
    "compute_match_team_features",
    "compute_team_rolling_from_sb",
]
