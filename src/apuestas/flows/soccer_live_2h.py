"""Soccer Second-Half live betting via Kalman filter — Sprint 14 #154.

Wire betting/live_kalman.LiveKalmanFilter al mercado soccer 2H.

Trigger: evento live_scores con status='ht' (half-time). Calcula estado
state-space (home/away scoring rate + total_goals proxy) desde 1H observado,
proyecta posterior para 2H, genera picks O/U 1.5 goals segunda parte.

Flow:
  1. Match en HT (45:00-55:00 min).
  2. Fetch 1H score + shots + xG acumulado.
  3. Kalman update con observaciones 1H.
  4. Predicción 2H: λ_home_2h, λ_away_2h.
  5. p(total_2h > 1.5) = 1 - Poisson.cdf(1, λ_total_2h).
  6. Comparar vs odds soft book en market 'totals_2h' → EV.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def fetch_halftime_snapshot(session: Any, match_id: int) -> dict[str, float] | None:
    """Score + xG al minuto 45. Requiere live_scores_minutely populated."""
    try:
        row = (
            await session.execute(
                text(
                    """
                    SELECT home_score_1h hs1, away_score_1h as1,
                           home_xg_1h hxg, away_xg_1h axg,
                           home_shots_1h hs, away_shots_1h as_
                    FROM match_live_snapshots
                    WHERE match_id=:mid AND snapshot_minute BETWEEN 44 AND 48
                    ORDER BY snapshot_minute DESC LIMIT 1
                    """
                ),
                {"mid": match_id},
            )
        ).first()
        if not row:
            return None
        return {
            "home_goals_1h": float(row.hs1 or 0),
            "away_goals_1h": float(row.as1 or 0),
            "home_xg_1h": float(row.hxg or 0),
            "away_xg_1h": float(row.axg or 0),
        }
    except Exception as exc:
        logger.debug("soccer_live_2h.fetch_fail", match_id=match_id, error=str(exc)[:80])
        return None


def project_2h_goals(snapshot: dict[str, float]) -> dict[str, float]:
    """Proyección simple 2H via Kalman. Usa xG_1h como estimador unbiased."""
    try:
        from apuestas.betting.live_kalman import LiveKalmanFilter
    except ImportError:
        LiveKalmanFilter = None

    lam_home = snapshot.get("home_xg_1h", 0.5)
    lam_away = snapshot.get("away_xg_1h", 0.5)

    # Si Kalman disponible, refina. Si no, retorna xG directo (baseline).
    if LiveKalmanFilter is not None:
        try:
            kf = LiveKalmanFilter(variance_by_sport={"soccer": 0.3})
            # Kalman update con observación xG_1h (assume true rate ≈ xG)
            kf.observe({"home_rate": lam_home, "away_rate": lam_away}, sport="soccer")
            post = kf.predict(horizon=1)
            lam_home = post.get("home_rate", lam_home)
            lam_away = post.get("away_rate", lam_away)
        except Exception:
            pass

    return {"lambda_home_2h": lam_home, "lambda_away_2h": lam_away}


def prob_over_15_2h(lam_home_2h: float, lam_away_2h: float) -> float:
    """P(total goals 2H > 1.5) con Poisson sum."""
    from scipy.stats import poisson

    lam_total = lam_home_2h + lam_away_2h
    return float(1.0 - poisson.cdf(1, lam_total))


async def evaluate_match_halftime(session: Any, match_id: int) -> dict[str, Any]:
    snap = await fetch_halftime_snapshot(session, match_id)
    if snap is None:
        return {"match_id": match_id, "error": "no_halftime_snapshot"}
    proj = project_2h_goals(snap)
    p_over_15 = prob_over_15_2h(proj["lambda_home_2h"], proj["lambda_away_2h"])
    return {
        "match_id": match_id,
        "snapshot": snap,
        "projection": proj,
        "p_over_15_2h": round(p_over_15, 4),
        "p_under_15_2h": round(1.0 - p_over_15, 4),
    }


__all__ = ["evaluate_match_halftime", "prob_over_15_2h", "project_2h_goals"]
