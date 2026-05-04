"""Re-clasifica pick_alerts resueltos usando la lógica actual de _classify_alert.

Cuando un bug de clasificación se corrige en código (ej: el bug spreads/away
del 24-abr donde `inv = away - home - line` sobrevió mal-clasificando picks),
los registros históricos en pick_alerts.outcome_result NO se actualizan
automáticamente.

Este módulo recorre todos los picks resueltos con score persistido y re-aplica
`_classify_alert`. Si difiere del valor actual, lo actualiza.

Idempotente. Llamado desde catchup_flow tras live_scores y mark_alert_results.
"""

from __future__ import annotations

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.flows.live_scores import _classify_alert
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def reclassify_resolved_alerts(*, since_days: int = 30) -> dict[str, int]:
    """Re-aplica _classify_alert a picks resueltos. Corrige misclasificaciones
    de bugs históricos (ej: spreads/away signo invertido fix 2026-04-24).

    Args:
        since_days: solo re-clasifica picks de los últimos N días (default 30).

    Returns:
        {"checked": N, "corrected": M}
    """
    checked = 0
    corrected = 0
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT pa.id, pa.market, pa.outcome, pa.line,
                           pa.outcome_result AS current_result,
                           m.home_score, m.away_score, m.sport_code
                    FROM pick_alerts pa
                    JOIN matches m ON m.id = pa.match_id
                    WHERE pa.outcome_result IN ('won', 'lost', 'void')
                      AND m.home_score IS NOT NULL
                      AND m.away_score IS NOT NULL
                      AND pa.placed_at >= NOW() - make_interval(days => :d)
                    """
                ),
                {"d": int(since_days)},
            )
        ).all()

        for r in rows:
            checked += 1
            recalc = _classify_alert(
                market=str(r.market).lower(),
                outcome=str(r.outcome).lower(),
                line=float(r.line) if r.line is not None else None,
                home_score=int(r.home_score),
                away_score=int(r.away_score),
                sport=r.sport_code or "",
            )
            if recalc is None:
                continue
            if recalc == r.current_result:
                continue

            await session.execute(
                text(
                    """
                    UPDATE pick_alerts
                    SET outcome_result = :r,
                        result_settled_at = COALESCE(result_settled_at, now())
                    WHERE id = :id
                    """
                ),
                {"r": recalc, "id": int(r.id)},
            )
            corrected += 1
            logger.info(
                "reclassify.corrected",
                pick_id=int(r.id),
                from_result=r.current_result,
                to_result=recalc,
                market=r.market,
                outcome=r.outcome,
                line=float(r.line) if r.line is not None else None,
                score=f"{r.home_score}-{r.away_score}",
            )

    if corrected > 0:
        logger.info("reclassify.summary", checked=checked, corrected=corrected)
    return {"checked": checked, "corrected": corrected}


__all__ = ["reclassify_resolved_alerts"]
