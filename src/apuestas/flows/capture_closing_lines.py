"""Capture closing lines Pinnacle para picks activos — CLV tracking Sprint 12.

Prefect flow que corre cada 10 min y captura snapshot de Pinnacle odds
para matches dentro de 30-60 min del kickoff. Alimenta `pick_closing_lines`.

Uso programático:
    from apuestas.flows.capture_closing_lines import capture_closing_lines_flow
    await capture_closing_lines_flow()

CLI:
    uv run python -m apuestas.flows.capture_closing_lines
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CAPTURE_WINDOW_MIN = 30  # captura entre 30-60 min antes del kickoff
_CAPTURE_WINDOW_MAX = 60


async def _devig_two_way(odds_home: float, odds_away: float) -> tuple[float, float]:
    """De-vig simple 2-way (multiplicative)."""
    if odds_home <= 1.0 or odds_away <= 1.0:
        return 1.0, 1.0
    p_h = 1.0 / odds_home
    p_a = 1.0 / odds_away
    total = p_h + p_a
    if total <= 0:
        return 1.0, 1.0
    return 1.0 / (p_h / total), 1.0 / (p_a / total)


try:
    from prefect import flow as _prefect_flow  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover

    def _prefect_flow(**_kwargs):  # type: ignore[no-redef]
        def deco(fn):
            return fn

        return deco


@_prefect_flow(name="apuestas-capture-closing-lines", log_prints=True)
async def capture_closing_lines_flow() -> int:
    """Busca picks activos con kickoff en ventana 30-60 min y snapshot Pinnacle.

    Returns: número de snapshots capturados.
    """
    now = datetime.now(tz=UTC)
    window_start = now + timedelta(minutes=_CAPTURE_WINDOW_MIN)
    window_end = now + timedelta(minutes=_CAPTURE_WINDOW_MAX)

    captured = 0
    async with session_scope() as s:
        # Picks vivos con match en ventana
        rows = (
            await s.execute(
                text(
                    """
                    SELECT pa.id AS pick_id, pa.match_id, pa.market, pa.outcome, pa.line,
                           m.start_time
                    FROM pick_alerts pa
                    JOIN matches m ON m.id = pa.match_id
                    WHERE pa.outcome_result IS NULL
                      AND m.start_time BETWEEN :ws AND :we
                      AND pa.closing_captured_at IS NULL
                    """
                ),
                {"ws": window_start, "we": window_end},
            )
        ).fetchall()

        if not rows:
            logger.info("clv.capture.no_picks_in_window")
            return 0

        for row in rows:
            try:
                # Buscar última odds Pinnacle del mismo market+outcome
                pinn_rows = (
                    await s.execute(
                        text(
                            """
                            SELECT odds FROM odds_history
                            WHERE match_id = :mid AND market = :mkt AND outcome = :out
                              AND bookmaker = 'pinnacle'
                              AND ts <= :now
                            ORDER BY ts DESC LIMIT 2
                            """
                        ),
                        {
                            "mid": row.match_id,
                            "mkt": row.market,
                            "out": row.outcome,
                            "now": now,
                        },
                    )
                ).fetchall()
                if not pinn_rows:
                    continue
                pinn_odds = float(pinn_rows[0].odds)

                # Buscar odds del outcome opuesto para de-vig
                opposite = (
                    "away" if row.outcome == "home" else ("home" if row.outcome == "away" else None)
                )
                devigged = pinn_odds
                devig_method = "raw"
                if opposite:
                    opp_rows = (
                        await s.execute(
                            text(
                                """
                                SELECT odds FROM odds_history
                                WHERE match_id = :mid AND market = :mkt AND outcome = :out
                                  AND bookmaker = 'pinnacle'
                                  AND ts <= :now
                                ORDER BY ts DESC LIMIT 1
                                """
                            ),
                            {
                                "mid": row.match_id,
                                "mkt": row.market,
                                "out": opposite,
                                "now": now,
                            },
                        )
                    ).fetchall()
                    if opp_rows:
                        opp_odds = float(opp_rows[0].odds)
                        d_home, d_away = await _devig_two_way(
                            pinn_odds if row.outcome == "home" else opp_odds,
                            opp_odds if row.outcome == "home" else pinn_odds,
                        )
                        devigged = d_home if row.outcome == "home" else d_away
                        devig_method = "multiplicative_2way"

                minutes_to_kickoff = int((row.start_time - now).total_seconds() / 60)

                await s.execute(
                    text(
                        """
                        INSERT INTO pick_closing_lines (
                            pick_alert_id, match_id, market, outcome, line,
                            pinnacle_odds, pinnacle_odds_devigged, devig_method,
                            minutes_to_kickoff, captured_at
                        ) VALUES (
                            :pid, :mid, :mkt, :out, :ln,
                            :po, :pod, :dm, :mtk, :cap
                        )
                        ON CONFLICT (pick_alert_id, captured_at) DO NOTHING
                        """
                    ),
                    {
                        "pid": row.pick_id,
                        "mid": row.match_id,
                        "mkt": row.market,
                        "out": row.outcome,
                        "ln": row.line,
                        "po": pinn_odds,
                        "pod": devigged,
                        "dm": devig_method,
                        "mtk": minutes_to_kickoff,
                        "cap": now,
                    },
                )

                # Marcar en pick_alerts
                await s.execute(
                    text(
                        """
                        UPDATE pick_alerts SET
                            closing_pinn_odds = :po,
                            closing_captured_at = :cap
                        WHERE id = :pid
                        """
                    ),
                    {"pid": row.pick_id, "po": pinn_odds, "cap": now},
                )
                captured += 1
            except Exception as exc:
                logger.warning("clv.capture.pick_fail", pick=row.pick_id, error=str(exc)[:80])

        await s.commit()

    logger.info("clv.capture.done", captured=captured, n_window=len(rows))
    try:
        from apuestas.obs.metrics import inc_closing_lines_captured as _m

        _m("all", captured)
    except Exception:
        pass
    return captured


def compute_clv_pct(
    *, odds_at_pick: float, odds_closing: float, implied_is_fair: bool = False
) -> float:
    """CLV% = (odds_at_pick / odds_closing) - 1. Buchdahl 2023.

    Positivo = tomas odds mejores que el cierre (skill real).
    Ejemplo: tomaste @2.10, cerró @2.00 → CLV = 2.10/2.00 - 1 = +5%.

    Si `implied_is_fair=False` (default): compara odds raw.
    Si `True`: asume que ya están de-vigged.
    """
    if odds_at_pick <= 1.0 or odds_closing <= 1.0:
        return 0.0
    return (odds_at_pick / odds_closing) - 1.0


async def compute_clv_for_finished_picks() -> int:
    """Post-match: para picks con outcome_result y closing_pinn_odds, computa clv_pct."""
    updated = 0
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT id, odds_placed, closing_pinn_odds
                    FROM pick_alerts
                    WHERE outcome_result IS NOT NULL
                      AND outcome_result NOT IN ('expired', 'void')
                      AND closing_pinn_odds IS NOT NULL
                      AND clv_pct IS NULL
                    """
                )
            )
        ).fetchall()
        for r in rows:
            clv = compute_clv_pct(
                odds_at_pick=float(r.odds_placed),
                odds_closing=float(r.closing_pinn_odds),
            )
            await s.execute(
                text("UPDATE pick_alerts SET clv_pct = :clv WHERE id = :pid"),
                {"clv": clv, "pid": r.id},
            )
            updated += 1
        await s.commit()
    logger.info("clv.compute.done", updated=updated)
    return updated


async def clv_rolling_stats(days: int = 30) -> dict:
    """Stats rolling: CLV avg, % positivos, distribución por sport."""
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT
                    COUNT(*) AS n_picks,
                    AVG(clv_pct) * 100 AS clv_avg_pct,
                    COUNT(*) FILTER (WHERE clv_pct > 0)::float / NULLIF(COUNT(*), 0) * 100
                        AS positive_pct,
                    MIN(clv_pct) * 100 AS clv_min,
                    MAX(clv_pct) * 100 AS clv_max
                FROM pick_alerts
                WHERE clv_pct IS NOT NULL
                  AND placed_at >= NOW() - MAKE_INTERVAL(days => :d)
                """
            ),
            {"d": days},
        )
        row = r.first()
    if row is None or row.n_picks == 0:
        return {"n_picks": 0, "clv_avg_pct": 0.0, "positive_pct": 0.0}
    return {
        "n_picks": int(row.n_picks),
        "clv_avg_pct": float(row.clv_avg_pct or 0.0),
        "positive_pct": float(row.positive_pct or 0.0),
        "clv_min": float(row.clv_min or 0.0),
        "clv_max": float(row.clv_max or 0.0),
    }


async def _main_async() -> int:
    n = await capture_closing_lines_flow()
    m = await compute_clv_for_finished_picks()
    stats = await clv_rolling_stats(30)
    print(f"✓ Captured {n} closing lines, computed CLV for {m} finished picks")
    print(f"CLV rolling 30d: {stats}")
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "capture_closing_lines_flow",
    "clv_rolling_stats",
    "compute_clv_for_finished_picks",
    "compute_clv_pct",
]
