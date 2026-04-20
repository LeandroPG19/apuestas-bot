"""Odds spike detection + soft line alert (§17.4).

Dos detectores corriendo cada 5 min durante sesión:

1. Pricing error: odds se mueve >15% en <10 min sin movimiento correlacionado
   en otros books = probable error del bookmaker → alerta instantánea.
2. Soft line: odds Caliente/Strendus se aleja >5% de consenso sharp y hay
   EV+. Prioridad alta (se cierra pronto).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import text

from apuestas.betting.devig import shin
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

SoftTag = Literal["pricing_error", "soft_line", "steam_move"]


@dataclass(slots=True)
class SpikeAlert:
    match_id: int
    market: str
    outcome: str
    bookmaker: str
    tag: SoftTag
    odds_before: float
    odds_after: float
    pct_move: float
    detected_at: datetime
    details: dict[str, float | str] | None = None


async def detect_pricing_errors(
    *,
    window_minutes: int = 10,
    move_threshold_pct: float = 0.15,
    lookback_minutes: int = 30,
) -> list[SpikeAlert]:
    """Busca cambios abruptos aislados en un bookmaker.

    Query: por (match, market, outcome, bookmaker) encontrar pares de rows
    donde |delta_pct| > threshold en < window_minutes, y el movimiento NO
    está replicado en al menos otro bookmaker.
    """
    now = datetime.now(tz=UTC)
    cutoff = now - timedelta(minutes=lookback_minutes)

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH changes AS (
                  SELECT match_id, market, outcome, bookmaker,
                         ts, odds,
                         LAG(odds) OVER (
                           PARTITION BY match_id, market, outcome, bookmaker
                           ORDER BY ts
                         ) AS prev_odds,
                         LAG(ts) OVER (
                           PARTITION BY match_id, market, outcome, bookmaker
                           ORDER BY ts
                         ) AS prev_ts
                  FROM odds_history
                  WHERE ts >= :cutoff
                )
                SELECT match_id, market, outcome, bookmaker,
                       odds, prev_odds, ts, prev_ts
                FROM changes
                WHERE prev_odds IS NOT NULL
                  AND ABS(odds - prev_odds) / prev_odds > :thr
                  AND EXTRACT(EPOCH FROM (ts - prev_ts)) < :window_sec
                """
            ),
            {
                "cutoff": cutoff,
                "thr": move_threshold_pct,
                "window_sec": window_minutes * 60,
            },
        )
        candidates = [dict(r._mapping) for r in result.all()]

    alerts: list[SpikeAlert] = []
    for c in candidates:
        pct = float(c["odds"] - c["prev_odds"]) / float(c["prev_odds"])

        # ¿Algún otro book replicó el movimiento?
        replicated = await _was_replicated_by_other_books(
            match_id=int(c["match_id"]),
            market=str(c["market"]),
            outcome=str(c["outcome"]),
            moving_bookmaker=str(c["bookmaker"]),
            window_start=c["prev_ts"],
            window_end=c["ts"],
            pct_threshold=move_threshold_pct / 2,
        )
        if replicated:
            # Es un steam move (sharp movement cross-book), no error
            alerts.append(
                SpikeAlert(
                    match_id=int(c["match_id"]),
                    market=str(c["market"]),
                    outcome=str(c["outcome"]),
                    bookmaker=str(c["bookmaker"]),
                    tag="steam_move",
                    odds_before=float(c["prev_odds"]),
                    odds_after=float(c["odds"]),
                    pct_move=pct,
                    detected_at=now,
                    details={"replicated_by_other_books": "true"},
                )
            )
        else:
            alerts.append(
                SpikeAlert(
                    match_id=int(c["match_id"]),
                    market=str(c["market"]),
                    outcome=str(c["outcome"]),
                    bookmaker=str(c["bookmaker"]),
                    tag="pricing_error",
                    odds_before=float(c["prev_odds"]),
                    odds_after=float(c["odds"]),
                    pct_move=pct,
                    detected_at=now,
                )
            )

    logger.info(
        "odds_spike.pricing_errors",
        n=len([a for a in alerts if a.tag == "pricing_error"]),
        n_steam=len([a for a in alerts if a.tag == "steam_move"]),
    )
    return alerts


async def _was_replicated_by_other_books(
    *,
    match_id: int,
    market: str,
    outcome: str,
    moving_bookmaker: str,
    window_start: datetime,
    window_end: datetime,
    pct_threshold: float,
) -> bool:
    """¿Otro book también movió > threshold en la misma ventana temporal?"""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH same_window AS (
                  SELECT bookmaker, ts, odds,
                         LAG(odds) OVER (PARTITION BY bookmaker ORDER BY ts) AS prev
                  FROM odds_history
                  WHERE match_id = :match_id
                    AND market = :market
                    AND outcome = :outcome
                    AND bookmaker <> :moving
                    AND ts BETWEEN :ws AND :we
                )
                SELECT COUNT(*) AS cnt FROM same_window
                WHERE prev IS NOT NULL
                  AND ABS(odds - prev) / prev >= :thr
                """
            ),
            {
                "match_id": match_id,
                "market": market,
                "outcome": outcome,
                "moving": moving_bookmaker,
                "ws": window_start,
                "we": window_end + timedelta(minutes=2),
                "thr": pct_threshold,
            },
        )
        row = result.first()
    return int(row.cnt or 0) > 0


async def detect_soft_lines(
    *,
    soft_books: tuple[str, ...] = ("caliente", "strendus", "codere"),
    spread_threshold: float = 0.05,
    lookback_minutes: int = 30,
) -> list[SpikeAlert]:
    """Odds soft que se alejan del fair Pinnacle.

    Pipeline: por cada (match, market, outcome) toma última odds de cada book,
    de-viga Pinnacle+Circa para obtener fair, compara soft vs fair.
    """
    now = datetime.now(tz=UTC)
    cutoff = now - timedelta(minutes=lookback_minutes)

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH latest AS (
                  SELECT DISTINCT ON (match_id, market, outcome, bookmaker)
                    match_id, market, outcome, bookmaker, odds, line, ts
                  FROM odds_history
                  WHERE ts >= :cutoff
                  ORDER BY match_id, market, outcome, bookmaker, ts DESC
                )
                SELECT * FROM latest
                ORDER BY match_id, market, outcome
                """
            ),
            {"cutoff": cutoff},
        )
        rows = [dict(r._mapping) for r in result.all()]

    # Agrupar por (match_id, market, line)
    grouped: dict[tuple[int, str, float | None], dict[str, dict[str, float]]] = {}
    for r in rows:
        key = (int(r["match_id"]), str(r["market"]), r["line"])
        grouped.setdefault(key, {}).setdefault(str(r["outcome"]), {})[str(r["bookmaker"])] = float(
            r["odds"]
        )

    alerts: list[SpikeAlert] = []
    for (match_id, market, line), outcomes in grouped.items():
        sharp_odds = {}
        soft_outcomes: dict[str, dict[str, float]] = {}

        # Recolectar sharp y soft por outcome
        for outcome, books in outcomes.items():
            sharp_for_outcome = {
                b: o for b, o in books.items() if b in ("pinnacle", "circa", "betfair", "bookmaker")
            }
            soft_for_outcome = {b: o for b, o in books.items() if b in soft_books}
            if sharp_for_outcome:
                sharp_odds[outcome] = sharp_for_outcome
            if soft_for_outcome:
                soft_outcomes[outcome] = soft_for_outcome

        if not sharp_odds or not soft_outcomes:
            continue

        # Orden canónico para shin
        outcomes_order = sorted(sharp_odds)
        try:
            sharp_row = [
                min(sharp_odds[o].values()) if sharp_odds[o] else float("inf")
                for o in outcomes_order
            ]
            fair_probs = shin(sharp_row)
            fair_odds_vec = 1.0 / fair_probs
        except (ValueError, ZeroDivisionError):  # fmt: skip
            continue

        for i, outcome in enumerate(outcomes_order):
            fair_o = float(fair_odds_vec[i])
            for soft_book, soft_o in soft_outcomes.get(outcome, {}).items():
                rel = (soft_o - fair_o) / fair_o
                if rel >= spread_threshold:
                    alerts.append(
                        SpikeAlert(
                            match_id=match_id,
                            market=market,
                            outcome=outcome,
                            bookmaker=soft_book,
                            tag="soft_line",
                            odds_before=fair_o,
                            odds_after=soft_o,
                            pct_move=rel,
                            detected_at=now,
                            details={
                                "fair_odds_estimate": fair_o,
                                "spread_vs_fair_pct": rel,
                                "line": str(line) if line is not None else "none",
                            },
                        )
                    )

    logger.info("odds_spike.soft_lines", n=len(alerts))
    return alerts


async def run_all_detectors() -> list[SpikeAlert]:
    """Ejecuta ambos detectores en paralelo."""
    import asyncio

    pricing_task = detect_pricing_errors()
    soft_task = detect_soft_lines()
    results = await asyncio.gather(pricing_task, soft_task, return_exceptions=True)
    alerts: list[SpikeAlert] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("odds_spike.detector_error", error=str(r))
            continue
        alerts.extend(r)
    return alerts
