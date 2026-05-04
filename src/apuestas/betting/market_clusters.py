"""Fase 4.16 — Cross-market cluster detection.

Cuando 2+ mercados mispricing la misma causa (ej. over 2.5 goals + BTTS + anytime
scorer Haaland todos sub-priced) → edge concentrado. Un detector de clusters
flag "sharp conviction": si 3+ markets mismo match con EV≥3% → bump Kelly
portfolio.

API:
    detect_correlated_mispricings(match_id) -> list[MispriceCluster]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class MispriceCluster:
    match_id: int
    market_ids_in_cluster: list[str]  # ej ["over_2.5", "btts_yes", "haaland_anytime"]
    n_markets: int
    avg_ev: float
    conviction_score: float  # 0-1, combina EVs + coherence
    detected_at: datetime


async def detect_correlated_mispricings(
    match_id: int,
    *,
    min_cluster_size: int = 3,
    min_ev_per_pick: float = 0.03,
) -> MispriceCluster | None:
    """Detecta clusters. Si ≥3 markets EV≥3% en el mismo match → cluster fuerte."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT p.market, p.outcome, p.ev
                    FROM predictions p
                    WHERE p.match_id = :mid
                      AND p.ev >= :min_ev
                      AND p.test_data = false
                      AND p.created_at > now() - interval '2 hours'
                    ORDER BY p.ev DESC
                    """
                ),
                {"mid": match_id, "min_ev": min_ev_per_pick},
            )
        ).all()

    if len(rows) < min_cluster_size:
        return None

    market_ids = [f"{r.market}:{r.outcome}" for r in rows]
    avg_ev = sum(float(r.ev) for r in rows) / len(rows)
    # Conviction score: combina n_markets (más = mejor) + avg EV (más = mejor)
    # Normalizado a [0, 1]
    conviction = min(1.0, (len(rows) / 6.0) * 0.5 + min(1.0, avg_ev / 0.10) * 0.5)

    logger.info(
        "market_clusters.detected",
        match_id=match_id,
        n_markets=len(rows),
        avg_ev=avg_ev,
        conviction=conviction,
    )

    return MispriceCluster(
        match_id=match_id,
        market_ids_in_cluster=market_ids,
        n_markets=len(rows),
        avg_ev=avg_ev,
        conviction_score=conviction,
        detected_at=datetime.now(tz=UTC),
    )


async def scan_all_upcoming_matches(
    *,
    hours_ahead: int = 48,
    min_cluster_size: int = 3,
) -> list[MispriceCluster]:
    """Escanea todos los matches próximos. Útil para cron periódico."""
    since = datetime.now(tz=UTC)
    until = since + timedelta(hours=hours_ahead)

    async with session_scope() as session:
        match_ids = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT p.match_id
                    FROM predictions p
                    JOIN matches m ON m.id = p.match_id
                    WHERE m.start_time BETWEEN :since AND :until
                      AND p.ev >= 0.03
                      AND p.test_data = false
                    """
                ),
                {"since": since, "until": until},
            )
        ).all()

    clusters: list[MispriceCluster] = []
    for row in match_ids:
        c = await detect_correlated_mispricings(row.match_id, min_cluster_size=min_cluster_size)
        if c is not None:
            clusters.append(c)

    logger.info(
        "market_clusters.scan_complete",
        n_matches=len(match_ids),
        n_clusters=len(clusters),
    )
    return clusters
