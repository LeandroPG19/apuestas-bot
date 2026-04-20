"""Mercados por periodo (F5 MLB, Q1/H1 NBA/NFL) — exploit Voulgaris.

Voulgaris descubrió que sportsbooks ponían:
    H1_total = game_total / 2
    H2_total = game_total / 2

Pero estadísticamente H1 y H2 NO son simétricos:
- NBA: Q4 tiende a scoring mayor (clutch isolation + fouls)
- NFL: 1H tiene menos scoring (prevent defense late)
- MLB: innings 1-5 tienen starting pitcher fresco (F5 < half)

Este módulo:
1. Detecta líneas H1/F5 asimétricas y marca edge.
2. Emite pick si la asimetría > threshold basado en histórico real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Deporte → fracción histórica de scoring en "primer periodo"
#   NBA Q1: 24%  ·  NBA H1: 50%  ·  NFL 1H: 46%  ·  MLB F5: 53%
HISTORICAL_FIRST_HALF_SHARE: dict[str, dict[str, float]] = {
    "nba": {"H1": 0.50, "Q1": 0.246, "Q1_Q2": 0.50, "Q1_Q2_Q3": 0.74},
    "nfl": {"H1": 0.46, "Q1": 0.22},
    "mlb": {"F5": 0.527, "F3": 0.32},  # first 5/first 3 innings
    "nhl": {"P1": 0.33, "P1_P2": 0.67},
}


@dataclass(slots=True)
class PeriodMarketEdge:
    market_code: str  # "H1_total", "F5_total", ...
    expected_period_share: float
    book_implicit_share: float
    edge_direction: str  # "over" | "under"
    edge_magnitude: float


def compute_period_edge(
    *,
    sport_code: str,
    period_market: str,
    game_total: float,
    period_total: float,
) -> PeriodMarketEdge | None:
    """Detecta asimetría entre line del período vs proporción histórica.

    Args:
        sport_code: "nba" | "nfl" | "mlb" | "nhl"
        period_market: "H1" | "Q1" | "F5" | ...
        game_total: línea total del game (ej. 220.5 NBA)
        period_total: línea total del período (ej. H1 110.0)

    Returns:
        PeriodMarketEdge si hay asimetría > 2%, None si mercado eficiente.
    """
    shares = HISTORICAL_FIRST_HALF_SHARE.get(sport_code) or {}
    expected = shares.get(period_market)
    if expected is None or game_total <= 0:
        return None

    implicit = period_total / game_total
    gap = expected - implicit
    if abs(gap) < 0.02:
        return None

    direction = "over" if gap > 0 else "under"
    return PeriodMarketEdge(
        market_code=f"{period_market.lower()}_total",
        expected_period_share=expected,
        book_implicit_share=implicit,
        edge_direction=direction,
        edge_magnitude=abs(gap),
    )


async def seed_historical_shares_from_pbp(sport_code: str = "nba") -> dict[str, float]:
    """Calibra HISTORICAL_FIRST_HALF_SHARE desde play_by_play real.

    Refresca las fracciones cada temporada con datos propios (más precisos
    que los hardcoded si tienes ≥300 games en BD).
    """
    async with session_scope() as s:
        if sport_code == "nba":
            r = await s.execute(
                text(
                    """
                    WITH period_scores AS (
                        SELECT
                            match_id, period,
                            MAX(home_score + away_score) -
                            COALESCE(LAG(MAX(home_score + away_score))
                                OVER (PARTITION BY match_id ORDER BY period), 0)
                            AS period_points
                        FROM play_by_play
                        WHERE sport_code = 'nba' AND period <= 4
                        GROUP BY match_id, period
                    ),
                    totals AS (
                        SELECT match_id, SUM(period_points) AS total
                        FROM period_scores GROUP BY match_id
                    )
                    SELECT
                        ps.period,
                        SUM(ps.period_points)::float / NULLIF(SUM(t.total), 0) AS share
                    FROM period_scores ps
                    JOIN totals t ON t.match_id = ps.match_id
                    GROUP BY ps.period
                    ORDER BY ps.period
                    """
                )
            )
            rows = r.all()
        else:
            rows = []

    if not rows:
        logger.info("half_period.insufficient_data", sport=sport_code)
        return {}

    shares = {f"Q{row.period}": float(row.share or 0) for row in rows if row.share}
    # Deriva H1 = Q1 + Q2
    if "Q1" in shares and "Q2" in shares:
        shares["H1"] = shares["Q1"] + shares["Q2"]
    logger.info("half_period.shares_calibrated", sport=sport_code, shares=shares)
    return shares


def detect_asymmetric_picks(
    *,
    sport_code: str,
    game_total_line: float,
    period_odds: list[dict[str, Any]],
) -> list[PeriodMarketEdge]:
    """Escanea una lista de mercados por período y retorna edges."""
    picks: list[PeriodMarketEdge] = []
    for p in period_odds:
        edge = compute_period_edge(
            sport_code=sport_code,
            period_market=p["period"],
            game_total=game_total_line,
            period_total=float(p["line"]),
        )
        if edge and edge.edge_magnitude >= 0.025:
            picks.append(edge)
    return picks
