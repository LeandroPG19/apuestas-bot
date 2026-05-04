"""Lookup de midpoints Polymarket + Kalshi para consensus en emit_alerts.

Plan §6 wire. Dado un match (home/away/sport), busca el midpoint más
reciente publicado por cada fuente cross-market. Usamos fuzzy match por
nombre de equipo en `polymarket_prices.question` y `kalshi_prices.title`.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz
from sqlalchemy import text


def _best_row_match(rows: list[Any], home: str, away: str, text_col: str) -> Any | None:
    """Retorna la row cuyo texto tiene mayor fuzzy score contra home/away."""
    if not rows:
        return None
    home_norm = home.lower()
    away_norm = away.lower()
    best = None
    best_score = 0
    for r in rows:
        title = str(getattr(r, text_col, "") or "").lower()
        score = max(
            fuzz.partial_ratio(home_norm, title),
            fuzz.partial_ratio(away_norm, title),
        )
        if score > best_score:
            best_score = score
            best = r
    # Umbral 60 — fuzzy es permisivo; un match bajo significa que no hay
    # mercado correspondiente y debemos retornar None.
    return best if best_score >= 60 else None


async def fetch_polymarket_midpoint(
    session: Any, *, match_id: int
) -> tuple[float, float | None] | None:
    """Retorna (midpoint, volume_usd) del mejor match Polymarket reciente."""
    meta = (
        await session.execute(
            text(
                """
                SELECT ht.name AS home, at.name AS away, m.sport_code
                FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE m.id = :mid
                """
            ),
            {"mid": match_id},
        )
    ).first()
    if meta is None:
        return None

    rows = (
        await session.execute(
            text(
                """
                SELECT question, midpoint, volume_usd
                FROM polymarket_prices
                WHERE sport = :sp AND captured_at > now() - interval '12 hours'
                ORDER BY captured_at DESC LIMIT 200
                """
            ),
            {"sp": meta.sport_code.lower()},
        )
    ).all()
    hit = _best_row_match(rows, meta.home, meta.away, "question")
    if hit is None:
        return None
    try:
        return float(hit.midpoint), (float(hit.volume_usd) if hit.volume_usd else None)
    except (TypeError, ValueError):
        return None


async def fetch_kalshi_midpoint(session: Any, *, match_id: int) -> float | None:
    """Retorna yes_midpoint del mejor match Kalshi reciente."""
    meta = (
        await session.execute(
            text(
                """
                SELECT ht.name AS home, at.name AS away, m.sport_code
                FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE m.id = :mid
                """
            ),
            {"mid": match_id},
        )
    ).first()
    if meta is None:
        return None

    rows = (
        await session.execute(
            text(
                """
                SELECT title, yes_midpoint
                FROM kalshi_prices
                WHERE sport = :sp AND captured_at > now() - interval '12 hours'
                ORDER BY captured_at DESC LIMIT 200
                """
            ),
            {"sp": meta.sport_code.lower()},
        )
    ).all()
    hit = _best_row_match(rows, meta.home, meta.away, "title")
    if hit is None:
        return None
    try:
        return float(hit.yes_midpoint)
    except (TypeError, ValueError):
        return None


__all__ = ["fetch_kalshi_midpoint", "fetch_polymarket_midpoint"]
