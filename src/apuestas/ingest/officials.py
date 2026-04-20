"""Ingesta de árbitros/umpires con stats agregadas.

Fuentes:
- NBA: nba.com/officials + Basketball-Reference
- NFL: nflreadpy officials
- MLB: Umpire Scorecards (umpscorecards.com) scraping
- Soccer: FBref + API-Football

Para el MVP se implementan stubs y el scraping se delega a cuba-search
para BoxRec/Umpire Scorecards cuando no hay API oficial.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Métricas mínimas esperadas por deporte (para validación posterior)
EXPECTED_METRICS: dict[str, list[str]] = {
    "nfl": ["home_win_rate", "penalties_per_game", "ot_rate"],
    "nba": ["home_win_rate", "ft_per_game", "pace_differential", "home_cover_rate"],
    "mlb": ["strikezone_consistency", "runs_per_game", "home_win_rate"],
    "soccer": ["yellow_cards_avg", "red_cards_avg", "goals_per_game", "home_win_rate"],
}


async def upsert_official(
    *,
    name: str,
    external_id: str | None,
    sport_code: str,
    role: str | None = None,
    stats: dict[str, Any] | None = None,
) -> int:
    """Insert/update oficial. Retorna official_id."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                INSERT INTO officials (name, external_id, sport_code, role, stats, last_computed)
                VALUES (:name, :external_id, :sport_code, :role, :stats, NOW())
                ON CONFLICT (external_id) DO UPDATE
                  SET stats = EXCLUDED.stats,
                      last_computed = NOW()
                RETURNING id
                """
            ),
            {
                "name": name,
                "external_id": external_id,
                "sport_code": sport_code,
                "role": role,
                "stats": stats or {},
            },
        )
        row = result.first()
        if row is None:
            # No external_id → upsert by name
            result = await session.execute(
                text(
                    """
                    INSERT INTO officials (name, sport_code, role, stats, last_computed)
                    VALUES (:name, :sport_code, :role, :stats, NOW())
                    RETURNING id
                    """
                ),
                {
                    "name": name,
                    "sport_code": sport_code,
                    "role": role,
                    "stats": stats or {},
                },
            )
            row = result.first()
    return int(row[0]) if row else 0


async def link_match_official(*, match_id: int, official_id: int, role: str) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO match_officials (match_id, official_id, role)
                VALUES (:match_id, :official_id, :role)
                ON CONFLICT DO NOTHING
                """
            ),
            {"match_id": match_id, "official_id": official_id, "role": role},
        )


async def fetch_nfl_officials_from_pbp() -> int:
    """Extrae árbitros NFL del play-by-play nflreadpy.

    nflreadpy load_officials() devuelve referee por game_id.
    """
    try:
        import nflreadpy as nfl  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("officials.nfl.nflreadpy_missing")
        return 0

    import asyncio

    def _fetch() -> Any:
        return nfl.load_officials(seasons=[2024])

    df = await asyncio.to_thread(_fetch)
    if df is None or (hasattr(df, "height") and df.height == 0):
        return 0

    # nflreadpy devuelve Polars; iteramos y upsert
    count = 0
    for row in df.iter_rows(named=True) if hasattr(df, "iter_rows") else []:
        await upsert_official(
            name=row.get("official_name", ""),
            external_id=str(row.get("official_id", "")) if row.get("official_id") else None,
            sport_code="nfl",
            role=row.get("off_pos", "referee"),
        )
        count += 1

    logger.info("officials.nfl.ingested", count=count)
    return count


async def seed_mlb_umpires_stub() -> int:
    """Placeholder — Umpire Scorecards requiere scraping manual/cuba-search."""
    logger.info("officials.mlb.stub", msg="Implementar con cuba-search en Fase 9-10")
    return 0
