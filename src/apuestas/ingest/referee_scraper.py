"""Referee scraper — extrae árbitros desde Sofascore event detail.

Poblar referees + match_referees permite activar features/referee_bias.py
que aporta ~5-10% WR edge en NBA/Soccer (§4.4 Voulgaris + Starlizard).

Uso:
    apuestas scrape-referees --sport soccer --days 7
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.sofascore_scraper import fetch_event_detail
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def _upsert_referee(
    session: Any, *, external_id: str, name: str, sport_code: str, role: str | None = None
) -> int | None:
    r = await session.execute(
        text(
            """
            INSERT INTO referees (external_id, name, sport_code, role)
            VALUES (:ext, :name, :sc, :role)
            ON CONFLICT (external_id) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """
        ),
        {"ext": external_id, "name": name, "sc": sport_code, "role": role},
    )
    row = r.first()
    return int(row.id) if row else None


async def _link_match_referee(
    session: Any, *, match_id: int, referee_id: int, role: str | None
) -> None:
    await session.execute(
        text(
            """
            INSERT INTO match_referees (match_id, referee_id, role)
            VALUES (:m, :r, :role)
            ON CONFLICT DO NOTHING
            """
        ),
        {"m": match_id, "r": referee_id, "role": role},
    )


async def scrape_referees_for_match(*, match_id: int, sofascore_event_id: int) -> int:
    """Fetch Sofascore event detail → extrae referee → persist."""
    detail = await fetch_event_detail(sofascore_event_id)
    if not detail:
        return 0
    event = detail.get("event", detail)
    referee = event.get("referee")
    if not referee:
        return 0

    sport_code = (
        (event.get("tournament", {}) or {})
        .get("category", {})
        .get("sport", {})
        .get("slug", "soccer")
    )
    if sport_code not in ("soccer", "basketball", "american-football", "baseball", "ice-hockey"):
        return 0
    # Normalizar a nuestro sport_code interno
    sport_map = {
        "basketball": "nba",
        "american-football": "nfl",
        "baseball": "mlb",
        "ice-hockey": "nhl",
    }
    sport_code = sport_map.get(sport_code, sport_code)

    async with session_scope() as session:
        ref_id = await _upsert_referee(
            session,
            external_id=f"sofascore:{referee.get('id')}",
            name=str(referee.get("name", "Unknown")),
            sport_code=sport_code,
            role=referee.get("country", {}).get("name") if referee.get("country") else None,
        )
        if ref_id is None:
            return 0
        await _link_match_referee(
            session,
            match_id=match_id,
            referee_id=ref_id,
            role="main",
        )
    return 1


async def scrape_recent_matches(*, sport: str = "soccer", days: int = 7) -> dict[str, int]:
    """Scan matches próximos N días con sofascore_event_id linkeado."""
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT m.id, (m.metadata ->> 'sofascore_event_id')::int AS sfid
                FROM matches m
                WHERE m.sport_code = :sc
                  AND m.start_time BETWEEN NOW() - INTERVAL '7 days'
                                       AND NOW() + INTERVAL ':d days'
                  AND m.metadata ->> 'sofascore_event_id' IS NOT NULL
                LIMIT 500
                """
            ),
            {"sc": sport, "d": days},
        )
        matches = r.all()

    results = {"checked": len(matches), "referees_added": 0}
    for match in matches:
        try:
            added = await scrape_referees_for_match(
                match_id=int(match.id), sofascore_event_id=int(match.sfid)
            )
            results["referees_added"] += added
        except Exception as exc:
            logger.debug(
                "referee_scraper.match_fail",
                match_id=match.id,
                error=str(exc)[:80],
            )
    return results


async def main(args: argparse.Namespace) -> None:
    result = await scrape_recent_matches(sport=args.sport, days=args.days)
    print(f"✅ Referees: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="soccer")
    parser.add_argument("--days", type=int, default=7)
    asyncio.run(main(parser.parse_args()))
