"""Bulk ingest StatsBomb Open Data — event-level soccer gratis.

Fuente: https://github.com/statsbomb/open-data
Licencia: CC BY-NC-SA 4.0 (no-comercial). USO ACADÉMICO/BACKTESTING OK.

Descarga competitions → matches → events JSON raw y los persiste en
`statsbomb_events`. Soporta ~75 competitions incluyendo:
- World Cup 2018, 2022
- Euros 2020, 2024
- La Liga masculina 2004-2020 (subset Messi era)
- La Liga femenina 2018-2020
- Premier League 2003-04
- Champions League finales 2003-2019
- NWSL 2018

Volumen: ~3k matches × 2k eventos = ~6M eventos.

Uso:
    uv run python scripts/ingest_statsbomb_open.py
    uv run python scripts/ingest_statsbomb_open.py --competition 43 --season 3  # WC 2018
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


async def _fetch_json(client: httpx.AsyncClient, url: str):  # type: ignore[no-untyped-def]
    try:
        r = await client.get(url, timeout=60.0)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as exc:
        logger.debug("sb.fetch_fail", url=url[-80:], error=str(exc)[:100])
        return None


async def ingest_match_events(
    client: httpx.AsyncClient,
    session,
    match_id: int,
    competition_id: int,
    season_id: int,
) -> int:  # type: ignore[no-untyped-def]
    url = f"{BASE_URL}/events/{match_id}.json"
    events = await _fetch_json(client, url)
    if events is None or len(events) == 0:
        return 0

    from sqlalchemy import text as _text

    # Dedup: borrar eventos del match si ya existen
    await session.execute(
        _text("DELETE FROM statsbomb_events WHERE match_id = :mid"),
        {"mid": match_id},
    )

    inserted = 0
    for ev in events:
        period = int(ev.get("period") or 1)
        minute = ev.get("minute")
        team = ev.get("team") or {}
        player = ev.get("player") or {}
        event_type_dict = ev.get("type") or {}
        await session.execute(
            _text(
                """
                INSERT INTO statsbomb_events (
                    competition_id, season_id, match_id, period, minute,
                    team_id, player_id, event_type, event_jsonb
                ) VALUES (
                    :cid, :sid, :mid, :p, :mn, :tid, :pid, :et, CAST(:ej AS jsonb)
                )
                """
            ),
            {
                "cid": competition_id,
                "sid": season_id,
                "mid": match_id,
                "p": period,
                "mn": int(minute) if minute is not None else None,
                "tid": int(team.get("id")) if team.get("id") else None,
                "pid": int(player.get("id")) if player.get("id") else None,
                "et": str(event_type_dict.get("name") or "")[:64],
                "ej": json.dumps(ev),
            },
        )
        inserted += 1
    await session.commit()
    return inserted


async def ingest_competition(
    client: httpx.AsyncClient,
    competition_id: int,
    season_id: int,
) -> int:
    """Ingesta todos los matches de una competition × season."""
    from apuestas.db import session_scope

    url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
    matches = await _fetch_json(client, url)
    if not matches:
        logger.info("sb.no_matches", cid=competition_id, sid=season_id)
        return 0

    total = 0
    for m in matches:
        match_id = int(m.get("match_id") or 0)
        if match_id == 0:
            continue
        try:
            async with session_scope() as session:
                n = await ingest_match_events(client, session, match_id, competition_id, season_id)
                total += n
        except Exception as exc:
            logger.warning("sb.match_fail", match=match_id, error=str(exc)[:80])
    logger.info(
        "sb.competition_done",
        competition=competition_id,
        season=season_id,
        matches=len(matches),
        events=total,
    )
    return total


async def main_async(competition_filter: int | None, season_filter: int | None) -> int:
    async with httpx.AsyncClient() as client:
        competitions = await _fetch_json(client, f"{BASE_URL}/competitions.json")
        if not competitions:
            logger.error("sb.no_competitions_index")
            return 1

        logger.info("sb.competitions_loaded", n=len(competitions))
        total = 0
        for comp in competitions:
            cid = int(comp.get("competition_id") or 0)
            sid = int(comp.get("season_id") or 0)
            if competition_filter is not None and cid != competition_filter:
                continue
            if season_filter is not None and sid != season_filter:
                continue
            try:
                n = await ingest_competition(client, cid, sid)
                total += n
            except Exception as exc:
                logger.warning("sb.competition_fail", cid=cid, sid=sid, error=str(exc)[:80])
        logger.info("sb.done", total_events=total)
        print(f"✓ Inserted {total} statsbomb events")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", type=int, default=None)
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args.competition, args.season))


if __name__ == "__main__":
    raise SystemExit(main())
