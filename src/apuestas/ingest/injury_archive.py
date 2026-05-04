"""Injury archive — recupera historial de injuries via Wayback Machine.

Fase 0.6 del plan. ESPN/Rotowire exponen injuries actuales pero no histórico.
Wayback Machine permite scrape snapshots semanales de `espn.com/nba/injuries`
y recuperar estado de injuries históricas. Útil para backfill de features
`player_injured_previous_game` y `team_injury_count_at_match_time`.

Uso:
    apuestas archive-injuries --sport nba --weeks 52
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_WAYBACK_API = "https://archive.org/wayback/available"
_ESPN_BASE = {
    "nba": "https://www.espn.com/nba/injuries",
    "nfl": "https://www.espn.com/nfl/injuries",
    "mlb": "https://www.espn.com/mlb/injuries",
    "nhl": "https://www.espn.com/nhl/injuries",
}


async def fetch_wayback_snapshot(url: str, timestamp_yyyymmdd: str) -> str | None:
    """Fetch snapshot archivado vía Wayback API."""
    params = {"url": url, "timestamp": timestamp_yyyymmdd}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            r = await client.get(_WAYBACK_API, params=params)
            data = r.json()
            closest = data.get("archived_snapshots", {}).get("closest", {})
            if not closest.get("available"):
                return None
            snapshot_url = closest.get("url")
            if not snapshot_url:
                return None
            snap = await client.get(snapshot_url)
            return str(snap.text) if snap.status_code == 200 else None
        except Exception as exc:
            logger.debug(
                "injury_archive.wayback_fail",
                url=url,
                ts=timestamp_yyyymmdd,
                error=str(exc)[:80],
            )
            return None


def _parse_espn_injuries_html(html: str) -> list[dict[str, Any]]:
    """Extracción mínima de tabla injuries ESPN (legacy + modern layouts)."""
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    rows: list[dict[str, Any]] = []
    # Modern ESPN: div.Table__TR con .AnchorLink player, .TextTableCell status
    for row in tree.css("tr.Table__TR, table.tablehead tr.odd, table.tablehead tr.even"):
        cells = row.css("td")
        if len(cells) < 3:
            continue
        player = cells[0].text(strip=True)
        position = cells[1].text(strip=True) if len(cells) > 1 else ""
        status = cells[-1].text(strip=True)
        if not player or "Name" in player:
            continue
        rows.append(
            {
                "player_name": player,
                "position": position,
                "status": status,
            }
        )
    return rows


async def _persist_archive_event(
    *, player_name: str, sport_code: str, status: str, snapshot_date: datetime
) -> int:
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                INSERT INTO injury_feed
                    (player_name_raw, source, status_reported, raw_text, reported_at)
                VALUES (:name, 'wayback_espn', :st, :raw, :ts)
                RETURNING id
                """
            ),
            {
                "name": player_name,
                "st": status[:100],
                "raw": f"archive_snapshot:{sport_code}:{snapshot_date.strftime('%Y-%m-%d')}",
                "ts": snapshot_date,
            },
        )
        row = r.first()
    return 1 if row else 0


async def backfill_sport(*, sport: str, weeks: int = 52) -> dict[str, int]:
    """Descarga snapshots semanales de los últimos N weeks."""
    url = _ESPN_BASE.get(sport)
    if not url:
        return {"snapshots": 0, "events": 0, "sport": 0}

    now = datetime.now(tz=UTC)
    events_persisted = 0
    snapshots_ok = 0

    for w in range(weeks):
        snap_date = now - timedelta(weeks=w)
        ts = snap_date.strftime("%Y%m%d")
        html = await fetch_wayback_snapshot(url, ts)
        if not html:
            continue
        snapshots_ok += 1
        try:
            rows = _parse_espn_injuries_html(html)
        except Exception as exc:
            logger.debug("injury_archive.parse_fail", sport=sport, error=str(exc)[:80])
            continue

        for row in rows:
            try:
                events_persisted += await _persist_archive_event(
                    player_name=row["player_name"],
                    sport_code=sport,
                    status=row["status"],
                    snapshot_date=snap_date,
                )
            except Exception as exc:
                logger.debug("injury_archive.persist_fail", error=str(exc)[:80])

        # Rate limit: Wayback permite ~1 req/s
        await asyncio.sleep(1.0)

    return {"snapshots": snapshots_ok, "events": events_persisted, "sport": 1}


async def main(args: argparse.Namespace) -> None:
    sports = [s.strip() for s in args.sport.split(",") if s.strip()]
    for sport in sports:
        try:
            result = await backfill_sport(sport=sport, weeks=args.weeks)
            print(f"✅ {sport}: {result}")
        except Exception as exc:
            print(f"❌ {sport}: {str(exc)[:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba", help="CSV sports")
    parser.add_argument("--weeks", type=int, default=12)
    asyncio.run(main(parser.parse_args()))
