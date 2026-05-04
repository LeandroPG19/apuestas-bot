"""Popula `matches.external_id_odds_api` haciendo fuzzy match de los eventos
de The Odds API contra matches internos.

Endpoint `/events` cuesta 0 créditos (gratis incluso en paid tier).
Ejecutar pre-game cada ~30 min — una vez poblado, el flow de player props
puede usar directamente `/events/{id}/odds`.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rapidfuzz import fuzz
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apuestas.db import session_scope
from apuestas.flows.live_scores import _normalize_team
from apuestas.ingest.odds_api import OddsAPIClient
from apuestas.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Sport interno → list of odds_api keys a probar
_SPORT_TO_ODDS_KEYS: dict[str, list[str]] = {
    "nba": ["basketball_nba"],
    "nfl": ["americanfootball_nfl"],
    "mlb": ["baseball_mlb"],
    "nhl": ["icehockey_nhl"],
    "soccer": [
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_france_ligue_one",
        "soccer_uefa_champs_league",
        "soccer_mexico_ligamx",
        "soccer_usa_mls",
    ],
    "epl": ["soccer_epl"],
    "laliga": ["soccer_spain_la_liga"],
    "bundesliga": ["soccer_germany_bundesliga"],
    "seriea": ["soccer_italy_serie_a"],
    "ligue1": ["soccer_france_ligue_one"],
    "liga_mx": ["soccer_mexico_ligamx"],
}


async def _matches_missing_event_id(hours_ahead: int = 48) -> list[dict]:
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT m.id, m.sport_code, ht.name AS home, at.name AS away,
                       m.start_time
                FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE m.external_id_odds_api IS NULL
                  AND m.start_time BETWEEN NOW() AND NOW() + INTERVAL ':h hours'
                ORDER BY m.start_time ASC
                LIMIT 500
                """.replace(":h", str(hours_ahead))
            )
        )
        return [dict(row._mapping) for row in r.all()]


async def _update_event_id(match_id: int, event_id: str) -> None:
    async with session_scope() as session:
        await session.execute(
            text("UPDATE matches SET external_id_odds_api = :eid WHERE id = :mid"),
            {"eid": event_id, "mid": match_id},
        )


async def main() -> None:
    configure_logging()
    missing = await _matches_missing_event_id()
    logger.info("populate_odds_event.start", n_missing=len(missing))
    if not missing:
        return

    # Agrupar por sport para minimizar requests
    sport_to_matches: dict[str, list[dict]] = {}
    for m in missing:
        sport_to_matches.setdefault(str(m["sport_code"]), []).append(m)

    client = OddsAPIClient()
    linked = 0
    async with client.session():
        for sport_code, matches in sport_to_matches.items():
            odds_keys = _SPORT_TO_ODDS_KEYS.get(sport_code)
            if not odds_keys:
                continue

            # Fetch events de Odds API para cada key (1 req/key, 0 créditos)
            all_events: list[dict] = []
            for okey in odds_keys:
                try:
                    events = await client.list_events(
                        okey,
                        date_from=datetime.now(tz=UTC),
                        date_to=datetime.now(tz=UTC) + timedelta(hours=72),
                    )
                    for e in events:
                        e["_odds_key"] = okey
                    all_events.extend(events)
                except Exception as exc:
                    logger.debug(
                        "populate_odds_event.list_fail",
                        sport=sport_code,
                        key=okey,
                        error=str(exc)[:100],
                    )

            if not all_events:
                continue

            # Lookup por (home_norm, away_norm, date_str)
            by_key: dict[tuple[str, str, str], str] = {}
            for e in all_events:
                try:
                    commence = e.get("commence_time") or ""
                    date_str = commence[:10]
                    key = (
                        _normalize_team(e.get("home_team", "")),
                        _normalize_team(e.get("away_team", "")),
                        date_str,
                    )
                    by_key[key] = e["id"]
                except Exception:
                    continue

            for m in matches:
                date_str = m["start_time"].strftime("%Y-%m-%d")
                target = (
                    _normalize_team(m["home"]),
                    _normalize_team(m["away"]),
                    date_str,
                )
                eid = by_key.get(target)
                if eid is None:
                    # Fuzzy fallback
                    for (lh, la, ld), cand in by_key.items():
                        if ld != date_str:
                            continue
                        if fuzz.WRatio(target[0], lh) >= 85 and fuzz.WRatio(target[1], la) >= 85:
                            eid = cand
                            break
                if eid:
                    await _update_event_id(int(m["id"]), eid)
                    linked += 1

    credits = client.remaining_credits()
    logger.info(
        "populate_odds_event.done",
        linked=linked,
        missing=len(missing),
        credits_remaining=credits.get("remaining"),
    )


if __name__ == "__main__":
    asyncio.run(main())
