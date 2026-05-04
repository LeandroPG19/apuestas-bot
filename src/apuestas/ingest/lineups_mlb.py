"""MLB lineups ingester — Fase 1 wire (Sprint 14 #147 support).

Consume MLB Stats API (statsapi.mlb.com) gratuito para fetch probable
pitchers + confirmed lineups ~2h pre-kickoff.

Tabla `match_lineups` (CREATE IF NOT EXISTS):
  match_id bigint PK
  home_starting_pitcher_id bigint
  away_starting_pitcher_id bigint
  home_lineup jsonb
  away_lineup jsonb
  is_confirmed bool
  updated_at timestamptz

Uso:
  python -m apuestas.ingest.lineups_mlb  # ingest hoy+mañana
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def ensure_match_lineups_table() -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS match_lineups (
                  match_id bigint PRIMARY KEY,
                  home_starting_pitcher_id bigint,
                  away_starting_pitcher_id bigint,
                  home_lineup jsonb,
                  away_lineup jsonb,
                  is_confirmed bool DEFAULT false,
                  updated_at timestamptz DEFAULT now()
                )
                """
            )
        )


async def fetch_mlb_schedule_lineups(date: datetime | None = None) -> int:
    """Fetch MLB schedule + probablePitcher via Stats API.

    Gratis, sin auth. Rate limit ~500 req/5min.
    Populate match_lineups.
    """
    await ensure_match_lineups_table()
    target = (date or datetime.now(tz=UTC)).strftime("%Y-%m-%d")
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={target}&hydrate=probablePitcher,lineups"
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
        if r.status_code != 200:
            logger.warning("lineups_mlb.http_fail", status=r.status_code)
            return 0
        data = r.json()
    except Exception as exc:
        logger.warning("lineups_mlb.fetch_fail", error=str(exc)[:80])
        return 0

    n_updated = 0
    async with session_scope() as s:
        for day in data.get("dates", []):
            for game in day.get("games", []):
                game_pk = game.get("gamePk")
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                home_pp = home.get("probablePitcher", {}).get("id")
                away_pp = away.get("probablePitcher", {}).get("id")

                # Match DB match_id: first by external_id exact, then fuzzy team+date
                home_name = home.get("team", {}).get("name", "")
                away_name = away.get("team", {}).get("name", "")
                from datetime import date as _date

                gd_str = game.get("gameDate", "")[:10]
                game_date = _date.fromisoformat(gd_str) if gd_str else None
                db_match = (
                    (
                        await s.execute(
                            text(
                                """
                            SELECT m.id FROM matches m
                            JOIN teams ht ON ht.id=m.home_team_id
                            JOIN teams at2 ON at2.id=m.away_team_id
                            WHERE m.sport_code='mlb'
                              AND (
                                m.external_id LIKE :pk
                                OR m.external_id_odds_api LIKE :pk
                                OR (
                                  (ht.name ILIKE :hlk OR :hn ILIKE '%' || ht.name || '%')
                                  AND (at2.name ILIKE :alk OR :an ILIKE '%' || at2.name || '%')
                                  AND DATE(m.start_time) = :dt
                                )
                              )
                            LIMIT 1
                            """
                            ),
                            {
                                "pk": f"%{game_pk}%",
                                "hn": home_name,
                                "hlk": f"%{home_name}%",
                                "an": away_name,
                                "alk": f"%{away_name}%",
                                "dt": game_date,
                            },
                        )
                    ).first()
                    if game_date
                    else None
                )
                if db_match is None:
                    continue

                await s.execute(
                    text(
                        """
                        INSERT INTO match_lineups (
                          match_id, home_starting_pitcher_id, away_starting_pitcher_id,
                          is_confirmed, updated_at
                        ) VALUES (:mid, :hp, :ap, :conf, NOW())
                        ON CONFLICT (match_id) DO UPDATE SET
                          home_starting_pitcher_id=EXCLUDED.home_starting_pitcher_id,
                          away_starting_pitcher_id=EXCLUDED.away_starting_pitcher_id,
                          updated_at=NOW()
                        """
                    ),
                    {
                        "mid": int(db_match.id),
                        "hp": int(home_pp) if home_pp else None,
                        "ap": int(away_pp) if away_pp else None,
                        "conf": bool(home.get("probablePitcher") and away.get("probablePitcher")),
                    },
                )
                n_updated += 1
    logger.info("lineups_mlb.ingested", n=n_updated, date=target)
    return n_updated


async def main():
    today = datetime.now(tz=UTC)
    tomorrow = today + timedelta(days=1)
    n1 = await fetch_mlb_schedule_lineups(today)
    n2 = await fetch_mlb_schedule_lineups(tomorrow)
    print(f"MLB lineups ingested: today={n1} tomorrow={n2}")


if __name__ == "__main__":
    asyncio.run(main())
