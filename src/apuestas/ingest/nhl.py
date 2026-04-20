"""Ingesta NHL — NHL Stats API oficial + MoneyPuck CSV (§25.2).

NHL API endpoints:
- /v1/schedule/{date}    — games del día
- /v1/gamecenter/{game_id}/boxscore
- /v1/player/{id}/landing
- /v1/standings

MoneyPuck CSV drops:
- https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv
- https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/goalies/goalies.csv
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from typing import Any

import httpx
import polars as pl

from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_fixtures

logger = get_logger(__name__)


class NHLStatsAPIClient(BaseAPIClient):
    """NHL API oficial — gratis sin API key."""

    base_url = "https://api-web.nhle.com/v1"
    source_name = "nhl_api"
    rate_limit = (60, 60.0)

    async def fetch_schedule(self, date_str: str | None = None) -> dict[str, Any]:
        """Schedule games por fecha (default=hoy)."""
        date_str = date_str or datetime.now(tz=UTC).strftime("%Y-%m-%d")
        return await self.get(f"/schedule/{date_str}")

    async def fetch_boxscore(self, game_id: int) -> dict[str, Any]:
        return await self.get(f"/gamecenter/{game_id}/boxscore")

    async def fetch_season_schedule(self, team_abbr: str, season: str) -> dict[str, Any]:
        """Ej. team_abbr='EDM', season='20252026'."""
        return await self.get(f"/club-schedule-season/{team_abbr}/{season}")

    async def fetch_player_landing(self, player_id: int) -> dict[str, Any]:
        return await self.get(f"/player/{player_id}/landing")

    async def fetch_standings(self) -> dict[str, Any]:
        return await self.get("/standings/now")


def schedule_to_fixtures(data: dict[str, Any]) -> pl.DataFrame:
    """Normaliza schedule NHL a fixtures universal."""
    games = []
    for day in data.get("gameWeek", []):
        games.extend(day.get("games", []))
    if not games:
        # Fallback a formato /schedule/{date}
        games = data.get("games", [])

    if not games:
        return _empty_fixtures_df()

    rows: list[dict[str, Any]] = []
    status_map = {
        "FUT": "scheduled",
        "PRE": "scheduled",
        "LIVE": "live",
        "FINAL": "finished",
        "OFF": "finished",
    }
    for g in games:
        game_id = g.get("id")
        if not game_id:
            continue
        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        start = g.get("startTimeUTC")
        state = g.get("gameState", "FUT")
        season = g.get("season")
        rows.append(
            {
                "external_id": str(game_id),
                "sport_code": "nhl",
                "home_team_external_id": str(home.get("id") or home.get("abbrev", "")),
                "away_team_external_id": str(away.get("id") or away.get("abbrev", "")),
                "start_time": start,
                "status": status_map.get(state, "scheduled"),
                "league_external_id": "nhl",
                "season": str(season) if season else None,
            }
        )
    if not rows:
        return _empty_fixtures_df()
    df = pl.DataFrame(rows).with_columns(
        pl.col("start_time").str.to_datetime(time_zone="UTC", strict=False).alias("start_time")
    )
    return df.filter(
        pl.col("home_team_external_id").is_not_null()
        & pl.col("away_team_external_id").is_not_null()
    )


def _empty_fixtures_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "external_id": pl.Utf8,
            "sport_code": pl.Utf8,
            "home_team_external_id": pl.Utf8,
            "away_team_external_id": pl.Utf8,
            "start_time": pl.Datetime(time_zone="UTC"),
            "status": pl.Utf8,
            "league_external_id": pl.Utf8,
            "season": pl.Utf8,
        }
    )


async def ingest_nhl_schedule(date_str: str | None = None) -> pl.DataFrame:
    """Ingesta schedule NHL validado."""
    client = NHLStatsAPIClient()
    async with client.session():
        data = await client.fetch_schedule(date_str)
    df = schedule_to_fixtures(data)
    if df.height == 0:
        return df
    return validate_fixtures(df)


# ═══════════════════════ MoneyPuck CSV scraping ═══════════════════════════


async def fetch_moneypuck_teams(year: int = 2026) -> pl.DataFrame:
    """xG, Corsi, Fenwick por equipo."""
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/teams.csv"
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("nhl.moneypuck_teams_fail", year=year, error=str(exc))
            return pl.DataFrame()
    try:
        return pl.read_csv(StringIO(resp.text), infer_schema_length=5000)
    except Exception as exc:
        logger.warning("nhl.moneypuck_parse_fail", error=str(exc))
        return pl.DataFrame()


async def fetch_moneypuck_goalies(year: int = 2026) -> pl.DataFrame:
    """GSAx (Goals Saved Above Expected) por goalie."""
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/goalies.csv"
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("nhl.moneypuck_goalies_fail", year=year, error=str(exc))
            return pl.DataFrame()
    try:
        return pl.read_csv(StringIO(resp.text), infer_schema_length=5000)
    except Exception as exc:
        logger.warning("nhl.moneypuck_goalies_parse_fail", error=str(exc))
        return pl.DataFrame()
