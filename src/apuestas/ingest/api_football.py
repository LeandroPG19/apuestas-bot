"""Cliente API-Football (api-sports.io) — fixtures, odds, lineups, injuries.

Pro tier $19/mes: 7,500 req/día. Cubre Liga MX, Liga de Expansión MX, MLS,
Big-5 Europa, Champions, Europa League, Copa Libertadores, y 1,100+ ligas.

Rate limit planificado: 300 req/hora ≈ 7,200/día (margen 4%).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from apuestas.config import get_settings
from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import (
    validate_fixtures,
    validate_injuries,
    validate_lineups,
    validate_odds,
)

logger = get_logger(__name__)

LEAGUE_IDS: dict[str, int] = {
    "liga_mx": 262,
    "liga_expansion_mx": 263,
    "mls": 253,
    "epl": 39,
    "la_liga": 140,
    "bundesliga": 78,
    "serie_a": 135,
    "ligue_1": 61,
    "champions": 2,
    "europa": 3,
    "libertadores": 13,
    "copa_mx": 268,
    "brasileirao": 71,
}


class APIFootballClient(BaseAPIClient):
    base_url = "https://v3.football.api-sports.io"
    source_name = "api_football"
    rate_limit = (300, 3600.0)  # 300 req/hora = 7,200/día

    def __init__(self, *, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.apis.api_football_key.get_secret_value()
            if settings.apis.api_football_key
            else None
        )
        if not key:
            msg = "API_FOOTBALL_KEY requerida"
            raise ValueError(msg)
        super().__init__(api_key=key)
        self._key = key

    def _default_headers(self) -> dict[str, str]:
        return {
            "x-apisports-key": self._key,
            "User-Agent": "apuestas-bot/0.1 (+api_football)",
        }

    async def list_leagues(
        self, *, country: str | None = None, season: int | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season
        data = await self.get("/leagues", params=params)
        return data.get("response", [])

    async def fetch_fixtures(
        self,
        *,
        league: int,
        season: int,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"league": league, "season": season}
        if date_from:
            params["from"] = date_from.strftime("%Y-%m-%d")
        if date_to:
            params["to"] = date_to.strftime("%Y-%m-%d")
        if status:
            params["status"] = status
        data = await self.get("/fixtures", params=params)
        return data.get("response", [])

    async def fetch_odds(
        self,
        *,
        fixture_id: int | None = None,
        league: int | None = None,
        season: int | None = None,
        date: str | None = None,
        bookmaker: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if fixture_id:
            params["fixture"] = fixture_id
        if league:
            params["league"] = league
        if season:
            params["season"] = season
        if date:
            params["date"] = date
        if bookmaker:
            params["bookmaker"] = bookmaker
        data = await self.get("/odds", params=params)
        return data.get("response", [])

    async def fetch_injuries(
        self,
        *,
        league: int | None = None,
        team: int | None = None,
        fixture: int | None = None,
        season: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if league:
            params["league"] = league
        if team:
            params["team"] = team
        if fixture:
            params["fixture"] = fixture
        if season:
            params["season"] = season
        data = await self.get("/injuries", params=params)
        return data.get("response", [])

    async def fetch_lineups(self, *, fixture_id: int) -> list[dict[str, Any]]:
        data = await self.get("/fixtures/lineups", params={"fixture": fixture_id})
        return data.get("response", [])

    async def fetch_teams(self, *, league: int, season: int) -> list[dict[str, Any]]:
        data = await self.get("/teams", params={"league": league, "season": season})
        return data.get("response", [])

    async def fetch_team_stats(self, *, league: int, team: int, season: int) -> dict[str, Any]:
        data = await self.get(
            "/teams/statistics",
            params={"league": league, "team": team, "season": season},
        )
        return data.get("response", {})

    async def fetch_h2h(self, *, h2h: str, last: int = 10) -> list[dict[str, Any]]:
        """h2h format: 'teamA_id-teamB_id'."""
        data = await self.get("/fixtures/headtohead", params={"h2h": h2h, "last": last})
        return data.get("response", [])


# ═══════════════════════ Transformadores a Polars ═══════════════════════


def fixtures_to_polars(raw: list[dict[str, Any]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for f in raw:
        fixture = f.get("fixture", {})
        league = f.get("league", {})
        teams = f.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        status_short = fixture.get("status", {}).get("short", "NS")
        status_map = {
            "NS": "scheduled",
            "TBD": "scheduled",
            "PST": "postponed",
            "LIVE": "live",
            "1H": "live",
            "HT": "live",
            "2H": "live",
            "ET": "live",
            "P": "live",
            "BT": "live",
            "FT": "finished",
            "AET": "finished",
            "PEN": "finished",
            "CANC": "cancelled",
            "ABD": "cancelled",
            "AWD": "void",
            "WO": "void",
            "INT": "void",
        }
        rows.append(
            {
                "external_id": str(fixture.get("id", "")),
                "sport_code": "soccer",
                "home_team_external_id": str(home.get("id", "")),
                "away_team_external_id": str(away.get("id", "")),
                "start_time": fixture.get("date"),
                "status": status_map.get(status_short, "scheduled"),
                "league_external_id": str(league.get("id", "")) if league.get("id") else None,
                "season": str(league.get("season")) if league.get("season") else None,
            }
        )
    if not rows:
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
    return pl.DataFrame(rows).with_columns(
        pl.col("start_time").str.to_datetime(time_zone="UTC").alias("start_time")
    )


def odds_to_polars(raw: list[dict[str, Any]], ts: datetime) -> pl.DataFrame:
    """API-Football odds nested: response[i].fixture.id, bookmakers[j].bets[k].values[l]."""
    rows: list[dict[str, Any]] = []
    bookmaker_slug = {
        8: "bet365",
        6: "bwin",
        11: "1xbet",
        16: "betway",
        18: "betsson",
        20: "pinnacle",
        3: "bet365",
        4: "bwin",
        5: "betway",
    }
    bet_market_map = {
        "Match Winner": "h2h",
        "Goals Over/Under": "totals",
        "Both Teams To Score": "btts",
        "Asian Handicap": "asian_handicap",
        "Double Chance": "double_chance",
    }
    for entry in raw:
        fixture_id = str(entry.get("fixture", {}).get("id", ""))
        for bm in entry.get("bookmakers", []):
            bm_id = bm.get("id")
            bm_name = bookmaker_slug.get(bm_id, str(bm.get("name", "unknown")).lower())
            for bet in bm.get("bets", []):
                market = bet_market_map.get(bet.get("name", ""))
                if market is None:
                    continue
                for v in bet.get("values", []):
                    try:
                        odds_val = float(v.get("odd", 0))
                    except (TypeError, ValueError):  # fmt: skip
                        continue
                    if odds_val <= 1.0:
                        continue
                    outcome = str(v.get("value", ""))
                    line: float | None = None
                    if market in {"totals", "asian_handicap"} and " " in outcome:
                        try:
                            parts = outcome.split()
                            line = float(parts[-1])
                            outcome = parts[0]
                        except (ValueError, IndexError):  # fmt: skip
                            pass
                    rows.append(
                        {
                            "ts": ts,
                            "match_external_id": fixture_id,
                            "bookmaker": bm_name,
                            "market": market,
                            "outcome": outcome,
                            "line": line,
                            "odds": odds_val,
                        }
                    )
    if not rows:
        return pl.DataFrame(
            schema={
                "ts": pl.Datetime(time_zone="UTC"),
                "match_external_id": pl.Utf8,
                "bookmaker": pl.Utf8,
                "market": pl.Utf8,
                "outcome": pl.Utf8,
                "line": pl.Float64,
                "odds": pl.Float64,
            }
        )
    return pl.DataFrame(rows)


def injuries_to_polars(raw: list[dict[str, Any]]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    status_map = {
        "Missing Fixture": "out",
        "Questionable": "questionable",
        "Probable": "probable",
        "Doubtful": "doubtful",
    }
    for e in raw:
        player = e.get("player", {})
        fixture = e.get("fixture", {})
        reported_at = fixture.get("date") or datetime.now(tz=__import__("datetime").UTC).isoformat()
        rows.append(
            {
                "player_external_id": str(player.get("id", "")),
                "sport_code": "soccer",
                "status": status_map.get(player.get("type", ""), "out"),
                "body_part": player.get("reason"),
                "reported_at": reported_at,
                "source": "api_football",
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "player_external_id": pl.Utf8,
                "sport_code": pl.Utf8,
                "status": pl.Utf8,
                "body_part": pl.Utf8,
                "reported_at": pl.Datetime(time_zone="UTC"),
                "source": pl.Utf8,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("reported_at").str.to_datetime(time_zone="UTC").alias("reported_at")
    )


def lineups_to_polars(raw: list[dict[str, Any]], fixture_id: int) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for team_lineup in raw:
        team = team_lineup.get("team", {})
        starter_ids = [
            str(p.get("player", {}).get("id", "")) for p in team_lineup.get("startXI", [])
        ]
        rows.append(
            {
                "match_external_id": str(fixture_id),
                "team_external_id": str(team.get("id", "")),
                "starter_ids": starter_ids,
                "formation": team_lineup.get("formation"),
                "confirmed": True,
                "source": "api_football",
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "match_external_id": pl.Utf8,
                "team_external_id": pl.Utf8,
                "starter_ids": pl.List(pl.Utf8),
                "formation": pl.Utf8,
                "confirmed": pl.Boolean,
                "source": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


# ═══════════════════════ Flujos orquestados ═════════════════════════════


async def ingest_league_fixtures(league_slug: str, season: int) -> pl.DataFrame:
    """Ingesta fixtures + validación. Retorna DF listo para INSERT."""
    league_id = LEAGUE_IDS.get(league_slug)
    if league_id is None:
        msg = f"Liga desconocida: {league_slug}. Agregar a LEAGUE_IDS."
        raise ValueError(msg)

    client = APIFootballClient()
    async with client.session():
        raw = await client.fetch_fixtures(league=league_id, season=season)

    df = fixtures_to_polars(raw)
    if df.height == 0:
        logger.info("api_football.no_fixtures", league=league_slug, season=season)
        return df
    validated = validate_fixtures(df)
    logger.info(
        "api_football.fixtures_ok",
        league=league_slug,
        season=season,
        rows=validated.height,
    )
    return validated


async def ingest_league_odds(league_slug: str, season: int) -> pl.DataFrame:
    league_id = LEAGUE_IDS.get(league_slug)
    if league_id is None:
        msg = f"Liga desconocida: {league_slug}"
        raise ValueError(msg)

    client = APIFootballClient()
    async with client.session():
        raw = await client.fetch_odds(league=league_id, season=season)

    ts = datetime.now(tz=__import__("datetime").UTC)
    df = odds_to_polars(raw, ts)
    if df.height == 0:
        return df
    return validate_odds(df)


async def ingest_injuries(league_slug: str, season: int) -> pl.DataFrame:
    league_id = LEAGUE_IDS.get(league_slug)
    if league_id is None:
        msg = f"Liga desconocida: {league_slug}"
        raise ValueError(msg)

    client = APIFootballClient()
    async with client.session():
        raw = await client.fetch_injuries(league=league_id, season=season)

    df = injuries_to_polars(raw)
    if df.height == 0:
        return df
    return validate_injuries(df)


async def ingest_lineups(fixture_id: int) -> pl.DataFrame:
    client = APIFootballClient()
    async with client.session():
        raw = await client.fetch_lineups(fixture_id=fixture_id)

    df = lineups_to_polars(raw, fixture_id)
    if df.height == 0:
        return df
    return validate_lineups(df)
