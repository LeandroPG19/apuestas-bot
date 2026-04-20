"""Ingesta desde fuentes GRATIS — alternativa/complemento a API-Football.

Fuentes incluidas (todas $0/mes):

- **football-data.org** — 10 req/min free tier. Big-5 + Champions + Libertadores.
  No trae Liga MX en free; sirve para cobertura europea sin pagar $19/mes.
  Docs: https://www.football-data.org/documentation/quickstart

- **TheSportsDB** — key=`3` para acceso básico gratis sin registro. Cubre
  Liga MX, MLS, Liga Expansión MX, NBA, MLB, NFL, NHL, y muchas ligas
  secundarias. Fixtures + resultados históricos.
  Docs: https://www.thesportsdb.com/free_sports_api

- **ESPN hidden API** — sin key, semi-documentado. Scoreboards + odds para
  NBA/NFL/MLB/NHL/Soccer. URL pattern: `site.api.espn.com/apis/site/v2/sports/*`
  Ref: https://github.com/pseudo-r/Public-ESPN-API

- **Open-Meteo** — sin key, 10k req/día. Clima forecast global (ideal Liga MX).
  Docs: https://open-meteo.com/en/docs

- **NHL Stats API** — sin key (`api-web.nhle.com`). Schedule + boxscores + rosters.

- **MoneyPuck CSV** — scraping CSV público, xG + Corsi + Fenwick NHL.

Estas fuentes combinadas reemplazan >70% de la cobertura de API-Football sin costo.
Lo único que NO cubren gratis: odds en tiempo real de bookmakers MX/US
(para eso sigue siendo útil The Odds API free 500 créditos/mes).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from apuestas.config import get_settings
from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# ════════════════════════════ football-data.org ═══════════════════════════════


class FootballDataOrgClient(BaseAPIClient):
    """Cliente football-data.org free tier: 10 req/min, Big-5 + internacionales."""

    base_url = "https://api.football-data.org/v4"
    source_name = "football_data_org"
    rate_limit = (10, 60.0)  # 10 req/min

    COMPETITION_CODES: dict[str, str] = {
        "epl": "PL",
        "la_liga": "PD",
        "bundesliga": "BL1",
        "serie_a": "SA",
        "ligue_1": "FL1",
        "champions": "CL",
        "europa": "EL",
        "world_cup": "WC",
        "copa_america": "CA",
        "eurocopa": "EC",
        "brasileirao": "BSA",
    }

    def __init__(self, *, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.apis.football_data_org_key.get_secret_value()
            if getattr(settings.apis, "football_data_org_key", None)
            else None
        )
        super().__init__(api_key=key or "")

    def _default_headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self._api_key:
            h["X-Auth-Token"] = self._api_key
        return h

    async def fetch_matches(
        self,
        *,
        competition: str,
        season: int | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Partidos de una competición (código FIFA FBREF)."""
        code = self.COMPETITION_CODES.get(competition, competition)
        params: dict[str, Any] = {}
        if season is not None:
            params["season"] = season
        if status:
            params["status"] = status
        data = await self.get(f"/competitions/{code}/matches", params=params)
        return data.get("matches", [])

    async def fetch_teams(self, *, competition: str) -> list[dict[str, Any]]:
        code = self.COMPETITION_CODES.get(competition, competition)
        data = await self.get(f"/competitions/{code}/teams")
        return data.get("teams", [])


# ════════════════════════════ TheSportsDB ═════════════════════════════════════


class TheSportsDBClient(BaseAPIClient):
    """TheSportsDB — key=3 gratis sin registro. Liga MX + MLS + deportes US."""

    base_url = "https://www.thesportsdb.com/api/v1/json"
    source_name = "thesportsdb"
    rate_limit = (60, 60.0)

    def __init__(self, *, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or getattr(settings.apis, "thesportsdb_key", None) or "3"
        super().__init__(api_key=str(key))

    def _default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    def _key(self) -> str:
        return self._api_key or "3"

    async def search_league(self, *, name: str) -> list[dict[str, Any]]:
        data = await self.get(f"/{self._key()}/search_all_leagues.php", params={"s": name})
        return data.get("leagues") or []

    async def next_events_by_league(self, *, league_id: int) -> list[dict[str, Any]]:
        """Próximos 15 eventos por liga."""
        data = await self.get(f"/{self._key()}/eventsnextleague.php", params={"id": league_id})
        return data.get("events") or []

    async def last_events_by_league(self, *, league_id: int) -> list[dict[str, Any]]:
        data = await self.get(f"/{self._key()}/eventspastleague.php", params={"id": league_id})
        return data.get("events") or []

    async def team_roster(self, *, team_id: int) -> list[dict[str, Any]]:
        data = await self.get(f"/{self._key()}/lookup_all_players.php", params={"id": team_id})
        return data.get("player") or []


# Liga MX + MLS IDs en TheSportsDB (verificados):
THESPORTSDB_LEAGUE_IDS: dict[str, int] = {
    "liga_mx": 4350,
    "liga_expansion_mx": 4533,
    "mls": 4346,
    "epl": 4328,
    "la_liga": 4335,
    "nba": 4387,
    "mlb": 4424,
    "nfl": 4391,
    "nhl": 4380,
}


# ════════════════════════════ ESPN hidden API ═════════════════════════════════


class ESPNClient(BaseAPIClient):
    """ESPN hidden API — sin key, útil para odds básicos + scoreboards US."""

    base_url = "https://site.api.espn.com/apis/site/v2/sports"
    source_name = "espn"
    rate_limit = (60, 60.0)

    SPORT_PATHS: dict[str, str] = {
        "nba": "basketball/nba",
        "wnba": "basketball/wnba",
        "mlb": "baseball/mlb",
        "nfl": "football/nfl",
        "nhl": "hockey/nhl",
        "mls": "soccer/usa.1",
        "liga_mx": "soccer/mex.1",
        "epl": "soccer/eng.1",
        "la_liga": "soccer/esp.1",
        "champions": "soccer/uefa.champions",
    }

    def __init__(self) -> None:
        super().__init__(api_key="")

    def _default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json", "User-Agent": "apuestas-bot/0.1"}

    async def scoreboard(self, *, sport: str, date: str | None = None) -> dict[str, Any]:
        """Scoreboard de un deporte. date='YYYYMMDD' opcional."""
        path = self.SPORT_PATHS.get(sport, sport)
        params: dict[str, Any] = {}
        if date:
            params["dates"] = date
        return await self.get(f"/{path}/scoreboard", params=params)

    async def teams(self, *, sport: str) -> list[dict[str, Any]]:
        path = self.SPORT_PATHS.get(sport, sport)
        data = await self.get(f"/{path}/teams")
        teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        return [t.get("team", {}) for t in teams]


# ════════════════════════════ Open-Meteo ══════════════════════════════════════


class OpenMeteoClient(BaseAPIClient):
    """Open-Meteo gratis sin key. Forecast + histórico global."""

    base_url = "https://api.open-meteo.com/v1"
    source_name = "open_meteo"
    rate_limit = (10000, 86400.0)  # 10k req/día

    def __init__(self) -> None:
        super().__init__(api_key="")

    def _default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    async def forecast(
        self,
        *,
        lat: float,
        lon: float,
        hourly: list[str] | None = None,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "timezone": timezone,
        }
        if hourly:
            params["hourly"] = ",".join(hourly)
        else:
            params["hourly"] = "temperature_2m,precipitation,windspeed_10m,winddirection_10m"
        return await self.get("/forecast", params=params)


# ════════════════════════════ NHL Stats API ═══════════════════════════════════


class NHLStatsClient(BaseAPIClient):
    """NHL api-web.nhle.com — gratis sin key, oficial."""

    base_url = "https://api-web.nhle.com/v1"
    source_name = "nhl"
    rate_limit = (60, 60.0)

    def __init__(self) -> None:
        super().__init__(api_key="")

    def _default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    async def schedule(self, *, date: str | None = None) -> dict[str, Any]:
        """Schedule del día. date='YYYY-MM-DD', default hoy."""
        path = f"/schedule/{date}" if date else "/schedule/now"
        return await self.get(path)

    async def boxscore(self, *, game_id: str) -> dict[str, Any]:
        return await self.get(f"/gamecenter/{game_id}/boxscore")


# ════════════════════════════ Dispatcher de alto nivel ═════════════════════════


def thesportsdb_event_to_row(event: dict[str, Any]) -> dict[str, Any]:
    """Normaliza un event TheSportsDB a nuestro schema matches."""
    return {
        "external_id": str(event.get("idEvent")),
        "sport_code": event.get("strSport", "").lower(),
        "season": event.get("strSeason"),
        "home_team_name": event.get("strHomeTeam"),
        "away_team_name": event.get("strAwayTeam"),
        "start_time": f"{event.get('dateEvent', '')}T{event.get('strTime', '00:00:00')}Z",
        "venue_name": event.get("strVenue"),
        "home_score": event.get("intHomeScore"),
        "away_score": event.get("intAwayScore"),
    }


def events_to_polars(events: list[dict[str, Any]]) -> pl.DataFrame:
    if not events:
        return pl.DataFrame()
    rows = [thesportsdb_event_to_row(e) for e in events]
    return pl.DataFrame(rows)


async def ingest_liga_mx_upcoming() -> pl.DataFrame:
    """Ingresa próximos partidos Liga MX sin pagar API-Football."""
    async with TheSportsDBClient() as c:
        events = await c.next_events_by_league(league_id=THESPORTSDB_LEAGUE_IDS["liga_mx"])
    df = events_to_polars(events)
    logger.info("free_sources.liga_mx_ingested", count=df.height if not df.is_empty() else 0)
    return df


async def ingest_espn_scoreboard(sport: str, date: datetime | None = None) -> list[dict[str, Any]]:
    """Scoreboard NBA/NFL/MLB/NHL sin key."""
    async with ESPNClient() as c:
        date_str = date.strftime("%Y%m%d") if date else None
        data = await c.scoreboard(sport=sport, date=date_str)
    events = data.get("events") or []
    logger.info("free_sources.espn_scoreboard", sport=sport, count=len(events))
    return events
