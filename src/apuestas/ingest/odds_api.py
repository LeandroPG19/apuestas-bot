"""Cliente The Odds API (https://the-odds-api.com).

Free tier: 500 créditos/mes (1 crédito = 1 market × 1 region).
Paid $30: 20,000 créditos/mes.

Estrategia de muestreo para mantenerse en free tier:
- Pregame normal (>2h): cada 30 min
- T-2h a T-30min: cada 5 min
- T-30min a T-5min: cada 1 min
- Solo deportes activos en sesión
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from apuestas.config import get_settings
from apuestas.ingest.http_base import BaseAPIClient
from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_odds

logger = get_logger(__name__)

# Mapeo sport_code → The Odds API sport key
SPORT_KEY_MAP: dict[str, str] = {
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
    "soccer_epl": "soccer_epl",
    "soccer_laliga": "soccer_spain_la_liga",
    "soccer_bundesliga": "soccer_germany_bundesliga",
    "soccer_seriea": "soccer_italy_serie_a",
    "soccer_ligue1": "soccer_france_ligue_one",
    "soccer_ucl": "soccer_uefa_champs_league",
    "soccer_liga_mx": "soccer_mexico_ligamx",
    "boxing": "boxing_boxing",
    "mma": "mma_mixed_martial_arts",
}

MARKET_ALIASES: dict[str, str] = {
    "h2h": "h2h",
    "moneyline": "h2h",
    "spread": "spreads",
    "total": "totals",
    "runline": "spreads",
    "puckline": "spreads",
}


class OddsAPIClient(BaseAPIClient):
    base_url = "https://api.the-odds-api.com/v4"
    source_name = "the_odds_api"
    # 500 créditos/mes = ~16/día. Conservative rate limit:
    rate_limit = (30, 60.0)  # 30 req/min

    def __init__(self, *, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.apis.the_odds_api_key.get_secret_value()
            if settings.apis.the_odds_api_key
            else None
        )
        if not key:
            msg = "THE_ODDS_API_KEY requerida"
            raise ValueError(msg)
        super().__init__(api_key=key)
        self._key = key

    async def list_sports(self) -> list[dict[str, Any]]:
        return await self.get("/sports", params={"apiKey": self._key, "all": "true"})

    async def list_events(
        self,
        sport_key: str,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"apiKey": self._key, "dateFormat": "iso"}
        if date_from:
            params["commenceTimeFrom"] = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
        if date_to:
            params["commenceTimeTo"] = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")
        return await self.get(f"/sports/{sport_key}/events", params=params)

    async def fetch_odds(
        self,
        sport_key: str,
        *,
        regions: str = "us,us2,eu,uk",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "decimal",
        bookmakers: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch odds snapshot. 1 crédito por (region × market)."""
        params: dict[str, Any] = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return await self.get(f"/sports/{sport_key}/odds", params=params)

    async def fetch_historical_odds(
        self,
        sport_key: str,
        *,
        timestamp: datetime,
        regions: str = "us",
        markets: str = "h2h",
    ) -> dict[str, Any]:
        """Historical snapshot (solo paid tier)."""
        params = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "date": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return await self.get(f"/sports/{sport_key}/odds-history", params=params)

    async def check_remaining_credits(self) -> dict[str, int]:
        """Consulta cabeceras tras última request para créditos restantes."""
        # The Odds API pone el count en el header `x-requests-remaining`
        # Esta función requiere almacenar última respuesta; simplificada aquí
        return {}


def flatten_odds_to_polars(raw: list[dict[str, Any]], ts: datetime) -> pl.DataFrame:
    """Convierte respuesta raw de The Odds API a DataFrame polars para validación."""
    rows: list[dict[str, Any]] = []
    for event in raw:
        match_ext_id = event["id"]
        for bm in event.get("bookmakers", []):
            bookmaker = bm["key"]
            for market_info in bm.get("markets", []):
                market = _normalize_market(market_info["key"])
                for outcome in market_info.get("outcomes", []):
                    rows.append(
                        {
                            "ts": ts,
                            "match_external_id": match_ext_id,
                            "bookmaker": bookmaker,
                            "market": market,
                            "outcome": outcome["name"],
                            "line": float(outcome["point"]) if "point" in outcome else None,
                            "odds": float(outcome["price"]),
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


def _normalize_market(key: str) -> str:
    return MARKET_ALIASES.get(key, key)


async def ingest_sport(sport_key: str) -> int:
    """End-to-end: fetch → flatten → validate → return count rows válidas.

    Se completará el INSERT a Postgres en Fase 3-4.5 (cuando estén los modelos).
    """
    client = OddsAPIClient()
    async with client.session():
        raw = await client.fetch_odds(sport_key)
        ts = datetime.now(tz=__import__("datetime").UTC)
        df = flatten_odds_to_polars(raw, ts)
        if df.height == 0:
            logger.info("odds_api.no_data", sport_key=sport_key)
            return 0
        validated = validate_odds(df)
        logger.info("odds_api.ingest_ok", sport_key=sport_key, rows=validated.height)
        return validated.height
