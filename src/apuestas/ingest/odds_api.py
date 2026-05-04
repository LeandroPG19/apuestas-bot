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
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "soccer_epl": "soccer_epl",
    "soccer_laliga": "soccer_spain_la_liga",
    "soccer_bundesliga": "soccer_germany_bundesliga",
    "soccer_seriea": "soccer_italy_serie_a",
    "soccer_ligue1": "soccer_france_ligue_one",
    "soccer_ucl": "soccer_uefa_champs_league",
    "soccer_liga_mx": "soccer_mexico_ligamx",
    "soccer_mls": "soccer_usa_mls",
    "boxing": "boxing_boxing",
    "mma": "mma_mixed_martial_arts",
}

# Aliases internos (sport_code en matches) → key paid tier Odds API.
# Usados por `sync_odds_api_scores` para cubrir Liga MX + MLS con paid tier
# (reemplaza API-Football $19/mes — The Odds API $30/mes ya está pagado y
# sub-utilizado a ~5%).
INTERNAL_SPORT_TO_ODDS_KEY: dict[str, list[str]] = {
    "soccer": [
        # Top-5 Europa
        "soccer_epl",
        "soccer_spain_la_liga",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_france_ligue_one",
        # Segundas divisiones top europeas
        "soccer_efl_champ",  # Championship (Leicester-Millwall pick #117)
        "soccer_england_league1",
        "soccer_england_league2",
        "soccer_italy_serie_b",
        "soccer_spain_segunda_division",
        "soccer_germany_bundesliga2",
        "soccer_france_ligue_two",
        # Resto europeo modelable
        "soccer_netherlands_eredivisie",
        "soccer_portugal_primeira_liga",
        "soccer_belgium_first_div",
        "soccer_poland_ekstraklasa",  # pick #116 Zagłębie-Nieciecza
        "soccer_turkey_super_league",
        "soccer_sweden_allsvenskan",
        "soccer_norway_eliteserien",
        "soccer_denmark_superliga",
        "soccer_austria_bundesliga",
        "soccer_switzerland_superleague",
        "soccer_greece_super_league",
        # Copas europeas
        "soccer_uefa_champs_league",
        "soccer_uefa_europa_league",
        "soccer_uefa_europa_conference_league",
        "soccer_fa_cup",
        "soccer_england_efl_cup",
        # Americas + resto
        "soccer_mexico_ligamx",
        "soccer_usa_mls",
        "soccer_brazil_campeonato",
        "soccer_argentina_primera_division",
        "soccer_chile_campeonato",  # pick #47 Palestino-Deportes Concepción
        "soccer_conmebol_copa_libertadores",
        "soccer_conmebol_copa_sudamericana",
        "soccer_australia_aleague",
        "soccer_japan_j_league",
        "soccer_korea_kleague1",
        "soccer_china_superleague",
        "soccer_saudi_arabia_pro_league",
    ],
    "epl": ["soccer_epl"],
    "laliga": ["soccer_spain_la_liga"],
    "bundesliga": ["soccer_germany_bundesliga"],
    "seriea": ["soccer_italy_serie_a"],
    "ligue1": ["soccer_france_ligue_one"],
    "liga_mx": ["soccer_mexico_ligamx"],
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
    # Paid tier $30/mes: 20k créditos/mes. Rate limit generoso pero conservative
    # para no hit 429 (no hay publicado, ~60 req/min es safe).
    rate_limit = (50, 60.0)

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
        self._last_remaining: int | None = None
        self._last_used: int | None = None
        self._last_request_cost: int | None = None
        self._low_credit_alerted: bool = False

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
        regions: str = "us",  # solo 'us' por default: paid tier optimiza costo
        markets: str = "h2h,totals",
        odds_format: str = "decimal",
        bookmakers: str | None = "pinnacle,betonlineag,circasports",  # sharp books
        include_sids: bool = True,  # selection IDs estables para CLV matching
        include_bet_limits: bool = True,  # límites apostables → slippage
        include_links: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch odds snapshot (paid-optimized).

        Coste: regions × markets créditos por evento.
        Defaults paid: regions='us' (1× vs 'us,eu' que duplica), markets='h2h,totals'
        (2 créditos vs 3 con spreads — el detector usa h2h primario y totals para
        derivative markets; spreads queda opt-in para reducir baseline 33%).
        """
        params: dict[str, Any] = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        if include_sids:
            params["includeSids"] = "true"
        if include_bet_limits:
            params["includeBetLimits"] = "true"
        if include_links:
            params["includeLinks"] = "true"
        return await self.get(f"/sports/{sport_key}/odds", params=params)

    async def fetch_event_odds(
        self,
        sport_key: str,
        event_id: str,
        *,
        markets: str,
        regions: str = "us",
        bookmakers: str | None = "pinnacle,betonlineag,circasports,draftkings,fanduel",
        include_sids: bool = True,
    ) -> dict[str, Any]:
        """Event-level odds (player props, alternate lines, period markets).

        Coste: markets × regions por evento.
        Usado para player props que no se pueden pedir en bulk `/odds`.
        """
        params: dict[str, Any] = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        if include_sids:
            params["includeSids"] = "true"
        return await self.get(f"/sports/{sport_key}/events/{event_id}/odds", params=params)

    async def fetch_historical_odds(
        self,
        sport_key: str,
        *,
        timestamp: datetime,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = "pinnacle",
    ) -> dict[str, Any]:
        """Historical bulk odds snapshot (paid-only, costo 10× markets×regions).

        Retorna {timestamp, previous_timestamp, next_timestamp, data: [events]}.
        Útil para backtesting y CLV real (closing line Pinnacle).
        """
        params = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "date": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return await self.get(f"/historical/sports/{sport_key}/odds", params=params)

    async def fetch_historical_event_odds(
        self,
        sport_key: str,
        event_id: str,
        *,
        timestamp: datetime,
        regions: str = "us",
        markets: str = "h2h",
        bookmakers: str | None = "pinnacle",
    ) -> dict[str, Any]:
        """Historical event-level odds (player props, alternate lines).

        Player props/alternates solo disponibles desde 2023-05-03.
        Costo: markets × regions (sin 10× multiplier).
        """
        params = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "date": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return await self.get(
            f"/historical/sports/{sport_key}/events/{event_id}/odds", params=params
        )

    async def list_historical_events(
        self,
        sport_key: str,
        *,
        timestamp: datetime,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        """Historical events (paid-only). 1 crédito."""
        params: dict[str, Any] = {
            "apiKey": self._key,
            "dateFormat": "iso",
            "date": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if date_from:
            params["commenceTimeFrom"] = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
        if date_to:
            params["commenceTimeTo"] = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")
        return await self.get(f"/historical/sports/{sport_key}/events", params=params)

    def _on_response(self, response: Any) -> None:
        """Hook override: parsear cabeceras The Odds API para tracking de créditos
        + alerta Telegram auto cuando remaining < 2000 (10% del plan 20k)."""
        try:
            headers = response.headers
            rem = headers.get("x-requests-remaining")
            used = headers.get("x-requests-used")
            last_cost = headers.get("x-requests-last")
            if rem is not None:
                self._last_remaining = int(rem)
            if used is not None:
                self._last_used = int(used)
            if last_cost is not None:
                self._last_request_cost = int(last_cost)
            if self._last_remaining is not None:
                # Alerta temprana a <2000 (antes era <100, muy tarde)
                if self._last_remaining < 2000 and not self._low_credit_alerted:
                    logger.warning(
                        "odds_api.low_credits",
                        remaining=self._last_remaining,
                        used=self._last_used,
                    )
                    self._low_credit_alerted = True
                    self._try_telegram_alert()
                elif self._last_remaining >= 2000:
                    self._low_credit_alerted = False
        except (ValueError, TypeError, AttributeError):
            pass

    def _try_telegram_alert(self) -> None:
        """Best-effort Telegram alert cuando créditos <2000. Async fire-and-forget."""
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            loop.create_task(self._send_telegram_alert())
        except Exception:
            pass

    async def _send_telegram_alert(self) -> None:
        try:
            from apuestas.bot.telegram import send_admin_alert

            msg = (
                f"⚠️ Odds API paid tier créditos bajos\n"
                f"Remaining: {self._last_remaining}/20000\n"
                f"Used: {self._last_used}\n"
                f"Last cost: {getattr(self, '_last_request_cost', '?')}"
            )
            await send_admin_alert(msg)
        except Exception as exc:
            logger.debug("odds_api.telegram_alert_fail", error=str(exc)[:80])

    def remaining_credits(self) -> dict[str, int | None]:
        """Créditos tracking de la última respuesta recibida."""
        return {
            "remaining": self._last_remaining,
            "used": self._last_used,
            "last_request_cost": getattr(self, "_last_request_cost", None),
        }


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


def _odds_api_key_available() -> bool:
    """Fail-soft check para The Odds API. Sin key, el bot no rompe — skip."""
    settings = get_settings()
    key_obj = settings.apis.the_odds_api_key
    if key_obj is None:
        return False
    k = key_obj.get_secret_value().strip()
    # Detectar placeholders del .env.example
    return bool(k) and not k.startswith(("your-", "change-", "paste-"))


async def ingest_sport(sport_key: str) -> int:
    """End-to-end: fetch → flatten → validate → return count rows válidas.

    Fail-soft: si THE_ODDS_API_KEY ausente o placeholder, log info y 0 rows
    (no raise — el catchup_flow sigue con Pinnacle + Caliente).
    """
    if not _odds_api_key_available():
        logger.info("odds_api.disabled", reason="no_valid_key", sport=sport_key)
        return 0
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
