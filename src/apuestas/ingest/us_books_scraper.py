"""US books scraper — DraftKings, FanDuel, BetMGM via endpoints JSON directos.

Los 3 sites protegen sus CDN con Akamai Bot Manager (DK/FD) y Cloudflare (MGM).
Por eso se usa `camoufox` (Firefox patched anti-fingerprint, ya usado para
Caliente.mx). La página monta las odds via XHR JSON al mismo host, así que
captura la response del XHR y parsea. No usa el DOM.

Para cada sportsbook:
- DraftKings: `sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{id}`
  eventgroup_id: NBA=42648, MLB=84240, NFL=88808, NHL=42133, EPL=40253
- FanDuel: `sbapi.{state}.sportsbook.fanduel.com/api/content-managed-page`
  Requiere state URL (NJ, PA, CO, etc.). Usa `sbapi.nj.` por default.
- BetMGM: `sports.{state}.betmgm.com/cds-api/bettingoffer/fixtures`
  Requiere state también.

Todos devuelven JSON con estructura propia — parse_* convierte a schema común.
Fallback: si camoufox no disponible, logs warning + retorna listas vacías.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import resolve_or_create_match
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_VPN_CHECK_CACHE: dict[str, Any] = {"checked_at": None, "ok": False}

DK_EVENTGROUP_IDS: dict[str, int] = {
    "nba": 42648,
    "mlb": 84240,
    "nfl": 88808,
    "nhl": 42133,
    "soccer_epl": 40253,
    "soccer_laliga": 40030,
    "soccer_ucl": 40821,
}


@dataclass(slots=True, frozen=True)
class USBookOdds:
    """Una línea de un US book."""

    bookmaker: str  # draftkings | fanduel | betmgm
    sport_code: str
    event_external_id: str
    home: str
    away: str
    start_time: datetime
    market: str  # h2h | totals | spreads
    outcome: str  # home | away | over | under
    odds_decimal: float
    line: float | None = None


async def _fetch_with_camoufox(url: str) -> str | None:
    """Abre URL en camoufox y captura el HTML (que contiene JSON para APIs)."""
    try:
        from camoufox.async_api import AsyncCamoufox  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("us_books.camoufox_missing", url=url[:60])
        return None

    try:
        async with AsyncCamoufox(
            headless=True,
            geoip=True,
            humanize=False,
            locale="en-US",
        ) as browser:
            page = await browser.new_page()
            # Los endpoints JSON devuelven Content-Type: application/json
            # que el browser renderiza como texto pre-formatted
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            body = await page.content()
            # El browser envuelve JSON en <html><body><pre>...</pre></body></html>
            if "<pre>" in body:
                start = body.index("<pre>") + len("<pre>")
                end = body.index("</pre>")
                return body[start:end]
            return body
    except Exception as exc:
        logger.warning("us_books.camoufox_fail", url=url[:80], error=str(exc)[:100])
        return None


def _american_to_decimal(price: int | float) -> float:
    price = int(price)
    if price > 0:
        return round(price / 100 + 1, 4)
    return round(100 / abs(price) + 1, 4)


def parse_draftkings(raw: dict[str, Any], sport_code: str) -> list[USBookOdds]:
    """Parse DraftKings eventgroup JSON → lista de USBookOdds.

    Estructura:
        eventGroup.offerCategories[].offerSubcategoryDescriptors[].offerSubcategory.offers[].outcomes[]
    """
    out: list[USBookOdds] = []
    eg = raw.get("eventGroup") or {}
    events_by_id: dict[int, dict[str, Any]] = {e["eventId"]: e for e in eg.get("events") or []}

    # Market type map — DK categories
    cat_map = {
        "Game Lines": {
            "Moneyline": "h2h",
            "Point Spread": "spreads",
            "Total": "totals",
            "Total Points": "totals",
            "Total Runs": "totals",
            "Total Goals": "totals",
        },
    }

    for cat in eg.get("offerCategories") or []:
        cat_name = cat.get("name", "")
        for sub in cat.get("offerSubcategoryDescriptors") or []:
            sub_name = sub.get("name", "")
            market = cat_map.get(cat_name, {}).get(sub_name)
            if market is None:
                continue
            for offer_group in (sub.get("offerSubcategory") or {}).get("offers") or []:
                for offer in offer_group:
                    event_id = offer.get("eventId")
                    event = events_by_id.get(event_id)
                    if not event:
                        continue
                    home = event.get("teamName1") or event.get("nameIdentifier", "")
                    away = event.get("teamName2") or ""
                    try:
                        start = datetime.fromisoformat(
                            event.get("startDate", "").replace("Z", "+00:00")
                        )
                    except (TypeError, ValueError):  # fmt: skip
                        continue
                    for outcome in offer.get("outcomes") or []:
                        label = (outcome.get("label") or "").lower()
                        price = outcome.get("oddsDecimal") or outcome.get("oddsAmerican")
                        if price is None:
                            continue
                        # Normalizar outcome name
                        if market == "h2h":
                            oc = "home" if label == home.lower() else "away"
                        elif market == "totals":
                            oc = "over" if "over" in label else "under"
                        else:  # spreads
                            oc = "home" if home.lower() in label else "away"
                        line_raw = outcome.get("line")
                        try:
                            decimal = float(price)
                            if decimal > 100 or decimal < 1:  # asumir americano
                                decimal = _american_to_decimal(decimal)
                        except (TypeError, ValueError):  # fmt: skip
                            continue
                        out.append(
                            USBookOdds(
                                bookmaker="draftkings",
                                sport_code=sport_code,
                                event_external_id=str(event_id),
                                home=home,
                                away=away,
                                start_time=start,
                                market=market,
                                outcome=oc,
                                odds_decimal=decimal,
                                line=float(line_raw) if line_raw is not None else None,
                            )
                        )
    return out


async def fetch_draftkings(sport_code: str) -> list[USBookOdds]:
    """Descarga + parsea DK para un deporte. Requiere camoufox."""
    eg_id = DK_EVENTGROUP_IDS.get(sport_code)
    if eg_id is None:
        logger.info("us_books.dk_unsupported_sport", sport=sport_code)
        return []
    url = f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{eg_id}?format=json"
    body = await _fetch_with_camoufox(url)
    if body is None:
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        logger.warning("us_books.dk_json_invalid", sport=sport_code, error=str(exc)[:80])
        return []
    return parse_draftkings(data, sport_code)


# BetMGM sport IDs (CDS API, descubiertos en network tab; globales por deporte)
BETMGM_SPORT_IDS: dict[str, int] = {
    "nba": 7,
    "mlb": 4,
    "nfl": 11,
    "nhl": 17,
    "soccer_epl": 29,
    "soccer_laliga": 29,
    "soccer_ucl": 29,
    "tennis": 5,
    "boxing": 34,
    "mma": 34,
}


def parse_betmgm(data: dict[str, Any], sport_code: str) -> list[USBookOdds]:
    """Parser BetMGM CDS fixtures JSON.

    Estructura: `fixtures[].optionMarkets[].options[]` con `price.decimal`
    y `type` mapeado a moneyline/spread/total.
    """
    out: list[USBookOdds] = []
    fixtures = data.get("fixtures") or []
    market_map = {
        "MoneyLine": "h2h",
        "Handicap": "spreads",
        "PointSpread": "spreads",
        "Total": "totals",
        "Totals": "totals",
    }
    for fx in fixtures:
        participants = fx.get("participants") or []
        if len(participants) < 2:
            continue
        home = (
            participants[0].get("name", {}).get("value", "")
            if isinstance(participants[0], dict)
            else ""
        )
        away = (
            participants[1].get("name", {}).get("value", "")
            if isinstance(participants[1], dict)
            else ""
        )
        try:
            start = datetime.fromisoformat(fx.get("startDate", "").replace("Z", "+00:00"))
        except (TypeError, ValueError):  # fmt: skip
            continue
        event_id = str(fx.get("id") or fx.get("fixtureId") or "")

        for market in fx.get("optionMarkets") or []:
            m_type_raw = (
                market.get("name", {}).get("value")
                if isinstance(market.get("name"), dict)
                else market.get("name", "")
            )
            m_type = market_map.get(m_type_raw or "")
            if m_type is None:
                continue
            for i, option in enumerate(market.get("options") or []):
                price_obj = option.get("price") or {}
                decimal = price_obj.get("decimal")
                if decimal is None:
                    continue
                label = (
                    option.get("name", {}).get("value")
                    if isinstance(option.get("name"), dict)
                    else str(option.get("name", ""))
                )
                label_lower = (label or "").lower()
                if m_type == "totals":
                    outcome = "over" if "over" in label_lower or i == 0 else "under"
                elif m_type == "h2h":
                    outcome = "home" if i == 0 else "away"
                else:
                    outcome = "home" if i == 0 else "away"
                line_raw = (
                    market.get("attr", {}).get("points")
                    if isinstance(market.get("attr"), dict)
                    else None
                )
                try:
                    odd_dec = float(decimal)
                except (TypeError, ValueError):  # fmt: skip
                    continue
                out.append(
                    USBookOdds(
                        bookmaker="betmgm",
                        sport_code=sport_code,
                        event_external_id=event_id,
                        home=home,
                        away=away,
                        start_time=start,
                        market=m_type,
                        outcome=outcome,
                        odds_decimal=odd_dec,
                        line=float(line_raw) if line_raw is not None else None,
                    )
                )
    return out


async def fetch_betmgm(sport_code: str, state: str | None = None) -> list[USBookOdds]:
    """BetMGM via CDS API. state default viene de APUESTAS_US_BOOKS_STATE."""
    state = (state or os.environ.get("APUESTAS_US_BOOKS_STATE") or "nj").lower()
    sport_id = BETMGM_SPORT_IDS.get(sport_code)
    if sport_id is None:
        return []
    subdivision = f"US-{state.upper()}"
    url = (
        f"https://sports.{state}.betmgm.com/cds-api/bettingoffer/fixtures?"
        f"x-bwin-accessid=&lang=en-us&country=US&userCountry=US&subdivision={subdivision}&"
        f"fixtureTypes=Standard&state=Latest&offerCategories=Gridset&offerMapping=Filtered&"
        f"fixtureCategories=Gridable,NonGridable,Other&sportIds={sport_id}&competitionIds=&"
        f"skip=0&take=80&sortBy=Tags"
    )
    body = await _fetch_with_camoufox(url)
    if body is None:
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    parsed = parse_betmgm(data, sport_code)
    logger.info("us_books.betmgm.fetched", sport=sport_code, state=state, rows=len(parsed))
    return parsed


# FanDuel event IDs varían por state. El endpoint content-managed-page es
# stable pero requiere customPageId por liga. Estos son los más comunes:
FANDUEL_CUSTOM_PAGE_IDS: dict[str, str] = {
    "nba": "NBA",
    "mlb": "MLB",
    "nfl": "NFL",
    "nhl": "NHL",
    "soccer_epl": "ENGLISH_PREMIER_LEAGUE",
    "soccer_laliga": "LA_LIGA",
    "soccer_ucl": "UEFA_CHAMPIONS_LEAGUE",
}


def parse_fanduel(data: dict[str, Any], sport_code: str) -> list[USBookOdds]:
    """Parser FanDuel content-managed-page JSON.

    Estructura: `attachments.events[id]` con `markets[].runners[].winRunnerOdds.decimalBetterOdds`
    o `trueOdds.decimalOdds.decimalOdds`.
    """
    out: list[USBookOdds] = []
    attachments = data.get("attachments") or {}
    events = attachments.get("events") or {}
    markets_dict = attachments.get("markets") or {}
    runners_dict = attachments.get("runners") or {}

    for event_id, event in events.items():
        participants = event.get("participants") or []
        home = participants[0].get("participantName") if len(participants) > 0 else ""
        away = participants[1].get("participantName") if len(participants) > 1 else ""
        try:
            start = datetime.fromisoformat(event.get("openDate", "").replace("Z", "+00:00"))
        except (TypeError, ValueError):  # fmt: skip
            continue

        for market_id in event.get("markets") or []:
            market = markets_dict.get(str(market_id))
            if not market:
                continue
            m_name = (market.get("marketType") or "").lower()
            if "money" in m_name or m_name == "match_odds":
                market_type = "h2h"
            elif "total" in m_name:
                market_type = "totals"
            elif "handicap" in m_name or "spread" in m_name:
                market_type = "spreads"
            else:
                continue
            line = market.get("handicap")
            for i, runner_id in enumerate(market.get("runners") or []):
                runner = runners_dict.get(str(runner_id))
                if not runner:
                    continue
                odds_obj = (runner.get("winRunnerOdds") or {}).get("decimalBetterOdds")
                if odds_obj is None:
                    odds_obj = (
                        (runner.get("trueOdds") or {}).get("decimalOdds", {}).get("decimalOdds")
                    )
                try:
                    odd_dec = float(odds_obj) if odds_obj is not None else None
                except (TypeError, ValueError):  # fmt: skip
                    continue
                if odd_dec is None:
                    continue
                name = (runner.get("runnerName") or "").lower()
                if market_type == "totals":
                    outcome = "over" if "over" in name else "under"
                else:
                    outcome = "home" if i == 0 else "away"
                out.append(
                    USBookOdds(
                        bookmaker="fanduel",
                        sport_code=sport_code,
                        event_external_id=str(event_id),
                        home=home or "",
                        away=away or "",
                        start_time=start,
                        market=market_type,
                        outcome=outcome,
                        odds_decimal=odd_dec,
                        line=float(line) if line is not None else None,
                    )
                )
    return out


async def fetch_fanduel(sport_code: str, state: str | None = None) -> list[USBookOdds]:
    """FanDuel via content-managed-page JSON."""
    state = (state or os.environ.get("APUESTAS_US_BOOKS_STATE") or "nj").lower()
    page_id = FANDUEL_CUSTOM_PAGE_IDS.get(sport_code)
    if page_id is None:
        return []
    url = (
        f"https://sbapi.{state}.sportsbook.fanduel.com/api/content-managed-page?"
        f"page=CUSTOM&customPageId={page_id}&timezone=America%2FNew_York"
    )
    body = await _fetch_with_camoufox(url)
    if body is None:
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    parsed = parse_fanduel(data, sport_code)
    logger.info("us_books.fanduel.fetched", sport=sport_code, state=state, rows=len(parsed))
    return parsed


def _book_enabled(book: str, *, default: bool = False) -> bool:
    """Revisa flag APUESTAS_ENABLE_{DK,FANDUEL,BETMGM} en env."""
    flag = f"APUESTAS_ENABLE_{book.upper()}"
    raw = os.environ.get(flag, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


async def _vpn_active(ttl_seconds: int = 300) -> bool:
    """Gap #13: verifica que el tráfico sale de una IP estadounidense.

    1. Flag obligatorio `APUESTAS_US_VPN_ACTIVE=true` (explícito; default false).
    2. Healthcheck a ipinfo.io con TTL para no spammear en cada call.
    """
    flag = os.environ.get("APUESTAS_US_VPN_ACTIVE", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False

    now = datetime.now(tz=UTC)
    cached_at = _VPN_CHECK_CACHE["checked_at"]
    if (
        cached_at is not None
        and isinstance(cached_at, datetime)
        and (now - cached_at).total_seconds() < ttl_seconds
    ):
        return bool(_VPN_CHECK_CACHE["ok"])

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://ipinfo.io/json")
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("us_books.vpn_check_failed", error=str(exc)[:80])
        _VPN_CHECK_CACHE["checked_at"] = now
        _VPN_CHECK_CACHE["ok"] = False
        return False

    country = str(data.get("country", "")).upper()
    is_us = country == "US"
    _VPN_CHECK_CACHE["checked_at"] = now
    _VPN_CHECK_CACHE["ok"] = is_us
    if not is_us:
        logger.warning("us_books.not_us_ip", country=country)
    return is_us


_SPORT_CODE_NORMALIZE: dict[str, str] = {
    "soccer_epl": "soccer",
    "soccer_laliga": "soccer",
    "soccer_bundesliga": "soccer",
    "soccer_seriea": "soccer",
    "soccer_ligue1": "soccer",
    "soccer_ucl": "soccer",
    "soccer_liga_mx": "soccer",
    "soccer_mls": "soccer",
    "nba": "nba",
    "mlb": "mlb",
    "nfl": "nfl",
    "nhl": "nhl",
}


async def persist_us_book_odds(odds: list[USBookOdds]) -> int:
    """Resuelve matches vía fuzzy y persiste en `odds_history`."""
    if not odds:
        return 0
    ts = datetime.now(tz=UTC)
    inserted = 0
    async with session_scope() as session:
        for o in odds:
            sport_code = _SPORT_CODE_NORMALIZE.get(o.sport_code, o.sport_code)
            match_id = await resolve_or_create_match(
                session,
                sport_code=sport_code,
                home_name=o.home,
                away_name=o.away,
                start_time=o.start_time,
                source=o.bookmaker,
            )
            if match_id is None:
                continue
            await session.execute(
                text(
                    """
                    INSERT INTO odds_history
                      (ts, match_id, bookmaker, market, outcome, line, odds)
                    VALUES
                      (:ts, :mid, :bk, :mk, :oc, :ln, :od)
                    """
                ),
                {
                    "ts": ts,
                    "mid": match_id,
                    "bk": o.bookmaker,
                    "mk": o.market,
                    "oc": o.outcome,
                    "ln": o.line,
                    "od": o.odds_decimal,
                },
            )
            inserted += 1
    return inserted


async def fetch_all(sport_codes: list[str]) -> dict[str, list[USBookOdds]]:
    """Fetch en paralelo DK + FD + MGM solo para books habilitados + VPN US activa.

    Defaults:
    - `APUESTAS_ENABLE_DK=true` por default (DK es el menos agresivo).
    - `APUESTAS_ENABLE_FANDUEL`/`BETMGM` default false (TOS más estricto).
    - `APUESTAS_US_VPN_ACTIVE=true` obligatorio (sino skip completo).

    El persist en odds_history se hace al final vía `persist_us_book_odds`.
    """
    results: dict[str, list[USBookOdds]] = {"draftkings": [], "fanduel": [], "betmgm": []}

    if not await _vpn_active():
        logger.info(
            "us_books.skip_no_vpn",
            hint="set APUESTAS_US_VPN_ACTIVE=true and connect to a US VPN",
        )
        return results

    books_to_run: list[tuple[str, Any]] = []
    if _book_enabled("dk", default=True):
        books_to_run.append(("draftkings", fetch_draftkings))
    if _book_enabled("fanduel"):
        books_to_run.append(("fanduel", fetch_fanduel))
    if _book_enabled("betmgm"):
        books_to_run.append(("betmgm", fetch_betmgm))
    if not books_to_run:
        logger.info("us_books.all_disabled", hint="set APUESTAS_ENABLE_DK=true to activate")
        return results

    tasks: list[tuple[str, str, Any]] = []
    for sport in sport_codes:
        for book, fn in books_to_run:
            tasks.append((book, sport, fn(sport)))

    gathered = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)
    for (book, _sport, _task), res in zip(tasks, gathered, strict=True):
        if isinstance(res, list):
            results[book].extend(res)
        else:
            logger.warning("us_books.task_failed", book=book, error=str(res)[:100])

    total_odds = [o for book_odds in results.values() for o in book_odds]
    if total_odds:
        try:
            rows = await persist_us_book_odds(total_odds)
            logger.info("us_books.persisted", rows=rows, total=len(total_odds))
        except Exception:
            logger.exception("us_books.persist_failed")

    return results
