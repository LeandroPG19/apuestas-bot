"""OddsJam internal backend scraper — cobertura 85 books gratis sin auth.

Endpoint interno que la web oddsjam.com usa. Responde desde MX sin VPN.
Incluye books sharp raros (Pinnacle alt, Novig, ProphetX, Circa, BetOnline).

Endpoint: https://oddsjam.com/api/backend/oddscreen/v2/game/data
Params: sport, league, state, market_name

Formato: `data[].rows[]` donde cada row tiene odds por book. Odds americanas (+155, -200).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import resolve_or_create_match
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

BASE = "https://oddsjam.com/api/backend/oddscreen/v2/game/data"

# Map sport_code → (sport, league) para OddsJam
SPORT_MAP: dict[str, tuple[str, str]] = {
    "nba": ("basketball", "nba"),
    "nfl": ("football", "nfl"),
    "nhl": ("hockey", "nhl"),
    "mlb": ("baseball", "mlb"),
    "soccer_epl": ("soccer", "epl"),
    "soccer_laliga": ("soccer", "la-liga"),
}


def american_to_decimal(american: float) -> float | None:
    """Convierte +155/-200 a decimal."""
    if american is None:
        return None
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0 or (abs(a) < 100 and abs(a) > 10):
        return None
    if a > 0:
        return round(a / 100 + 1, 4)
    return round(100 / abs(a) + 1, 4)


async def fetch_oddsjam_sport(
    sport_code: str, *, market: str = "moneyline", timeout: float = 10.0
) -> dict[str, Any]:
    """Descarga JSON completo desde OddsJam. Retorna dict raw."""
    mapping = SPORT_MAP.get(sport_code)
    if mapping is None:
        return {}
    sport, league = mapping
    url = f"{BASE}?sport={sport}&league={league}&state=MX&market_name={market}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (apuestas-bot/1.0)"})
            if resp.status_code != 200:
                logger.info("oddsjam.non_200", sport=sport_code, status=resp.status_code)
                return {}
            return resp.json()
    except Exception as exc:
        logger.info("oddsjam.fetch_fail", sport=sport_code, error=str(exc)[:120])
        return {}


def parse_game(game: dict[str, Any]) -> list[tuple[str, str, str, float]] | None:
    """Extrae (home_name, away_name, book, odds_decimal) de un game OddsJam.

    Returns list con ambos outcomes × múltiples books o None si inválido.
    """
    rows = game.get("rows") or []
    if len(rows) < 2:
        return None

    home_row = next((r for r in rows if r.get("home_or_away") == "HOME"), None)
    away_row = next((r for r in rows if r.get("home_or_away") == "AWAY"), None)
    if not (home_row and away_row):
        return None

    home_name_raw = home_row.get("display", {}).get("Moneyline", {}).get(
        "team_name", ""
    ) or home_row.get("display", {}).get("Moneyline", {}).get("title", "")
    away_name_raw = away_row.get("display", {}).get("Moneyline", {}).get(
        "team_name", ""
    ) or away_row.get("display", {}).get("Moneyline", {}).get("title", "")
    if not (home_name_raw and away_name_raw):
        return None

    out: list[tuple[str, str, str, float]] = []
    for row, outcome in ((home_row, "home"), (away_row, "away")):
        odds_by_book = row.get("odds") or {}
        for book_name, odds_list in odds_by_book.items():
            if not odds_list:
                continue
            first = odds_list[0]
            price = first.get("price")
            if price is None:
                continue
            decimal = american_to_decimal(price)
            if decimal is None or decimal < 1.01 or decimal > 50:
                continue
            out.append((outcome, book_name, home_name_raw, decimal))
            out.append((outcome, book_name, away_name_raw, decimal))  # marker
    # Return simplified: flat list
    return out


async def ingest_oddsjam_sport(sport_code: str, *, persist: bool = True) -> int:
    """Fetch + persist odds OddsJam multi-book. Retorna rows insertadas."""
    data = await fetch_oddsjam_sport(sport_code)
    games = data.get("data") or []
    if not games:
        return 0

    inserted = 0
    ts = datetime.now(tz=UTC)

    async with session_scope() as session:
        for game in games:
            rows = game.get("rows") or []
            if len(rows) < 2:
                continue
            home_row = next((r for r in rows if r.get("home_or_away") == "HOME"), None)
            away_row = next((r for r in rows if r.get("home_or_away") == "AWAY"), None)
            if not (home_row and away_row):
                continue

            home_name = home_row.get("display", {}).get("Moneyline", {}).get(
                "team_name"
            ) or home_row.get("display", {}).get("Moneyline", {}).get("title")
            away_name = away_row.get("display", {}).get("Moneyline", {}).get(
                "team_name"
            ) or away_row.get("display", {}).get("Moneyline", {}).get("title")
            if not (home_name and away_name):
                continue

            # Fecha del partido: game_id contiene "YYYY-MM-DD"
            gid = game.get("game_id", "")
            date_part = gid.split("-")[-3:] if "-" in gid else []
            if len(date_part) == 3:
                try:
                    start = datetime.strptime("-".join(date_part), "%Y-%m-%d").replace(
                        tzinfo=UTC, hour=19
                    )
                except ValueError:
                    start = datetime.now(tz=UTC)
            else:
                start = datetime.now(tz=UTC)

            match_id = await resolve_or_create_match(
                session,
                sport_code=_canonical_sport(sport_code),
                home_name=home_name,
                away_name=away_name,
                start_time=start,
                source="oddsjam",
            )
            if match_id is None:
                continue

            # Collect odds por book
            home_odds_by_book: dict[str, float] = {}
            away_odds_by_book: dict[str, float] = {}

            for row, target in ((home_row, home_odds_by_book), (away_row, away_odds_by_book)):
                for book_name, odds_list in (row.get("odds") or {}).items():
                    if not odds_list:
                        continue
                    price = odds_list[0].get("price")
                    decimal = american_to_decimal(price)
                    if decimal is None or decimal < 1.01 or decimal > 50:
                        continue
                    target[book_name.lower()] = decimal

            # Persistir con sanity check por book: overround [0.97, 1.15]
            common_books = set(home_odds_by_book) & set(away_odds_by_book)
            if persist:
                for book in common_books:
                    ho = home_odds_by_book[book]
                    ao = away_odds_by_book[book]
                    overround = 1.0 / ho + 1.0 / ao
                    if overround < 0.97 or overround > 1.15:
                        continue
                    for outcome, odds_val in (("home", ho), ("away", ao)):
                        await session.execute(
                            text(
                                """
                                INSERT INTO odds_history
                                  (ts, match_id, bookmaker, market, outcome, line, odds)
                                VALUES (:ts, :mid, :bk, 'h2h', :oc, NULL, :od)
                                """
                            ),
                            {
                                "ts": ts,
                                "mid": match_id,
                                "bk": book.lower(),
                                "oc": outcome,
                                "od": odds_val,
                            },
                        )
                        inserted += 1

    logger.info("oddsjam.persisted", sport=sport_code, games=len(games), rows=inserted)
    return inserted


def _canonical_sport(sport_code: str) -> str:
    if sport_code.startswith("soccer_"):
        return "soccer"
    return sport_code
