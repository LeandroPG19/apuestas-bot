"""Fase 2.2 — Scraper % público vs money (contrarian signal).

Los sharps actúan contrarian cuando público está 80%+ en un lado pero línea
NO mueve → sharps en el otro lado presionando. Scrapea de fuentes públicas
gratis con fallback chain:

  1. BettingPros public-money (API JSON)
  2. Vegas Insider (HTML scrape)
  3. Action Network (requiere account — fallback final)

Feature derivada: `sharps_contrarian_signal` bump p_model +3pp cuando:
  |pct_money - pct_bets| > 15pp (sharps pesan más que múltiples públicos)
  AND line_movement_24h < 0.5pp (línea no movió a pesar de volumen público)

Persistencia en `public_betting_snapshots` (creada en migración 0011).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from selectolax.parser import HTMLParser
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class PublicBettingSnapshot:
    match_id: int
    market: str
    outcome: str
    line: float | None
    book: str | None
    pct_bets: float  # 0-1
    pct_money: float  # 0-1
    source: str
    captured_at: datetime


async def _fetch_html(url: str, *, timeout: float = 15.0) -> str | None:
    """HTTP GET con UA realista."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"),
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        logger.info("public_betting.http_fail", url=url[:60], error=str(exc)[:80])
        return None
    if resp.status_code >= 400:
        logger.info("public_betting.not_found", url=url[:60], status=resp.status_code)
        return None
    return resp.text


async def scrape_vegas_insider(sport: str = "nfl") -> list[dict[str, Any]]:
    """Scrape Vegas Insider consensus page para un sport.

    URL: https://www.vegasinsider.com/{sport}/matchups/

    Es un scrape básico; Vegas Insider cambia estructura DOM periódicamente
    (adaptar selectors si falla). Fail-soft: retorna [] si parse no funciona.
    """
    url = f"https://www.vegasinsider.com/{sport}/matchups/"
    html = await _fetch_html(url)
    if html is None:
        return []

    parser = HTMLParser(html)
    results: list[dict[str, Any]] = []
    # Vegas Insider usa tablas con clase "cgw"; cada row tiene team + consensus
    # Adaptar a su estructura real requiere inspección live.
    # Implementación placeholder: retorna vacío si no detecta tabla.
    tables = parser.css("table.main_sbox")
    if not tables:
        logger.info("public_betting.vegas_insider_no_data", sport=sport)
        return []
    # TODO: parseo real tras inspección live estructura — placeholder.
    return results


async def scrape_bettingpros(sport: str = "nfl") -> list[dict[str, Any]]:
    """Fuente BettingPros: API JSON pública.

    Endpoint: https://api.bettingpros.com/v3/picks/consensus?sport={sport}

    Requiere api-key = "CHi8Hy5CEE4khd46XNYL23dCFX96oUdw6qOt1Dnh" (público).
    """
    url = f"https://api.bettingpros.com/v3/picks/consensus?sport={sport.upper()}"
    headers = {
        "X-Api-Key": "CHi8Hy5CEE4khd46XNYL23dCFX96oUdw6qOt1Dnh",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        logger.info("public_betting.bettingpros_fail", error=str(exc)[:80])
        return []
    if resp.status_code >= 400:
        logger.info("public_betting.bettingpros_not_found", status=resp.status_code)
        return []

    data = resp.json()
    results: list[dict[str, Any]] = []
    for item in data.get("picks") or []:
        # Structure: {event: {home, away}, market, outcome, pct_bets, pct_money}
        event = item.get("event", {})
        results.append(
            {
                "home": event.get("home"),
                "away": event.get("away"),
                "market": item.get("market", "h2h"),
                "outcome": item.get("outcome"),
                "pct_bets": item.get("pct_bets"),
                "pct_money": item.get("pct_money"),
                "source": "bettingpros",
            }
        )
    logger.info("public_betting.bettingpros_fetched", sport=sport, rows=len(results))
    return results


async def persist_snapshots(
    snapshots: list[dict[str, Any]],
    *,
    match_id_resolver: Any = None,
) -> int:
    """Inserta filas en `public_betting_snapshots`.

    `match_id_resolver` opcional: función async (home, away) → match_id.
    Si None, intenta fuzzy match simple contra matches.
    """
    if not snapshots:
        return 0

    inserted = 0
    async with session_scope() as session:
        for snap in snapshots:
            home = snap.get("home")
            away = snap.get("away")
            if not home or not away:
                continue

            # Fuzzy match match_id por home/away en matches próximos 48h
            match_row = (
                await session.execute(
                    text(
                        """
                        SELECT m.id FROM matches m
                        JOIN teams ht ON ht.id = m.home_team_id
                        JOIN teams at ON at.id = m.away_team_id
                        WHERE similarity(ht.name, :h) > 0.6
                          AND similarity(at.name, :a) > 0.6
                          AND m.start_time BETWEEN now() AND now() + interval '48 hours'
                        ORDER BY similarity(ht.name, :h) + similarity(at.name, :a) DESC
                        LIMIT 1
                        """
                    ),
                    {"h": home, "a": away},
                )
            ).first()
            if match_row is None:
                continue

            await session.execute(
                text(
                    """
                    INSERT INTO public_betting_snapshots
                      (match_id, market, outcome, book,
                       pct_bets, pct_money, source)
                    VALUES
                      (:mid, :mk, :oc, :bk, :pb, :pm, :src)
                    """
                ),
                {
                    "mid": int(match_row.id),
                    "mk": snap.get("market", "h2h"),
                    "oc": snap.get("outcome", "home"),
                    "bk": snap.get("book"),
                    "pb": snap.get("pct_bets"),
                    "pm": snap.get("pct_money"),
                    "src": snap.get("source", "unknown"),
                },
            )
            inserted += 1

    logger.info("public_betting.persisted", rows=inserted)
    return inserted


async def get_contrarian_signal(
    match_id: int,
    market: str,
    outcome: str,
    *,
    money_vs_bets_threshold_pp: float = 0.15,
) -> dict[str, Any] | None:
    """Retorna señal contrarian si hay data pública y diverge de money.

    Returns:
        None si no hay datos suficientes.
        {signal_strength, direction_outcome, pct_bets, pct_money, delta}
        con `signal_strength ∈ {weak, strong}`.
    """
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT pct_bets, pct_money, captured_at
                    FROM public_betting_snapshots
                    WHERE match_id = :mid
                      AND market = :mk
                      AND outcome = :oc
                      AND captured_at > now() - interval '4 hours'
                    ORDER BY captured_at DESC
                    LIMIT 1
                    """
                ),
                {"mid": match_id, "mk": market, "oc": outcome},
            )
        ).first()

    if row is None or row.pct_bets is None or row.pct_money is None:
        return None

    pct_bets = float(row.pct_bets)
    pct_money = float(row.pct_money)
    delta = pct_money - pct_bets

    if abs(delta) < money_vs_bets_threshold_pp:
        return None

    # Signal: money > bets → sharps están respaldando este outcome
    # (pocas cuentas grandes mueven el money %).
    strength = "strong" if abs(delta) > 0.25 else "weak"
    return {
        "signal_strength": strength,
        "direction_outcome": outcome if delta > 0 else f"against_{outcome}",
        "pct_bets": pct_bets,
        "pct_money": pct_money,
        "delta_pp": delta,
    }


async def run_scraper_once(sports: tuple[str, ...] = ("nba", "nfl", "mlb")) -> dict[str, int]:
    """Ejecuta scraping completo para todos sports soportados."""
    counts: dict[str, int] = {}
    for sport in sports:
        snapshots = await scrape_bettingpros(sport)
        if not snapshots:
            snapshots = await scrape_vegas_insider(sport)
        counts[sport] = await persist_snapshots(snapshots)
    return counts
