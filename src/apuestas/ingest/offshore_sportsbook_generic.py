"""Fase 5.10/5.11 — Scraper genérico parametrizable para sportsbooks offshore.

En vez de crear N scrapers casi idénticos (BetUS + BetWhale + Everygame +
SportsBetting.ag + Winpot + CampoBet + JugaBet + BC.GAME), un único scraper
toma configuración YAML por book.

Config esperada (`config/offshore_books/{slug}.yaml`):
```yaml
book_slug: betus
region: OFFSHORE  # OFFSHORE | MX
base_url: https://www.betus.com.pa/sportsbook
cloudflare_bypass: true  # usa camoufox
rate_limit_seconds: 1.5
urls:
  nba: "/basketball/nba"
  mlb: "/baseball/mlb"
  liga_mx: "/soccer/mexico/liga-mx"
selectors:
  event_card: "div.event-row"
  team_home: ".home-team"
  team_away: ".away-team"
  start_time: "time[datetime]"
  market_section: "section.market-block"
  market_name: ".market-title"
  outcome_button: "button.odds-btn"
  outcome_label: ".outcome-name"
  outcome_odds: ".odds-decimal"
market_aliases:
  "Moneyline": h2h
  "Point Spread": spreads
  "Total": totals
```

Uso:
```python
from apuestas.ingest.offshore_sportsbook_generic import OffshoreSportsbookScraper
scraper = OffshoreSportsbookScraper.from_yaml("betus")
rows = await scraper.scrape_sport("nba")
```

Integra con `_match_resolver.resolve_or_create_match` para persistir en
`odds_history`. Reutiliza patrón de `caliente.py::persist_caliente_events`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from selectolax.parser import HTMLParser
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import normalize_name, resolve_or_create_match
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CONFIG_DIRS = [
    Path(__file__).resolve().parents[3] / "config" / "offshore_books",
    Path(__file__).resolve().parents[3] / "config" / "mx_books",
]


_OUTCOME_NORMALIZER: dict[str, str] = {
    "home": "home",
    "away": "away",
    "local": "home",
    "visitante": "away",
    "visita": "away",
    "draw": "draw",
    "empate": "draw",
    "tie": "draw",
    "x": "draw",
    "over": "over",
    "mas": "over",
    "under": "under",
    "menos": "under",
    "yes": "yes",
    "no": "no",
    "si": "yes",
}


class OffshoreSportsbookScraper:
    """Scraper genérico configurable. Uso vía from_yaml()."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.book_slug: str = config["book_slug"]
        self.base_url: str = config["base_url"]
        self.cloudflare_bypass: bool = config.get("cloudflare_bypass", False)
        self.rate_limit: float = config.get("rate_limit_seconds", 1.5)
        self.urls: dict[str, str] = config.get("urls", {})
        self.selectors: dict[str, str] = config.get("selectors", {})
        self.market_aliases: dict[str, str] = config.get("market_aliases", {})
        self.sport_to_code: dict[str, str] = config.get("sport_to_code", {})

    @classmethod
    def from_yaml(cls, book_slug: str) -> OffshoreSportsbookScraper:
        """Carga configuración desde `config/{offshore_books,mx_books}/{slug}.yaml`."""
        for cfg_dir in _CONFIG_DIRS:
            cfg_path = cfg_dir / f"{book_slug}.yaml"
            if cfg_path.exists():
                return cls(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
        msg = f"Config YAML no encontrado para {book_slug!r} en {_CONFIG_DIRS}"
        raise FileNotFoundError(msg)

    async def _fetch_html(self, url: str) -> str | None:
        """Fetch HTML con camoufox o httpx según cloudflare_bypass."""
        if self.cloudflare_bypass:
            return await self._fetch_camoufox(url)
        return await self._fetch_httpx(url)

    async def _fetch_httpx(self, url: str, *, timeout: float = 20.0) -> str | None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
            ),
            "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
        }
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            logger.info(f"{self.book_slug}.http_fail", url=url[:60], error=str(exc)[:80])
            return None
        if resp.status_code >= 400:
            logger.info(f"{self.book_slug}.not_found", url=url[:60], status=resp.status_code)
            return None
        return resp.text

    async def _fetch_camoufox(self, url: str) -> str | None:
        """Fetch con camoufox (anti-Cloudflare). Lazy-import para evitar overhead."""
        try:
            from camoufox.async_api import AsyncCamoufox  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(f"{self.book_slug}.camoufox_missing")
            return None
        try:
            async with AsyncCamoufox(
                headless=True, geoip=True, humanize=True, locale="es-MX"
            ) as browser:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30_000)
                return await page.content()
        except Exception as exc:
            logger.warning(f"{self.book_slug}.camoufox_fail", url=url[:80], error=str(exc)[:100])
            return None

    def parse_events(self, html: str) -> list[dict[str, Any]]:
        """Parser DOM genérico usando selectors del YAML."""
        parser = HTMLParser(html)
        events: list[dict[str, Any]] = []
        for card in parser.css(self.selectors["event_card"]):
            home_el = card.css_first(self.selectors["team_home"])
            away_el = card.css_first(self.selectors["team_away"])
            time_el = card.css_first(self.selectors.get("start_time", ""))
            if not (home_el and away_el):
                continue

            start_time_raw = None
            if time_el:
                start_time_raw = time_el.attributes.get("datetime") or time_el.text(strip=True)

            ev: dict[str, Any] = {
                "home": home_el.text(strip=True),
                "away": away_el.text(strip=True),
                "start_time_raw": start_time_raw,
                "markets": [],
            }

            for mkt in card.css(self.selectors["market_section"]):
                name_el = mkt.css_first(self.selectors["market_name"])
                market_name = name_el.text(strip=True) if name_el else "unknown"
                outcomes: list[dict[str, Any]] = []
                for btn in mkt.css(self.selectors["outcome_button"]):
                    label_el = btn.css_first(self.selectors["outcome_label"])
                    odds_el = btn.css_first(self.selectors["outcome_odds"])
                    line_el = btn.css_first(self.selectors.get("outcome_line", ""))
                    try:
                        odds_val = (
                            float(odds_el.text(strip=True).replace(",", ".")) if odds_el else None
                        )
                    except ValueError:
                        odds_val = None
                    try:
                        line_val = (
                            float(line_el.text(strip=True).replace(",", ".")) if line_el else None
                        )
                    except ValueError:
                        line_val = None
                    if odds_val is None or odds_val <= 1.0:
                        continue
                    outcomes.append(
                        {
                            "outcome": label_el.text(strip=True) if label_el else "",
                            "odds": odds_val,
                            "line": line_val,
                        }
                    )
                if outcomes:
                    ev["markets"].append({"market": market_name, "outcomes": outcomes})
            if ev["markets"]:
                events.append(ev)
        return events

    def _normalize_outcome(self, raw: str) -> str:
        key = normalize_name(raw)
        return _OUTCOME_NORMALIZER.get(key, raw.strip())

    def _parse_start_time(self, raw: str | None) -> datetime | None:
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    async def persist_events(
        self, events: list[dict[str, Any]], sport_code: str, ts: datetime
    ) -> int:
        """Persiste a odds_history usando el resolver fuzzy compartido."""
        inserted = 0
        async with session_scope() as session:
            for ev in events:
                start_time = self._parse_start_time(ev.get("start_time_raw"))
                match_id = await resolve_or_create_match(
                    session,
                    sport_code=sport_code,
                    home_name=ev["home"],
                    away_name=ev["away"],
                    start_time=start_time,
                    source=self.book_slug,
                )
                if match_id is None:
                    continue
                for mkt in ev["markets"]:
                    market_std = self.market_aliases.get(mkt["market"])
                    if market_std is None:
                        continue
                    for out in mkt["outcomes"]:
                        outcome_norm = self._normalize_outcome(out["outcome"])
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
                                "bk": self.book_slug,
                                "mk": market_std,
                                "oc": outcome_norm,
                                "ln": out["line"],
                                "od": out["odds"],
                            },
                        )
                        inserted += 1
        return inserted

    async def scrape_sport(self, sport_slug: str) -> int:
        """Ingesta completa para un sport. Retorna filas persistidas."""
        url_path = self.urls.get(sport_slug)
        if url_path is None:
            logger.info(f"{self.book_slug}.sport_not_configured", sport=sport_slug)
            return 0
        full_url = self.base_url + url_path if url_path.startswith("/") else url_path
        html = await self._fetch_html(full_url)
        if html is None:
            return 0

        events = self.parse_events(html)
        if not events:
            logger.info(f"{self.book_slug}.no_events", sport=sport_slug)
            return 0

        sport_code = self.sport_to_code.get(sport_slug, "soccer")
        rows = await self.persist_events(events, sport_code, datetime.now(tz=UTC))
        logger.info(
            f"{self.book_slug}.persisted",
            sport=sport_slug,
            events=len(events),
            rows=rows,
        )
        return rows

    async def scrape_all_enabled_sports(self) -> dict[str, int]:
        """Scrapea todos los sports configurados en el YAML."""
        results: dict[str, int] = {}
        for sport_slug in self.urls:
            try:
                results[sport_slug] = await self.scrape_sport(sport_slug)
            except Exception as exc:
                logger.warning(
                    f"{self.book_slug}.sport_fail",
                    sport=sport_slug,
                    error=str(exc)[:100],
                )
                results[sport_slug] = 0
        return results
