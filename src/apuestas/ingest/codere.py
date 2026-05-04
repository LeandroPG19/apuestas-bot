"""Scraping Codere.mx — Gap #12 segundo book MX para line shopping.

Codere usa stack propio menos agresivo que Caliente (Bet365). Fetch HTTP plano
con httpx + parser selectolax. Si Codere empieza a aplicar anti-bot, migrar a
camoufox copiando el patrón de `caliente.py`.

Reutiliza `_match_resolver.resolve_or_create_match` (patrón compartido).
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

_SELECTORS_PATH = Path(__file__).resolve().parents[3] / "config" / "codere_selectors.yaml"

SPORT_SLUG_TO_CODE: dict[str, str] = {
    "liga_mx": "soccer",
    "nba": "nba",
    "mlb": "mlb",
    "nfl": "nfl",
}

_OUTCOME_NORMALIZER: dict[str, str] = {
    "local": "home",
    "casa": "home",
    "visitante": "away",
    "visita": "away",
    "empate": "draw",
    "x": "draw",
    "mas": "over",
    "over": "over",
    "menos": "under",
    "under": "under",
}


def _load_selectors() -> dict[str, Any]:
    with _SELECTORS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


async def fetch_html(url: str, *, timeout: float = 15.0) -> str | None:
    """HTTP plano con UA realista + retry simple. Retorna None si falla."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"),
        "Accept-Language": "es-MX,es;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code >= 400:
                logger.info("codere.http_error", url=url[:60], status=resp.status_code)
                return None
            return resp.text
    except httpx.HTTPError as exc:
        logger.info("codere.http_fail", error=str(exc)[:80])
        return None


def parse_events(html: str) -> list[dict[str, Any]]:
    cfg = _load_selectors()
    parser = HTMLParser(html)
    events: list[dict[str, Any]] = []

    for card in parser.css(cfg["selectors"]["event_card"]):
        home_el = card.css_first(cfg["selectors"]["team_home"])
        away_el = card.css_first(cfg["selectors"]["team_away"])
        time_el = card.css_first(cfg["selectors"]["start_time"])
        if not (home_el and away_el):
            continue

        ev: dict[str, Any] = {
            "home": home_el.text(strip=True),
            "away": away_el.text(strip=True),
            "start_time_raw": (time_el.attributes.get("datetime") if time_el else None)
            or (time_el.text(strip=True) if time_el else None),
            "markets": [],
        }

        for mkt in card.css(cfg["selectors"]["market_section"]):
            name_el = mkt.css_first(cfg["selectors"]["market_name"])
            market_name = name_el.text(strip=True) if name_el else "unknown"
            outcomes: list[dict[str, Any]] = []

            for btn in mkt.css(cfg["selectors"]["outcome_button"]):
                label_el = btn.css_first(cfg["selectors"]["outcome_label"])
                odds_el = btn.css_first(cfg["selectors"]["outcome_odds"])
                line_el = btn.css_first(cfg["selectors"]["outcome_line"])

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


def _parse_start_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _normalize_outcome(raw: str) -> str:
    key = normalize_name(raw)
    return _OUTCOME_NORMALIZER.get(key, raw.strip())


async def persist_codere_events(
    events: list[dict[str, Any]],
    sport_code: str,
    ts: datetime,
    market_aliases: dict[str, str],
) -> int:
    inserted = 0
    async with session_scope() as session:
        for ev in events:
            start_time = _parse_start_time(ev.get("start_time_raw"))
            match_id = await resolve_or_create_match(
                session,
                sport_code=sport_code,
                home_name=ev["home"],
                away_name=ev["away"],
                start_time=start_time,
                source="codere",
            )
            if match_id is None:
                continue

            for mkt in ev["markets"]:
                market_std = market_aliases.get(mkt["market"])
                if market_std is None:
                    continue
                for out in mkt["outcomes"]:
                    outcome_norm = _normalize_outcome(out["outcome"])
                    await session.execute(
                        text(
                            """
                            INSERT INTO odds_history
                              (ts, match_id, bookmaker, market, outcome, line, odds)
                            VALUES
                              (:ts, :mid, 'codere', :mk, :oc, :ln, :od)
                            """
                        ),
                        {
                            "ts": ts,
                            "mid": match_id,
                            "mk": market_std,
                            "oc": outcome_norm,
                            "ln": out["line"],
                            "od": out["odds"],
                        },
                    )
                    inserted += 1
    return inserted


async def ingest_codere_sport(sport_slug: str, *, persist: bool = True) -> int:
    """Fetch → parse → (persist). Retorna filas persistidas (o count de outcomes)."""
    cfg = _load_selectors()
    url = cfg["urls"].get(sport_slug)
    if not url:
        msg = f"URL no configurada para sport '{sport_slug}'"
        raise ValueError(msg)

    html = await fetch_html(url)
    if html is None:
        return 0

    events = parse_events(html)
    if not events:
        logger.info("codere.no_events", sport=sport_slug)
        return 0

    if persist:
        sport_code = SPORT_SLUG_TO_CODE.get(sport_slug, "soccer")
        rows = await persist_codere_events(
            events,
            sport_code,
            datetime.now(tz=UTC),
            cfg.get("market_aliases") or {},
        )
        logger.info("codere.persisted", sport=sport_slug, events=len(events), rows=rows)
        return rows

    return sum(len(m["outcomes"]) for ev in events for m in ev["markets"])
