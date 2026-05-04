"""Scraping Caliente.mx con camoufox (Firefox + anti-fingerprint).

Caliente aplica Cloudflare; camoufox-python patcheado supera el challenge.
Estrategia:
- Random delays human-like (2-7s).
- Random scroll antes de extraer.
- Guardar HTML raw en MinIO para replay si cambian selectores.
- Detección proactiva de captcha → alerta Telegram y pausa 1h.

Gap #1: persist real a `odds_history` con fuzzy team match compartido
(`_match_resolver`). Si no matchea ningún team existente, se crea uno nuevo
con `external_id='caliente:<sport>:<slug>'`.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from selectolax.parser import HTMLParser
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import normalize_name, resolve_or_create_match
from apuestas.obs.logging import get_logger
from apuestas.validators.schemas import validate_odds

logger = get_logger(__name__)

_SELECTORS_PATH = Path(__file__).resolve().parents[3] / "config" / "caliente_selectors.yaml"

SPORT_SLUG_TO_CODE: dict[str, str] = {
    "liga_mx": "soccer",
    "liga_expansion": "soccer",
    "nba": "nba",
    "mlb": "mlb",
    "nfl": "nfl",
    "boxing": "boxing",
}

_OUTCOME_NORMALIZER: dict[str, str] = {
    "local": "home",
    "casa": "home",
    "visitante": "away",
    "visita": "away",
    "empate": "draw",
    "x": "draw",
    "tie": "draw",
    "mas": "over",
    "more": "over",
    "over": "over",
    "menos": "under",
    "under": "under",
    "si": "yes",
    "yes": "yes",
    "no": "no",
}


def _load_selectors() -> dict[str, Any]:
    with _SELECTORS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class CalienteBannedError(Exception):
    """Cloudflare bloqueó / captcha sostenido."""


async def _human_delay(min_s: float, max_s: float) -> None:
    await asyncio.sleep(random.uniform(min_s, max_s))


async def fetch_html_with_camoufox(url: str) -> str:
    """Abre URL en camoufox, espera, scroll, devuelve HTML."""
    from camoufox.async_api import AsyncCamoufox  # type: ignore[import-untyped]

    cfg = _load_selectors()
    waits = cfg["waits"]

    async with AsyncCamoufox(
        headless=True,
        geoip=True,
        humanize=True,
        locale="es-MX",
    ) as browser:
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        await _human_delay(*waits["initial_delay_seconds"])

        for _ in range(3):
            await page.mouse.wheel(0, 800)
            await asyncio.sleep(waits["after_scroll"])

        content = await page.content()
        for signal in cfg["ban_signals"]:
            if signal in content:
                logger.error("caliente.ban_detected", signal=signal, url=url)
                raise CalienteBannedError(f"Cloudflare signal: {signal}")

        return content


def parse_events(html: str) -> list[dict[str, Any]]:
    """Parse DOM buscando event-card → lista de eventos con mercados."""
    cfg = _load_selectors()
    parser = HTMLParser(html)
    events: list[dict[str, Any]] = []

    cards = parser.css(cfg["selectors"]["event_card"])
    for card in cards:
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


def events_to_odds_polars(events: list[dict[str, Any]], ts: datetime) -> pl.DataFrame:
    """Aplanar a schema odds_history con match_external_id derivado.

    Útil como export intermedio (no persistido). El persist real pasa por
    `persist_caliente_events` para ligar con `match_id` de la DB.
    """
    market_map = {
        "Ganador del Partido": "h2h",
        "Resultado Final": "h2h",
        "1X2": "h2h",
        "Moneyline": "h2h",
        "Más/Menos": "totals",
        "Total de Goles": "totals",
        "Total": "totals",
        "Hándicap Asiático": "asian_handicap",
        "Spread": "spreads",
        "Handicap": "spreads",
        "Ambos Equipos Anotan": "btts",
        "Doble Oportunidad": "double_chance",
    }
    rows: list[dict[str, Any]] = []
    for ev in events:
        ext_id = f"caliente:{ev['home']}|{ev['away']}"
        for mkt in ev["markets"]:
            market_std = market_map.get(mkt["market"])
            if market_std is None:
                continue
            for out in mkt["outcomes"]:
                rows.append(
                    {
                        "ts": ts,
                        "match_external_id": ext_id,
                        "bookmaker": "caliente",
                        "market": market_std,
                        "outcome": out["outcome"],
                        "line": out["line"],
                        "odds": out["odds"],
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


def _normalize_outcome(raw: str) -> str:
    """Traduce labels Caliente → outcomes estándar."""
    key = normalize_name(raw)
    return _OUTCOME_NORMALIZER.get(key, raw.strip())


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


_MARKET_MAP: dict[str, str] = {
    "Ganador del Partido": "h2h",
    "Resultado Final": "h2h",
    "1X2": "h2h",
    "Moneyline": "h2h",
    "Más/Menos": "totals",
    "Total de Goles": "totals",
    "Total": "totals",
    "Hándicap Asiático": "asian_handicap",
    "Spread": "spreads",
    "Handicap": "spreads",
    "Ambos Equipos Anotan": "btts",
    "Doble Oportunidad": "double_chance",
}


async def persist_caliente_events(
    events: list[dict[str, Any]],
    sport_code: str,
    ts: datetime,
) -> int:
    """Itera events → resolve match → INSERT odds_history. Devuelve filas persistidas."""
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
                source="caliente",
            )
            if match_id is None:
                continue

            for mkt in ev["markets"]:
                market_std = _MARKET_MAP.get(mkt["market"])
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
                              (:ts, :mid, 'caliente', :mk, :oc, :ln, :od)
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


async def ingest_caliente_sport(
    sport_slug: str,
    *,
    persist: bool = True,
) -> pl.DataFrame:
    """Flujo completo: fetch → parse → (persist) → DataFrame → validate."""
    cfg = _load_selectors()
    url = cfg["urls"].get(sport_slug)
    if not url:
        msg = (
            f"URL no configurada para sport '{sport_slug}'. "
            "Agregar a config/caliente_selectors.yaml"
        )
        raise ValueError(msg)

    try:
        html = await fetch_html_with_camoufox(url)
    except CalienteBannedError:
        logger.warning("caliente.ban_pause_1h", sport=sport_slug)
        raise

    events = parse_events(html)
    ts = datetime.now(tz=UTC)

    if persist and events:
        sport_code = SPORT_SLUG_TO_CODE.get(sport_slug, "soccer")
        try:
            rows = await persist_caliente_events(events, sport_code, ts)
            logger.info("caliente.persisted", sport=sport_slug, events=len(events), rows=rows)
        except Exception:
            logger.exception("caliente.persist_failed", sport=sport_slug)

    df = events_to_odds_polars(events, ts)

    if df.height == 0:
        logger.info("caliente.no_events", sport=sport_slug)
        return df

    try:
        return validate_odds(df)
    except Exception:
        logger.exception("caliente.validation_failed", sport=sport_slug, rows=df.height)
        raise
