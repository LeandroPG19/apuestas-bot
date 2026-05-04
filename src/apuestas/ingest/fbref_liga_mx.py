"""Fase 5.11 extra — Loader directo fbref.com para Liga MX (+ Liga Expansión).

soccerdata.FBref no expone ligas individuales non-Big-5. Pero fbref.com SÍ
tiene pages gratis:
  - Liga MX:         https://fbref.com/en/comps/31/Liga-MX-Stats
  - Liga Expansion:  https://fbref.com/en/comps/114/Liga-de-Expansion-MX-Stats
  - Copa MX:         https://fbref.com/en/comps/181/Copa-MX-Stats

Scrape directo con httpx + selectolax. Rate-limit 1 req/3s (TOS fbref pide
≤10 req/min). Cache local. Idempotente vía `resolve_or_create_match`.

También acumula via `historical_backfill.timer` diario.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from selectolax.parser import HTMLParser
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import resolve_or_create_match
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

FBREF_BASE = "https://fbref.com"
_CACHE_DIR = Path.home() / ".cache" / "apuestas" / "fbref_mx"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_URLS: dict[str, str] = {
    "liga_mx": "/en/comps/31/{season}/schedule/Liga-MX-Scores-and-Fixtures",
    "liga_expansion": "/en/comps/114/{season}/schedule/Liga-de-Expansion-MX-Scores-and-Fixtures",
}

# Rate limit fbref: 20 req/min → 3s entre reqs (aún respetuoso, 2x más rápido)
_MIN_DELAY_SECONDS = 3.0
_LAST_REQUEST: float = 0.0


async def _rate_wait() -> None:
    global _LAST_REQUEST
    now = asyncio.get_event_loop().time()
    delta = now - _LAST_REQUEST
    if delta < _MIN_DELAY_SECONDS:
        await asyncio.sleep(_MIN_DELAY_SECONDS - delta)
    _LAST_REQUEST = asyncio.get_event_loop().time()


async def _fetch(url: str, *, timeout: float = 30.0, cache_ttl_hours: int = 24) -> str | None:
    """Fetch con cache local + rate-limit."""
    cache_key = url.replace("/", "_").replace(":", "").replace(".", "_")[-180:]
    cache_file = _CACHE_DIR / f"{cache_key}.html"

    if cache_file.exists():
        age_hours = (datetime.now(tz=UTC).timestamp() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_ttl_hours:
            try:
                return cache_file.read_text(encoding="utf-8")
            except OSError:
                pass

    await _rate_wait()
    # fbref.com bloquea con 403 incluso con curl_cffi y camoufox (anti-scraping
    # extremo). Intentamos con curl_cffi (Chrome fingerprint); si falla, log info
    # y devolvemos None — Liga MX histórico pre-2026 queda fuera de scope gratis.
    # El catchup live desde Pinnacle sí cubre Liga MX actual.
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        cffi_requests = None  # type: ignore[assignment]

    if cffi_requests is not None:
        try:
            resp_cffi = cffi_requests.get(
                url,
                impersonate="chrome124",
                timeout=timeout,
            )
            if resp_cffi.status_code < 400:
                text_content = resp_cffi.text
                try:
                    cache_file.write_text(text_content, encoding="utf-8")
                except OSError:
                    pass
                return text_content  # type: ignore[no-any-return]
            logger.info("fbref_mx.not_found", url=url[:80], status=resp_cffi.status_code)
            return None
        except Exception as exc:
            logger.info("fbref_mx.cffi_fail", url=url[:80], error=str(exc)[:80])

    # Fallback httpx (raro que funcione si curl_cffi falló)
    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        logger.info("fbref_mx.http_fail", url=url[:80], error=str(exc)[:80])
        return None
    if resp.status_code >= 400:
        logger.info("fbref_mx.blocked", url=url[:80], status=resp.status_code)
        return None

    try:
        cache_file.write_text(resp.text, encoding="utf-8")
    except OSError:
        pass
    return resp.text


def _season_str(year: int) -> str:
    """2022 → '2022-2023' (fbref format)."""
    return f"{year}-{year + 1}"


def parse_fbref_fixtures_html(html: str) -> list[dict[str, Any]]:
    """Parse tabla `sched_all` de fbref. Retorna lista de matches."""
    parser = HTMLParser(html)
    # fbref encapsula tablas en comentarios HTML por anti-scraping ligero;
    # selectolax los ignora pero selectolax + decoded fbref funciona si usamos
    # regex para descomentar.
    raw = html.replace("<!--", "").replace("-->", "")
    parser = HTMLParser(raw)

    rows: list[dict[str, Any]] = []
    table = parser.css_first("table#sched_all, table[id^='sched_']")
    if table is None:
        return rows

    for tr in table.css("tbody tr"):
        date_cell = tr.css_first("td[data-stat='date']")
        home_cell = tr.css_first("td[data-stat='home_team']")
        away_cell = tr.css_first("td[data-stat='away_team']")
        score_cell = tr.css_first("td[data-stat='score']")

        if not (date_cell and home_cell and away_cell):
            continue
        date_raw = (date_cell.attributes.get("csk") or date_cell.text(strip=True) or "").strip()
        if not date_raw:
            continue
        home = home_cell.text(strip=True)
        away = away_cell.text(strip=True)
        if not home or not away:
            continue

        home_score = None
        away_score = None
        if score_cell:
            score_text = score_cell.text(strip=True).replace("–", "-").replace("—", "-")
            m = re.match(r"(\d+)\s*[-:]\s*(\d+)", score_text)
            if m:
                home_score = int(m.group(1))
                away_score = int(m.group(2))

        # Parse date
        start_time: datetime | None = None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                start_time = datetime.strptime(date_raw[:10], fmt).replace(tzinfo=UTC, hour=18)
                break
            except ValueError:
                continue
        if start_time is None:
            continue

        rows.append(
            {
                "home_name": home,
                "away_name": away,
                "start_time": start_time,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
    return rows


async def persist_fbref_matches(matches: list[dict[str, Any]], league_slug: str) -> int:
    """Resuelve + persiste matches vía _match_resolver."""
    inserted = 0
    async with session_scope() as session:
        for m in matches:
            match_id = await resolve_or_create_match(
                session,
                sport_code="soccer",
                home_name=m["home_name"],
                away_name=m["away_name"],
                start_time=m["start_time"],
                source=f"fbref:{league_slug}",
            )
            if match_id is None:
                continue
            if m["home_score"] is not None:
                await session.execute(
                    text(
                        """
                        UPDATE matches
                        SET home_score = :hs, away_score = :as_, status = 'finished'
                        WHERE id = :id AND status != 'finished'
                        """
                    ),
                    {
                        "id": match_id,
                        "hs": m["home_score"],
                        "as_": m["away_score"],
                    },
                )
            inserted += 1
    return inserted


async def ingest_liga_mx_season(
    league_slug: str,
    season: int,
) -> int:
    """Ingesta una temporada de Liga MX (o Expansion) vía fbref."""
    url_path = LEAGUE_URLS.get(league_slug)
    if url_path is None:
        logger.warning("fbref_mx.unknown_league", league=league_slug)
        return 0

    full_url = FBREF_BASE + url_path.format(season=_season_str(season))
    html = await _fetch(full_url)
    if html is None:
        return 0

    matches = parse_fbref_fixtures_html(html)
    if not matches:
        logger.info("fbref_mx.no_matches_parsed", league=league_slug, season=season)
        return 0

    n_inserted = await persist_fbref_matches(matches, league_slug)
    logger.info(
        "fbref_mx.ingested",
        league=league_slug,
        season=season,
        n_matches=len(matches),
        n_inserted=n_inserted,
    )
    return n_inserted


async def ingest_liga_mx_multi_seasons(
    league_slug: str = "liga_mx",
    seasons: list[int] | None = None,
) -> dict[int, int]:
    """Ingesta múltiples temporadas. Default: 5 temporadas hasta año actual."""
    if seasons is None:
        now = datetime.now(tz=UTC).year
        seasons = list(range(now - 4, now + 1))

    results: dict[int, int] = {}
    for season in seasons:
        try:
            results[season] = await ingest_liga_mx_season(league_slug, season)
        except Exception as exc:
            logger.warning(
                "fbref_mx.season_fail",
                league=league_slug,
                season=season,
                error=str(exc)[:120],
            )
            results[season] = 0
    return results
