"""Fase 5 — Scraper Sofascore.com.

Sofascore es la fuente gratuita más rica de stats deportivas (30+ deportes,
ratings 0-10 por jugador, xG/xGOT/PPDA, confirmed lineups 30-60 min antes).
No tiene API pública — scraping de endpoints JSON (Cloudflare-protected).

Uso personal únicamente (TOS prohíbe comercial). Gate via env:
    APUESTAS_ENABLE_SOFASCORE=true (default false)

Endpoints implementados:
  - /api/v1/sport/{sport}/scheduled-events/{YYYY-MM-DD}
  - /api/v1/event/{event_id}
  - /api/v1/event/{event_id}/lineups
  - /api/v1/event/{event_id}/statistics
  - /api/v1/event/{event_id}/best-players/summary
  - /api/v1/event/{event_id}/head-to-head
  - /api/v1/player/{player_id}/events/last/0

Rate-limit 2 req/seg con jitter. Cache local 1h.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.sofascore.com/api/v1"
_CACHE_DIR = Path.home() / ".cache" / "apuestas" / "sofascore"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limit tracking in-memory
_LAST_REQUEST_TS: float = 0.0
_MIN_DELAY_SECONDS = 0.5  # 2 req/seg


def _enabled() -> bool:
    return os.environ.get("APUESTAS_ENABLE_SOFASCORE", "").lower() in ("1", "true", "yes")


async def _rate_limit_wait() -> None:
    global _LAST_REQUEST_TS
    elapsed = asyncio.get_event_loop().time() - _LAST_REQUEST_TS
    if elapsed < _MIN_DELAY_SECONDS:
        jitter = random.uniform(0.0, 0.2)
        await asyncio.sleep(_MIN_DELAY_SECONDS - elapsed + jitter)
    _LAST_REQUEST_TS = asyncio.get_event_loop().time()


def _cache_path(endpoint: str) -> Path:
    safe = endpoint.replace("/", "_").replace("?", "_q_")
    return _CACHE_DIR / f"{safe[:200]}.json"


async def _fetch_json(
    endpoint: str,
    *,
    cache_ttl_seconds: int = 3600,
    timeout: float = 15.0,
) -> dict[str, Any] | None:
    """Fetch con cache. Retorna None si falla o Sofascore no habilitado."""
    if not _enabled():
        return None

    cache = _cache_path(endpoint)
    if cache.exists():
        age = datetime.now(tz=UTC).timestamp() - cache.stat().st_mtime
        if age < cache_ttl_seconds:
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    await _rate_limit_wait()
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sofascore.com/",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        logger.info("sofascore.http_fail", endpoint=endpoint[:60], error=str(exc)[:80])
        return None
    if resp.status_code >= 400:
        logger.info("sofascore.not_found", endpoint=endpoint[:60], status=resp.status_code)
        return None
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return None

    try:
        cache.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
    return data


async def fetch_scheduled_events(sport: str, date_str: str) -> list[dict[str, Any]]:
    """sport ∈ {football, basketball, tennis, baseball, american-football, ice-hockey, ...}.
    date_str formato YYYY-MM-DD."""
    data = await _fetch_json(f"/sport/{sport}/scheduled-events/{date_str}")
    if data is None:
        return []
    return list(data.get("events", []))


async def fetch_event_detail(event_id: int) -> dict[str, Any] | None:
    return await _fetch_json(f"/event/{event_id}")


async def fetch_event_lineups(event_id: int) -> dict[str, Any] | None:
    return await _fetch_json(f"/event/{event_id}/lineups")


async def fetch_event_statistics(event_id: int) -> dict[str, Any] | None:
    return await _fetch_json(f"/event/{event_id}/statistics")


async def fetch_event_best_players(event_id: int) -> dict[str, Any] | None:
    """Ratings Sofascore 0-10 + breakdown por jugador."""
    return await _fetch_json(f"/event/{event_id}/best-players/summary")


async def fetch_head_to_head(event_id: int) -> dict[str, Any] | None:
    return await _fetch_json(f"/event/{event_id}/head-to-head/events")


async def fetch_player_form(player_id: int, *, last_n: int = 10) -> list[dict[str, Any]]:
    """Últimos N partidos del jugador con rating + stats."""
    data = await _fetch_json(f"/player/{player_id}/events/last/0")
    if data is None:
        return []
    events = list(data.get("events", []))[:last_n]
    return events


async def fetch_event_incidents(event_id: int) -> list[dict[str, Any]]:
    """Live incidents: goals, red cards, subs, VAR. Crítico para live betting."""
    data = await _fetch_json(
        f"/event/{event_id}/incidents",
        cache_ttl_seconds=60,  # live → short cache
    )
    if data is None:
        return []
    return list(data.get("incidents", []))


def parse_player_rating(best_players: dict[str, Any]) -> dict[int, float]:
    """Extrae {player_id: rating_float} de /best-players/summary."""
    ratings: dict[int, float] = {}
    for team_key in ("homeTeam", "awayTeam"):
        team_data = best_players.get(team_key, {})
        for bucket_key in ("bestPlayer", "secondBestPlayer", "thirdBestPlayer"):
            p = team_data.get(bucket_key)
            if p and "player" in p and "rating" in p:
                pid = p["player"].get("id")
                if pid is not None:
                    ratings[int(pid)] = float(p["rating"])
    return ratings


def parse_match_xg(statistics: dict[str, Any]) -> dict[str, float | None]:
    """Extrae xG/xGOT/possession/shots de /statistics."""
    out: dict[str, float | None] = {
        "xg_home": None,
        "xg_away": None,
        "xgot_home": None,
        "xgot_away": None,
        "possession_home": None,
        "possession_away": None,
        "shots_home": None,
        "shots_away": None,
        "shots_on_target_home": None,
        "shots_on_target_away": None,
    }
    for period in statistics.get("statistics", []):
        if period.get("period") != "ALL":
            continue
        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = item.get("name", "").lower()
                try:
                    home_val = float(item.get("home", ""))
                    away_val = float(item.get("away", ""))
                except (ValueError, TypeError):
                    continue
                if name == "expected goals":
                    out["xg_home"] = home_val
                    out["xg_away"] = away_val
                elif "expected goals on target" in name:
                    out["xgot_home"] = home_val
                    out["xgot_away"] = away_val
                elif name == "ball possession":
                    out["possession_home"] = home_val
                    out["possession_away"] = away_val
                elif name == "total shots":
                    out["shots_home"] = int(home_val)
                    out["shots_away"] = int(away_val)
                elif name == "shots on target":
                    out["shots_on_target_home"] = int(home_val)
                    out["shots_on_target_away"] = int(away_val)
    return out


async def compute_team_rolling_stats(team_id: int, *, last_n: int = 10) -> dict[str, float]:
    """Promedio stats últimos N partidos de un equipo (para modelo Poisson props).

    Returns dict con: goals_for, goals_against, shots_for, shots_against,
    shots_on_target_for, corners, fouls, yellow_cards, possession_pct.
    """
    events = await fetch_player_form(team_id, last_n=last_n)
    if not events:
        return {}
    totals = {
        "goals_for": 0.0,
        "goals_against": 0.0,
        "shots_for": 0.0,
        "shots_against": 0.0,
        "shots_on_target_for": 0.0,
        "corners": 0.0,
        "fouls": 0.0,
        "yellow_cards": 0.0,
        "possession_pct": 0.0,
    }
    n_ok = 0
    for ev in events[:last_n]:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        stats = await fetch_event_statistics(ev_id)
        if not stats:
            continue
        parsed = parse_match_xg(stats)
        is_home = ev.get("homeTeam", {}).get("id") == team_id
        score_h = ev.get("homeScore", {}).get("current", 0) or 0
        score_a = ev.get("awayScore", {}).get("current", 0) or 0
        totals["goals_for"] += score_h if is_home else score_a
        totals["goals_against"] += score_a if is_home else score_h
        if is_home:
            totals["shots_for"] += float(parsed.get("shots_home") or 0)
            totals["shots_against"] += float(parsed.get("shots_away") or 0)
            totals["shots_on_target_for"] += float(parsed.get("shots_on_target_home") or 0)
            totals["possession_pct"] += float(parsed.get("possession_home") or 50)
        else:
            totals["shots_for"] += float(parsed.get("shots_away") or 0)
            totals["shots_against"] += float(parsed.get("shots_home") or 0)
            totals["shots_on_target_for"] += float(parsed.get("shots_on_target_away") or 0)
            totals["possession_pct"] += float(parsed.get("possession_away") or 50)
        n_ok += 1
    if n_ok == 0:
        return {}
    return {k: v / n_ok for k, v in totals.items()}
