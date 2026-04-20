"""Flow live_scores — ingesta resultados post-match (§19.7).

Marca matches con `status='finished'` y persiste home_score/away_score.
Se ejecuta tras partidos terminados (T + 2h) dentro del flow `deep_analysis`
o via timer systemd user. Dispara el flow settle_bets al completar.

Fuentes: API-Football fixtures (status=FT/AET/PEN) para fútbol;
nba_api boxscore para NBA; NHL Stats API para NHL; The Odds API
scores endpoint para NFL/MLB; Jeff Sackmann/API-Tennis para tenis.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.api_football import LEAGUE_IDS, APIFootballClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

FINAL_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}


@task(retries=2, retry_delay_seconds=20)
async def pending_finished_matches(window_hours: int = 48) -> list[dict[str, Any]]:
    """Matches que empezaron hace >=2h y aún status != 'finished'."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, sport_code, league_id, external_id, start_time
                FROM matches
                WHERE status <> 'finished'
                  AND start_time <= NOW() - INTERVAL '2 hours'
                  AND start_time >= NOW() - INTERVAL ':w hours'
                ORDER BY start_time ASC
                LIMIT 200
                """.replace(":w", str(window_hours))
            )
        )
        return [dict(r._mapping) for r in result.all()]


async def _finalize_match(
    *, match_id: int, home_score: int, away_score: int, final_status: str
) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                UPDATE matches
                SET home_score = :hs,
                    away_score = :as_,
                    status = 'finished'
                WHERE id = :mid
                """
            ),
            {"mid": match_id, "hs": home_score, "as_": away_score},
        )
        _ = final_status  # reservado para futura columna final_status_raw


@task(retries=2, retry_delay_seconds=30)
async def sync_soccer_scores(matches: list[dict[str, Any]]) -> int:
    updated = 0
    if not matches:
        return 0
    league_to_matches: dict[int, list[dict[str, Any]]] = {}
    for m in matches:
        if m["sport_code"] != "soccer" or m.get("league_id") is None:
            continue
        league_to_matches.setdefault(int(m["league_id"]), []).append(m)
    if not league_to_matches:
        return 0

    reverse_leagues = {v: k for k, v in LEAGUE_IDS.items()}
    client = APIFootballClient()
    async with client.session():
        for league_id, group in league_to_matches.items():
            slug = reverse_leagues.get(league_id, f"league_{league_id}")
            season = datetime.now(tz=UTC).year
            try:
                fixtures = await client.fetch_fixtures(
                    league=league_id,
                    season=season,
                    date_from=datetime.now(tz=UTC) - timedelta(days=3),
                    date_to=datetime.now(tz=UTC),
                )
            except Exception as exc:
                logger.warning("live_scores.soccer_fetch_fail", league=slug, error=str(exc))
                continue

            by_external = {str(f["fixture"]["id"]): f for f in fixtures if "fixture" in f}
            for match in group:
                ext = str(match.get("external_id") or "")
                fx = by_external.get(ext)
                if not fx:
                    continue
                status = fx.get("fixture", {}).get("status", {}).get("short", "")
                if status not in FINAL_STATUSES:
                    continue
                goals = fx.get("goals") or {}
                hs = goals.get("home")
                as_ = goals.get("away")
                if hs is None or as_ is None:
                    continue
                await _finalize_match(
                    match_id=int(match["id"]),
                    home_score=int(hs),
                    away_score=int(as_),
                    final_status=status,
                )
                updated += 1
    return updated


@task(retries=2, retry_delay_seconds=30)
async def sync_nba_scores(matches: list[dict[str, Any]]) -> int:
    """NBA via nba_api boxscore. Blocking lib, envuelve en to_thread."""
    nba_matches = [m for m in matches if m["sport_code"] == "nba"]
    if not nba_matches:
        return 0

    def _fetch_boxscore(game_id: str) -> dict[str, Any] | None:
        try:
            from nba_api.stats.endpoints import boxscoresummaryv2

            bs = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, timeout=20)
            summary = bs.get_normalized_dict()
            line_score = summary.get("LineScore", [])
            if len(line_score) < 2:
                return None
            return {
                "home_pts": int(line_score[1].get("PTS") or 0),
                "away_pts": int(line_score[0].get("PTS") or 0),
                "status": "Final",
            }
        except Exception as exc:  # nba_api a veces devuelve 500
            logger.debug("live_scores.nba_boxscore_fail", game_id=game_id, error=str(exc))
            return None

    updated = 0
    for match in nba_matches:
        ext = str(match.get("external_id") or "")
        if not ext:
            continue
        bs = await asyncio.to_thread(_fetch_boxscore, ext)
        if not bs or bs.get("status") != "Final":
            continue
        await _finalize_match(
            match_id=int(match["id"]),
            home_score=bs["home_pts"],
            away_score=bs["away_pts"],
            final_status="FT",
        )
        updated += 1
    return updated


@task(retries=2, retry_delay_seconds=30)
async def sync_nhl_scores(matches: list[dict[str, Any]]) -> int:
    """NHL via api-web.nhle.com (gratis, sin key)."""
    from apuestas.ingest.http_base import BaseAPIClient

    nhl_matches = [m for m in matches if m["sport_code"] == "nhl"]
    if not nhl_matches:
        return 0

    class _NHLClient(BaseAPIClient):
        base_url = "https://api-web.nhle.com/v1"
        source_name = "nhl"
        rate_limit = (60, 60.0)

        def _default_headers(self) -> dict[str, str]:
            return {"Accept": "application/json"}

    updated = 0
    client = _NHLClient(api_key="")
    async with client.session():
        for match in nhl_matches:
            ext = str(match.get("external_id") or "")
            if not ext:
                continue
            try:
                data = await client.get(f"/gamecenter/{ext}/boxscore", params=None)
            except Exception as exc:
                logger.debug("live_scores.nhl_fail", game_id=ext, error=str(exc))
                continue
            if data.get("gameState") not in {"OFF", "FINAL"}:
                continue
            home = data.get("homeTeam", {})
            away = data.get("awayTeam", {})
            hs = home.get("score")
            as_ = away.get("score")
            if hs is None or as_ is None:
                continue
            await _finalize_match(
                match_id=int(match["id"]),
                home_score=int(hs),
                away_score=int(as_),
                final_status="FT",
            )
            updated += 1
    return updated


@flow(name="apuestas-live-scores", log_prints=True)
async def live_scores_flow(*, window_hours: int = 48) -> dict[str, int]:
    matches = await pending_finished_matches(window_hours=window_hours)
    logger.info("live_scores.start", candidates=len(matches))
    if not matches:
        return {"candidates": 0, "updated_total": 0}

    # Prefect 3.x: llamamos .fn() directo + asyncio.gather para paralelismo
    results: list[Any] = await asyncio.gather(
        sync_soccer_scores.fn(matches),
        sync_nba_scores.fn(matches),
        sync_nhl_scores.fn(matches),
        return_exceptions=True,
    )
    soccer_n: int = results[0] if not isinstance(results[0], BaseException) else 0
    nba_n: int = results[1] if not isinstance(results[1], BaseException) else 0
    nhl_n: int = results[2] if not isinstance(results[2], BaseException) else 0

    total = soccer_n + nba_n + nhl_n
    logger.info(
        "live_scores.done",
        candidates=len(matches),
        soccer=soccer_n,
        nba=nba_n,
        nhl=nhl_n,
        total=total,
    )
    return {
        "candidates": len(matches),
        "soccer_updated": soccer_n,
        "nba_updated": nba_n,
        "nhl_updated": nhl_n,
        "updated_total": total,
    }


if __name__ == "__main__":
    asyncio.run(live_scores_flow())
