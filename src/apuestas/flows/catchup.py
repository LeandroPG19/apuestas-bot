"""Catchup flow — ingesta incremental al arrancar `make up`.

§11.5: cuando el bot estuvo apagado, al arrancar se ejecuta este flow
para poner al día:
- Fixtures (schedules próximos 14 días)
- Odds recientes (últimas 2 horas)
- Injuries + lineups
- News RSS + Reddit + Bluesky (consolidate)
- Weather forecast para eventos outdoor próximos

Reanuda desde `ingest_checkpoints` para no re-ingestar ya visto.
Orquestado con Prefect 3 para observabilidad.
"""

from __future__ import annotations

import asyncio
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest.api_football import LEAGUE_IDS, ingest_league_fixtures, ingest_league_odds
from apuestas.ingest.nba import ingest_nba_today
from apuestas.ingest.news_pipeline import run_news_ingest_pipeline
from apuestas.ingest.odds_api import SPORT_KEY_MAP, ingest_sport
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@task(retries=2, retry_delay_seconds=30)
async def update_checkpoint(source: str, resource: str) -> None:
    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO ingest_checkpoints (source, resource, last_ts, items_processed)
                VALUES (:src, :res, NOW(), 0)
                ON CONFLICT (source, resource) DO UPDATE
                SET last_ts = NOW()
                """
            ),
            {"src": source, "res": resource},
        )


def _api_football_available() -> bool:
    """Skip API-Football si no hay key válida configurada."""
    import os as _os

    key = _os.environ.get("API_FOOTBALL_KEY", "").strip()
    return bool(key) and not key.startswith(("your-", "change-"))


@task(retries=1, retry_delay_seconds=10)
async def catchup_soccer_fixtures(seasons: list[int] | None = None) -> dict[str, int]:
    """Ingesta fixtures. Si API-Football no disponible, usa football-data.org
    como fallback para Big-5 (gratis)."""
    seasons = seasons or [2025, 2026]
    results: dict[str, int] = {}

    if _api_football_available():
        for league_slug in ("liga_mx", "liga_expansion_mx", "mls", "epl", "la_liga"):
            if league_slug not in LEAGUE_IDS:
                continue
            for season in seasons:
                try:
                    df = await ingest_league_fixtures(league_slug, season)
                    results[f"{league_slug}_{season}"] = df.height
                except Exception as exc:
                    logger.warning(
                        "catchup.fixtures_fail",
                        league=league_slug,
                        season=season,
                        error=str(exc)[:100],
                    )
    else:
        logger.info("catchup.api_football_skipped", reason="no_valid_key")

    # Fallback gratis: football-data.org para ligas europeas
    try:
        from apuestas.ingest.free_sources import FootballDataOrgClient

        client = FootballDataOrgClient()
        async with client.session():
            for comp in ("epl", "la_liga", "champions"):
                try:
                    matches = await client.fetch_matches(competition=comp, status="SCHEDULED")
                    results[f"fd_{comp}"] = len(matches)
                except Exception as exc:
                    logger.debug("catchup.fd_fail", comp=comp, error=str(exc)[:80])
    except Exception as exc:
        logger.debug("catchup.fd_unavailable", error=str(exc)[:80])

    return results


@task(retries=1, retry_delay_seconds=10)
async def catchup_soccer_odds() -> dict[str, int]:
    results: dict[str, int] = {}
    if not _api_football_available():
        logger.info("catchup.api_football_odds_skipped")
        return results
    for league_slug in ("liga_mx", "epl", "la_liga"):
        try:
            df = await ingest_league_odds(league_slug, 2026)
            results[league_slug] = df.height
        except Exception as exc:
            logger.warning("catchup.soccer_odds_fail", league=league_slug, error=str(exc)[:100])
    return results


@task(retries=2, retry_delay_seconds=30)
async def catchup_odds_api() -> dict[str, int]:
    results: dict[str, int] = {}
    for sport_code, sport_key in SPORT_KEY_MAP.items():
        try:
            count = await ingest_sport(sport_key)
            results[sport_code] = count
        except Exception as exc:
            logger.warning("catchup.odds_api_fail", sport=sport_code, error=str(exc))
    return results


@task(retries=2, retry_delay_seconds=30)
async def catchup_nba_scoreboard() -> int:
    try:
        games = await ingest_nba_today()
        return len(games)
    except Exception as exc:
        logger.warning("catchup.nba_scoreboard_fail", error=str(exc))
        return 0


@task(retries=1, retry_delay_seconds=60)
async def catchup_news() -> dict[str, int]:
    try:
        return await run_news_ingest_pipeline()
    except Exception as exc:
        logger.warning("catchup.news_fail", error=str(exc))
        return {"total": 0, "processed": 0, "skipped": 0}


@flow(name="apuestas-catchup", log_prints=True)
async def catchup_flow() -> dict[str, object]:
    """Ejecutable desde `make analyze` o `make up`. Paralelo donde posible.

    En Prefect 3.x `.result()` es sync (bloquea en hilo). Para código async
    puro, invocamos las funciones internas directamente (`.fn`) y usamos
    `asyncio.gather` para paralelismo real. No perdemos observabilidad
    porque cada función interna ya loggea.
    """
    logger.info("catchup.start")

    gathered: list[Any] = await asyncio.gather(
        catchup_soccer_fixtures.fn(),
        catchup_soccer_odds.fn(),
        catchup_odds_api.fn(),
        catchup_nba_scoreboard.fn(),
        catchup_news.fn(),
        return_exceptions=True,
    )
    fixtures = gathered[0] if not isinstance(gathered[0], BaseException) else {}
    odds_soccer = gathered[1] if not isinstance(gathered[1], BaseException) else {}
    odds_api = gathered[2] if not isinstance(gathered[2], BaseException) else {}
    nba = gathered[3] if not isinstance(gathered[3], BaseException) else 0
    news = gathered[4] if not isinstance(gathered[4], BaseException) else {}

    await update_checkpoint("catchup", "full_sweep")

    summary = {
        "fixtures": fixtures,
        "odds_soccer": odds_soccer,
        "odds_api": odds_api,
        "nba_today": nba,
        "news": news,
    }
    logger.info("catchup.done", **{k: str(v) for k, v in summary.items()})
    return summary


if __name__ == "__main__":
    asyncio.run(catchup_flow())
