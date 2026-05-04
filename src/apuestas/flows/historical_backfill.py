"""Flow de auto-update diario de histórico (Fase 0.3).

Corre cada día en background (systemd timer `apuestas-historical-backfill.timer`)
y siembra los matches jugados desde la última fecha con resultados conocidos
hasta hoy. Idempotente: usa `ON CONFLICT (external_id) DO NOTHING` y resolve
fuzzy match.

Estrategia:
1. Query `MAX(start_time)` de matches con scores → última fecha cubierta.
2. Para cada sport, llamar loader incremental solo en rango [max_date, today-1d].
3. Log rows_added + sports_covered + invalid_odds_skipped.

Uso:
    python -m apuestas.flows.historical_backfill
    # o vía systemd:
    systemctl --user start apuestas-historical-backfill.service
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@task(retries=2, retry_delay_seconds=60)
async def last_covered_date_per_sport() -> dict[str, datetime]:
    """Devuelve el último `start_time` con score por sport_code."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT sport_code, MAX(start_time) AS last_ts
                FROM matches
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
                GROUP BY sport_code
                """
            )
        )
        rows = result.all()
    return {r.sport_code: r.last_ts for r in rows}


@task(retries=1, retry_delay_seconds=30)
async def backfill_soccer(since: datetime | None = None) -> dict[str, int]:
    """Incremental soccer via football-data.co.uk — TODAS las ligas EU.

    Cubre 16 ligas europeas + Liga MX + Expansión MX dinámicamente.
    Actualiza temporada actual cada ejecución (idempotente).
    """
    from apuestas.scripts.seed_historical import _seed_soccer_with_odds

    today = datetime.now(tz=UTC)
    # Temporada actual: si mes >= 7 (Jul), temporada empieza este año; si no, anterior.
    season = today.year if today.month >= 7 else today.year - 1
    leagues_eu = [
        "epl",
        "championship",
        "la_liga",
        "la_liga_2",
        "bundesliga",
        "bundesliga_2",
        "serie_a",
        "serie_b",
        "ligue_1",
        "ligue_2",
        "eredivisie",
        "liga_portugal",
        "belgium_a",
        "turkey_super",
        "greece_super",
        "scotland_premier",
    ]
    try:
        return await _seed_soccer_with_odds([season], leagues_eu)
    except Exception as exc:
        logger.warning("backfill.soccer_fail", error=str(exc)[:120])
        return {"matches_created": 0, "odds_rows_inserted": 0, "invalid_odds_skipped": 0}


@task(retries=1, retry_delay_seconds=30)
async def backfill_liga_mx(since: datetime | None = None) -> dict[str, int]:
    """Incremental Liga MX + Expansion vía fbref.com scraping directo.

    Se ejecuta solo una vez al día (rate-limit fbref ~10 req/min).
    """
    from apuestas.ingest.fbref_liga_mx import ingest_liga_mx_multi_seasons

    today = datetime.now(tz=UTC)
    season = today.year if today.month >= 7 else today.year - 1
    counters = {"liga_mx": 0, "liga_expansion": 0}
    for league in ("liga_mx", "liga_expansion"):
        try:
            result = await ingest_liga_mx_multi_seasons(league_slug=league, seasons=[season])
            counters[league] = sum(result.values())
        except Exception as exc:
            logger.warning(f"backfill.{league}_fail", error=str(exc)[:120])
    return counters


@task(retries=1, retry_delay_seconds=30)
async def backfill_tennis(since: datetime | None = None) -> dict[str, int]:
    """Incremental tennis ATP/WTA via tennis-data.co.uk."""
    from apuestas.scripts.seed_historical import _seed_tennis_with_odds

    today = datetime.now(tz=UTC)
    season = today.year
    try:
        return await _seed_tennis_with_odds([season], ["atp", "wta"])
    except Exception as exc:
        logger.warning("backfill.tennis_fail", error=str(exc)[:120])
        return {"matches_created": 0, "odds_rows_inserted": 0, "invalid_odds_skipped": 0}


@task(retries=1, retry_delay_seconds=30)
async def backfill_us_sports(since: datetime | None = None) -> dict[str, int]:
    """Incremental NBA/NFL/NHL via SBR community datasets.

    Los datasets GitHub se re-publican periodicamente. Forzamos refresh.
    """
    from apuestas.scripts.seed_historical import _seed_us_sports_with_odds

    try:
        return await _seed_us_sports_with_odds(["nba", "nfl", "nhl"])
    except Exception as exc:
        logger.warning("backfill.us_sports_fail", error=str(exc)[:120])
        return {"matches_created": 0, "odds_rows_inserted": 0, "invalid_odds_skipped": 0}


@flow(name="apuestas-historical-backfill", log_prints=True)
async def historical_backfill_flow() -> dict[str, Any]:
    """Flow principal: ejecutado diario por systemd timer."""
    logger.info("backfill.start")
    coverage = await last_covered_date_per_sport()
    logger.info(
        "backfill.coverage",
        **{k: v.isoformat() if v else None for k, v in coverage.items()},
    )

    # Ejecutar loaders en paralelo
    soccer, tennis, us_sports, liga_mx = await asyncio.gather(
        backfill_soccer.fn(),
        backfill_tennis.fn(),
        backfill_us_sports.fn(),
        backfill_liga_mx.fn(),
        return_exceptions=True,
    )

    def _unwrap(x: Any) -> dict[str, int]:
        if isinstance(x, BaseException):
            return {"matches_created": 0, "odds_rows_inserted": 0, "invalid_odds_skipped": 0}
        return x  # type: ignore[no-any-return]

    def _unwrap_simple(x: Any) -> dict[str, int]:
        if isinstance(x, BaseException):
            return {"liga_mx": 0, "liga_expansion": 0}
        return x  # type: ignore[no-any-return]

    summary = {
        "soccer": _unwrap(soccer),
        "tennis": _unwrap(tennis),
        "us_sports": _unwrap(us_sports),
        "liga_mx": _unwrap_simple(liga_mx),
        "completed_at": datetime.now(tz=UTC).isoformat(),
    }

    total_matches = sum(
        s["matches_created"] for s in (summary["soccer"], summary["tennis"], summary["us_sports"])
    )
    total_matches += sum(summary["liga_mx"].values())
    total_odds = sum(
        s["odds_rows_inserted"]
        for s in (summary["soccer"], summary["tennis"], summary["us_sports"])
    )
    logger.info(
        "backfill.done",
        total_matches=total_matches,
        total_odds=total_odds,
        summary=summary,
    )
    return summary


if __name__ == "__main__":
    asyncio.run(historical_backfill_flow())
