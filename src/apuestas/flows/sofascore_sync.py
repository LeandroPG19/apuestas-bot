"""Flow de sincronización Sofascore → matches table.

Descubre matches de todos los sports configurados que Sofascore tiene pero
el bot aún no ha ingestado (Pinnacle/Kambi/OddsAPI no los cubren).

Trigger:
- Manual: `python -m apuestas.flows.sofascore_sync`
- Automático: agregado a catchup_flow como task adicional
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.ingest._match_resolver import resolve_or_create_match
from apuestas.ingest.sofascore_scraper import _enabled, fetch_scheduled_events
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Sofascore sport slug → sport_code interno del bot
_SOFASCORE_TO_INTERNAL = {
    "football": "soccer",
    "basketball": "nba",
    "tennis": "tennis",
    "baseball": "mlb",
    "american-football": "nfl",
    "ice-hockey": "nhl",
}


@task(retries=1, retry_delay_seconds=10)
async def sync_sport_from_sofascore(sport_slug: str, *, days_ahead: int = 2) -> dict[str, int]:
    """Descubre matches próximos de un sport desde Sofascore y los persiste.

    Returns:
        {"fetched": N, "inserted": M, "skipped": K}
    """
    if not _enabled():
        return {"fetched": 0, "inserted": 0, "skipped": 0, "reason": "disabled"}

    internal_sport = _SOFASCORE_TO_INTERNAL.get(sport_slug)
    if internal_sport is None:
        logger.debug("sofascore_sync.unknown_sport", sport=sport_slug)
        return {"fetched": 0, "inserted": 0, "skipped": 0}

    fetched = 0
    inserted = 0
    skipped = 0

    today = datetime.now(tz=UTC).date()
    for offset in range(days_ahead + 1):
        date = today + timedelta(days=offset)
        date_str = date.isoformat()
        try:
            events = await fetch_scheduled_events(sport_slug, date_str)
        except Exception as exc:
            logger.debug(
                "sofascore_sync.fetch_fail",
                sport=sport_slug,
                date=date_str,
                error=str(exc)[:80],
            )
            continue

        fetched += len(events)
        for event in events:
            try:
                home = event.get("homeTeam", {}).get("name", "")
                away = event.get("awayTeam", {}).get("name", "")
                start_ts = event.get("startTimestamp")
                sf_id = event.get("id")

                if not home or not away or not start_ts or not sf_id:
                    skipped += 1
                    continue

                start_time = datetime.fromtimestamp(int(start_ts), tz=UTC)

                # Solo matches futuros (status scheduled)
                if start_time < datetime.now(tz=UTC) - timedelta(hours=1):
                    skipped += 1
                    continue

                async with session_scope() as s:
                    match_id = await resolve_or_create_match(
                        session=s,
                        sport_code=internal_sport,
                        home_name=home,
                        away_name=away,
                        start_time=start_time,
                        source="sofascore",
                    )
                    if match_id is None:
                        skipped += 1
                        continue

                    # Guardar sofascore_event_id en metadata para lookups futuros
                    await s.execute(
                        text(
                            """
                            UPDATE matches
                            SET metadata = (COALESCE(metadata::jsonb, '{}'::jsonb) ||
                                            jsonb_build_object('sofascore_event_id',
                                                               CAST(:sfid AS text)))::json
                            WHERE id = :mid
                              AND (metadata IS NULL
                                   OR metadata->>'sofascore_event_id' IS NULL)
                            """
                        ),
                        {"sfid": str(sf_id), "mid": match_id},
                    )
                    inserted += 1
            except Exception as exc:
                logger.debug(
                    "sofascore_sync.event_fail",
                    sport=sport_slug,
                    error=str(exc)[:80],
                )
                skipped += 1

    logger.info(
        "sofascore_sync.sport_done",
        sport=sport_slug,
        internal=internal_sport,
        fetched=fetched,
        inserted=inserted,
        skipped=skipped,
    )
    return {"fetched": fetched, "inserted": inserted, "skipped": skipped}


@flow(name="apuestas-sofascore-sync", log_prints=True)
async def sofascore_sync_flow(*, days_ahead: int = 2) -> dict[str, Any]:
    """Sincroniza todos los sports supported desde Sofascore.

    Descubre matches que Pinnacle/Kambi/OddsAPI no cubren (especialmente
    soccer segunda división, tennis Challenger, mercados menos líquidos).
    """
    if not _enabled():
        logger.info("sofascore_sync.disabled")
        return {"status": "disabled"}

    sports = list(_SOFASCORE_TO_INTERNAL.keys())
    results: dict[str, dict[str, int]] = {}

    # Concurrent con límite (rate-limit Sofascore ~2 req/s)
    sem = asyncio.Semaphore(2)

    async def _with_sem(sport: str) -> None:
        async with sem:
            try:
                trigger_fn = getattr(sync_sport_from_sofascore, "fn", sync_sport_from_sofascore)
                result = await trigger_fn(sport, days_ahead=days_ahead)
                results[sport] = result
            except Exception as exc:
                logger.warning("sofascore_sync.sport_fail", sport=sport, error=str(exc)[:100])
                results[sport] = {"error": str(exc)[:80]}

    await asyncio.gather(*[_with_sem(s) for s in sports], return_exceptions=True)

    total_inserted = sum(r.get("inserted", 0) for r in results.values() if isinstance(r, dict))
    total_fetched = sum(r.get("fetched", 0) for r in results.values() if isinstance(r, dict))
    logger.info(
        "sofascore_sync.flow_done",
        total_fetched=total_fetched,
        total_inserted=total_inserted,
    )
    return {
        "status": "ok",
        "total_fetched": total_fetched,
        "total_inserted": total_inserted,
        "per_sport": results,
    }


if __name__ == "__main__":
    asyncio.run(sofascore_sync_flow())
