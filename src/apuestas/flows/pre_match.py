"""Flow pre_match — trigger por evento específico (T-45min) §13.

Variante ligera de deep_analysis para UN evento concreto (típicamente
lanzado desde worker-ml con trigger at_start_time-45min).

Delega a deep_analysis_flow restringido a ese event_id.
"""

from __future__ import annotations

import asyncio
import uuid

from prefect import flow
from sqlalchemy import text

from apuestas.betting.portfolio import allocate_portfolio
from apuestas.db import session_scope
from apuestas.flows.deep_analysis import (
    collect_odds_for_event,
    fetch_rag_context,
    llm_analyze_event,
    run_detector,
    run_mirror_validation,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@flow(name="apuestas-pre-match", log_prints=True)
async def pre_match_flow(event_id: int) -> dict[str, object]:
    """Analiza un evento específico T-45min antes de su inicio."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, external_id, sport_code, league_id,
                       home_team_id, away_team_id, venue_id, start_time
                FROM matches WHERE id = :id
                """
            ),
            {"id": event_id},
        )
        row = result.first()
    if row is None:
        logger.warning("pre_match.event_not_found", event_id=event_id)
        return {"found": False}

    event = dict(row._mapping)
    correlation_id = uuid.uuid4().hex[:12]

    mirror = await run_mirror_validation(event)
    if not mirror.get("analysis_complete"):
        logger.info(
            "pre_match.skip_incomplete",
            event_id=event_id,
            missing=mirror.get("missing", [])[:5],
        )
        return {"found": True, "skipped": True, "reason": "incomplete_data"}

    odds, rag = await asyncio.gather(
        collect_odds_for_event(event_id),
        fetch_rag_context(event),
        return_exceptions=True,
    )
    rag_text = rag if isinstance(rag, str) else ""

    llm_result = await llm_analyze_event(event, rag_text, correlation_id=correlation_id)
    picks = await run_detector(
        event, odds if not isinstance(odds, Exception) else None, correlation_id=correlation_id
    )

    if picks:
        bet_picks = [p for p in picks if p.is_bet]
        allocations = await allocate_portfolio(bet_picks) if bet_picks else []
    else:
        allocations = []

    return {
        "found": True,
        "skipped": False,
        "picks_count": len(picks) if picks else 0,
        "allocations": len(allocations),
        "llm_ok": llm_result is not None,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: pre_match.py <event_id>")
        sys.exit(1)
    asyncio.run(pre_match_flow(int(sys.argv[1])))
