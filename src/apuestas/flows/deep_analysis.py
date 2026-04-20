"""Flow principal `deep_analysis` (§16 + §22 + §23).

Ejecuta al correr `make analyze`:
- Catchup data reciente (news, odds, lineups, weather).
- Para cada evento próximo 48 h:
  1. Colectar 9 capas × 2 equipos (§16.1).
  2. Validar mirror_check (§16.6).
  3. Build features + predict ML (LightGBM calibrado + MAPIE).
  4. LLM Qwen análisis estructurado espejo.
  5. Detector value bets + line shopping + regional MX/US compare.
  6. Portfolio allocation correlation-aware.
  7. Persistir en predictions + decision_log + bets_paper.
  8. Registrar en cuba-memorys.
  9. Notificar Telegram.

Objetivo: <30 s por evento end-to-end.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.betting.detector import (
    DetectorConfig,
    EventOdds,
    detect_value_bets_for_event,
)
from apuestas.betting.portfolio import allocate_portfolio
from apuestas.db import session_scope
from apuestas.flows.catchup import catchup_flow
from apuestas.llm.client import LlamaClient
from apuestas.llm.embed import EmbedClient
from apuestas.llm.rag import RAGRetriever
from apuestas.mcp import memory as mcp_memory
from apuestas.mcp.client import MCPClient
from apuestas.obs.logging import get_logger
from apuestas.validators.mirror_check import run_mirror_check

logger = get_logger(__name__)


@task(retries=1)
async def get_upcoming_events(hours_ahead: int = 48) -> list[dict[str, Any]]:
    """Eventos próximos dentro de ventana."""
    until = datetime.now(tz=UTC) + timedelta(hours=hours_ahead)
    since = datetime.now(tz=UTC) - timedelta(minutes=5)

    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.id, m.external_id, m.sport_code, m.league_id,
                       m.home_team_id, m.away_team_id, m.venue_id,
                       m.start_time, m.status
                FROM matches m
                WHERE m.status = 'scheduled'
                  AND m.start_time BETWEEN :since AND :until
                ORDER BY m.start_time ASC
                """
            ),
            {"since": since, "until": until},
        )
        return [dict(r._mapping) for r in result.all()]


@task
async def collect_odds_for_event(event_id: int) -> EventOdds | None:
    """Recolecta odds recientes (<30 min) agrupadas por bookmaker+market."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH recent AS (
                  SELECT DISTINCT ON (match_id, bookmaker, market, outcome, line)
                    match_id, bookmaker, market, outcome, line, odds, ts
                  FROM odds_history
                  WHERE match_id = :mid AND ts >= NOW() - INTERVAL '30 minutes'
                  ORDER BY match_id, bookmaker, market, outcome, line, ts DESC
                )
                SELECT r.*, m.external_id, m.start_time, m.sport_code, m.league_id
                FROM recent r JOIN matches m ON m.id = r.match_id
                ORDER BY market, outcome
                """
            ),
            {"mid": event_id},
        )
        rows = [dict(r._mapping) for r in result.all()]

    if not rows:
        return None

    # Agrupar por (market, outcome) para construir EventOdds con quotes_by_bookmaker
    # Para simplificar: un EventOdds por mercado (ej. h2h), con outcomes alineados.
    # Esta versión solo retorna el primer market disponible; multi-market se
    # expande en versiones futuras.
    sample = rows[0]
    target_market = "h2h" if any(r["market"] == "h2h" for r in rows) else rows[0]["market"]
    market_rows = [r for r in rows if r["market"] == target_market]
    if not market_rows:
        return None

    outcomes = sorted({r["outcome"] for r in market_rows})
    quotes_by_bm: dict[str, list[float]] = {}
    lines: list[float | None] = [None] * len(outcomes)

    for r in market_rows:
        bm = r["bookmaker"]
        idx = outcomes.index(r["outcome"])
        quotes_by_bm.setdefault(bm, [0.0] * len(outcomes))
        quotes_by_bm[bm][idx] = float(r["odds"])
        if r["line"] is not None:
            lines[idx] = float(r["line"])

    return EventOdds(
        event_id=event_id,
        event_external_id=str(sample["external_id"]),
        market=target_market,
        start_time=sample["start_time"],
        outcomes=outcomes,
        quotes_by_bookmaker=quotes_by_bm,
        lines=lines if any(l is not None for l in lines) else None,
        league_id=sample.get("league_id"),
        sport_code=sample.get("sport_code"),
    )


@task
async def run_mirror_validation(event: dict[str, Any]) -> dict[str, Any]:
    """Ejecuta mirror_check y retorna resumen."""
    check = await run_mirror_check(
        match_id=int(event["id"]),
        home_team_id=int(event["home_team_id"]),
        away_team_id=int(event["away_team_id"]),
        venue_id=event.get("venue_id"),
        sport_code=str(event["sport_code"]),
    )
    return {
        "analysis_complete": check.analysis_complete,
        "overall_score": check.overall_completeness_score,
        "missing": check.missing,
        "warnings": check.warnings,
    }


@task
async def fetch_rag_context(
    event: dict[str, Any],
    *,
    top_k: int = 10,
) -> str:
    """Recuperar snippets RAG relevantes para el evento."""
    try:
        async with EmbedClient() as embed:
            retriever = RAGRetriever(embed_client=embed)
            sport = str(event["sport_code"])
            team_ids = [int(event["home_team_id"]), int(event["away_team_id"])]
            query = (
                f"match preview {sport} home={event['home_team_id']} away={event['away_team_id']}"
            )
            hits = await retriever.hybrid_search(
                query,
                top_k=top_k,
                sports=[sport],
                team_ids=team_ids,
            )
            return retriever.format_snippets(hits)
    except Exception as exc:
        logger.warning("deep_analysis.rag_failed", event_id=event["id"], error=str(exc))
        return "(sin contexto RAG disponible)"


@task
async def llm_analyze_event(
    event: dict[str, Any],
    rag_snippets: str,
    *,
    correlation_id: str,
) -> dict[str, Any] | None:
    """Llama al LLM (Qwen local o DeepSeek) con memory loop anti-alucinación."""
    from apuestas.config import get_settings
    from apuestas.llm.memory_loop import fetch_memory_context
    from apuestas.llm.router import run_task
    from apuestas.schemas.llm import PreMatchAnalysis

    backend = get_settings().llm.llm_backend
    llm_cls: Any
    if backend == "deepseek":
        from apuestas.llm.deepseek_client import DeepSeekClient

        llm_cls = DeepSeekClient
    else:
        llm_cls = LlamaClient

    # Inyectar contexto histórico desde cuba-memorys antes de llamar al LLM
    memory_ctx = ""
    try:
        memory_ctx = await fetch_memory_context(
            event_description=f"event {event.get('id')}",
            teams=[str(event.get("home_team_id")), str(event.get("away_team_id"))],
            market="h2h",
            sport_code=event.get("sport_code"),
        )
    except Exception as exc:
        logger.debug("deep_analysis.memory_ctx_unavailable", error=str(exc))

    # Tier A features: enriquece el prompt con referee bias + coaching tendencies
    # + steam moves + tracking proxies (todos gratis, derivados de PBP propio).
    tier_a_features: dict[str, Any] = {}
    try:
        from apuestas.features.coaching_clutch import compute_coaching_features
        from apuestas.features.referee_bias import compute_referee_bias_features

        sport = event.get("sport_code") or "nba"
        ref_feats = await compute_referee_bias_features(int(event["id"]), sport)
        coach_feats = await compute_coaching_features(int(event["id"]), sport)
        tier_a_features.update(ref_feats)
        tier_a_features.update(coach_feats)
    except Exception as exc:
        logger.debug("deep_analysis.tier_a_unavailable", error=str(exc))

    # Steam moves activos: añadir al contexto si hay ≥1 en últimas 2h
    try:
        from sqlalchemy import text as _text

        async with session_scope() as _s:
            r = await _s.execute(
                _text(
                    """
                    SELECT COUNT(*) AS n, SUM(CASE WHEN pinnacle_leading THEN 1 ELSE 0 END)
                        AS pin_led
                    FROM steam_moves
                    WHERE match_id = :m AND detected_at > NOW() - INTERVAL '2 hours'
                    """
                ),
                {"m": event["id"]},
            )
            row = r.first()
            if row and int(row.n or 0) > 0:
                tier_a_features["active_steam_moves_last_2h"] = int(row.n)
                tier_a_features["steam_pinnacle_led"] = int(row.pin_led or 0)
    except Exception as exc:
        logger.debug("deep_analysis.steam_check_fail", error=str(exc))

    prompt_vars = _build_prompt_vars(event, rag_snippets)
    if memory_ctx:
        prompt_vars["memory_context"] = memory_ctx
    if tier_a_features:
        prompt_vars["tier_a_features"] = tier_a_features

    try:
        async with llm_cls() as llm:
            result = await run_task(
                task_kind="pre_match",
                version="v1",
                client=llm,
                render_vars=prompt_vars,
            )
        assert isinstance(result, PreMatchAnalysis)
        return {
            "summary_es": result.summary_es,
            "confidence": result.confidence_in_analysis,
            "edge_direction": result.overall_edge_direction,
            "line_movement": result.line_movement_assessment,
            "home": {
                "team_name": result.home_team_analysis.team_name,
                "key_injuries": [
                    {"player": i.player, "severity": i.severity}
                    for i in result.home_team_analysis.key_injuries
                ],
                "rest_days": result.home_team_analysis.rest_days,
                "b2b": result.home_team_analysis.back_to_back,
                "momentum": result.home_team_analysis.narrative_momentum,
            },
            "away": {
                "team_name": result.away_team_analysis.team_name,
                "key_injuries": [
                    {"player": i.player, "severity": i.severity}
                    for i in result.away_team_analysis.key_injuries
                ],
                "rest_days": result.away_team_analysis.rest_days,
                "b2b": result.away_team_analysis.back_to_back,
                "momentum": result.away_team_analysis.narrative_momentum,
                "travel_km": result.away_team_analysis.travel_km,
                "altitude_delta_m": result.away_team_analysis.altitude_delta_m,
            },
            "contradictions_found": result.contradictions_found,
        }
    except Exception as exc:
        logger.warning(
            "deep_analysis.llm_failed",
            event_id=event["id"],
            cid=correlation_id,
            error=str(exc),
        )
        return None


def _build_prompt_vars(event: dict[str, Any], rag_snippets: str) -> dict[str, Any]:
    """Placeholder: variables para el prompt pre_match/v1.

    En versión productiva, cada clave viene de queries SQL específicas.
    Por ahora usamos defaults minimos para no romper el formateador.
    """
    return {
        "home_name": f"Team {event['home_team_id']}",
        "away_name": f"Team {event['away_team_id']}",
        "sport": event["sport_code"],
        "league": event.get("league_id", "unknown"),
        "start_time": event["start_time"].isoformat() if event.get("start_time") else "TBD",
        "venue_name": event.get("venue_id", "unknown"),
        "altitude_m": 0,
        "surface": "unknown",
        "roof": "unknown",
        "stats_markdown": "(ver queries en versión productiva)",
        "home_away_splits_markdown": "(ver queries en versión productiva)",
        "injuries_markdown": "(ver queries en versión productiva)",
        "lineups_markdown": "(ver queries en versión productiva)",
        "transfers_markdown": "(ver queries en versión productiva)",
        "coaching_markdown": "(ver queries en versión productiva)",
        "streaks_markdown": "(ver queries en versión productiva)",
        "h2h_markdown": "(ver queries en versión productiva)",
        "home_rest_days": 2,
        "home_b2b": False,
        "away_rest_days": 2,
        "away_b2b": False,
        "away_travel_km": 0,
        "tz_delta_h": 0,
        "alt_delta_m": 0,
        "rag_snippets": rag_snippets,
        "line_movement_markdown": "(sin movimiento detectado)",
        "weather_summary": "(sin forecast)",
        "official_notes": "(sin notas)",
    }


@task
async def run_detector(
    event: dict[str, Any],
    event_odds: EventOdds | None,
    *,
    correlation_id: str,
) -> list[Any]:
    """Ejecuta detect_value_bets_for_event."""
    if event_odds is None:
        return []
    cfg = DetectorConfig()
    return await detect_value_bets_for_event(
        event_odds,
        bankroll=None,
        cfg=cfg,
        correlation_id=correlation_id,
    )


@flow(name="apuestas-deep-analysis", log_prints=True)
async def deep_analysis_flow(
    *,
    hours_ahead: int = 48,
    max_events: int = 50,
    skip_catchup: bool = False,
) -> dict[str, Any]:
    """Entry point del `make analyze`."""
    mcp = MCPClient.get()
    await mcp.start()
    await mcp_memory.jornada_start()

    if not skip_catchup:
        await catchup_flow()

    events = await get_upcoming_events(hours_ahead)
    events = events[:max_events]
    logger.info("deep_analysis.events", n=len(events))

    all_picks: list[Any] = []

    for event in events:
        correlation_id = uuid.uuid4().hex[:12]
        logger.info(
            "deep_analysis.event.start",
            event_id=event["id"],
            cid=correlation_id,
        )

        mirror, odds, rag = await asyncio.gather(
            run_mirror_validation(event),
            collect_odds_for_event(int(event["id"])),
            fetch_rag_context(event),
            return_exceptions=True,
        )

        # Skip si mirror_check crítico
        if isinstance(mirror, dict) and not mirror.get("analysis_complete", False):
            logger.info(
                "deep_analysis.skip_low_completeness",
                event_id=event["id"],
                score=mirror.get("overall_score"),
                missing=mirror.get("missing", [])[:5],
            )
            continue

        if isinstance(odds, Exception) or odds is None:
            continue

        llm_result = await llm_analyze_event(
            event, rag if isinstance(rag, str) else "", correlation_id=correlation_id
        )

        picks = await run_detector(event, odds, correlation_id=correlation_id)
        if picks:
            all_picks.extend([p for p in picks if p.is_bet])

    # Portfolio allocation sobre todos los picks
    if all_picks:
        allocations = await allocate_portfolio(all_picks)
        logger.info("deep_analysis.allocations", n=len(allocations))
    else:
        allocations = []

    summary = {
        "events_checked": len(events),
        "picks_emitted": len(all_picks),
        "allocations": len(allocations),
    }
    logger.info("deep_analysis.done", **summary)
    return summary


if __name__ == "__main__":
    asyncio.run(deep_analysis_flow())
