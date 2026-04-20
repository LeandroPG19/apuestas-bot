"""Flow post_mortem automático tras settlement de bets (§21).

Dispara cuando `live_scores` + `settle_bets` marca bets como won/lost/void.
Por cada bet settleada:
1. Recolectar snapshot prediction + actual final_score + key_events.
2. Compute 7 métricas discrepancy (§17.5, §21.1).
3. LLM genera narrativa con prompt post_mortem/v1.
4. INSERT en post_mortems.
5. cuba_eco feedback + cuba_cronica episode_add.
6. Si discrepancy_score > threshold → cuba_alarma pattern.
7. Update calibration_rolling agregado por bucket.
"""

from __future__ import annotations

import asyncio
from typing import Any

from prefect import flow, task
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.llm.client import LlamaClient
from apuestas.llm.router import run_task
from apuestas.mcp import memory as mcp_memory
from apuestas.ml.discrepancy import compute_discrepancy
from apuestas.obs.logging import get_logger
from apuestas.schemas.llm import PostMortemNarrative

logger = get_logger(__name__)


@task
async def fetch_settled_bets_without_postmortem(limit: int = 50) -> list[dict[str, Any]]:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT b.id AS bet_id, b.match_id, b.market, b.outcome, b.line,
                       b.stake_units, b.odds_placed, b.status, b.pnl_units, b.clv,
                       b.prediction_id, b.settled_at,
                       p.probability AS p_model, p.p_lower, p.p_upper,
                       p.ev AS ev_predicted, p.kelly_fraction AS kelly_predicted,
                       p.features_snapshot, p.shap_top5, p.llm_analysis,
                       p.model_name, p.model_version,
                       m.home_score, m.away_score
                FROM bets b
                LEFT JOIN predictions p ON p.id = b.prediction_id
                LEFT JOIN matches m ON m.id = b.match_id
                LEFT JOIN post_mortems pm ON pm.bet_id = b.id
                WHERE b.status IN ('won', 'lost', 'void')
                  AND pm.id IS NULL
                  AND b.settled_at >= NOW() - INTERVAL '30 days'
                ORDER BY b.settled_at DESC
                LIMIT :lim
                """
            ),
            {"lim": limit},
        )
        return [dict(r._mapping) for r in result.all()]


@task
async def fetch_actual_events(match_id: int) -> list[dict[str, Any]]:
    """Recolecta actual_key_events del match finalizado.

    En producción esto vendría de play-by-play API. Por ahora usa
    noticias post-match + lineups confirmados.
    """
    async with session_scope() as session:
        # News post-match como proxy de key events
        result = await session.execute(
            text(
                """
                SELECT title, content, published_at, sentiment_score
                FROM news_articles
                WHERE :mid = ANY(teams_mentioned::bigint[] ||
                                 (SELECT ARRAY[home_team_id, away_team_id]
                                  FROM matches WHERE id = :mid))
                  AND published_at >= (
                      SELECT start_time FROM matches WHERE id = :mid
                  )
                ORDER BY published_at ASC
                LIMIT 20
                """
            ),
            {"mid": match_id},
        )
        rows = [dict(r._mapping) for r in result.all()]

    return [
        {
            "description": f"{r.get('title', '')}: {(r.get('content') or '')[:150]}",
            "sentiment": float(r.get("sentiment_score") or 0.0),
            "timestamp": str(r.get("published_at")),
        }
        for r in rows
    ]


@task
async def generate_narrative(
    *,
    bet: dict[str, Any],
    actual_events: list[dict[str, Any]],
    discrepancy_dict: dict[str, Any],
) -> dict[str, Any]:
    """Usa LLM para producir PostMortemNarrative estructurada."""
    outcome_binary = 1 if bet["status"] == "won" else 0
    final_score = {
        "home": bet.get("home_score"),
        "away": bet.get("away_score"),
    }
    render_vars = {
        "sport": "unknown",
        "market": str(bet["market"]),
        "outcome": str(bet["outcome"]),
        "p_model": float(bet.get("p_model") or 0.5),
        "p_lower": float(bet.get("p_lower") or 0.0),
        "p_upper": float(bet.get("p_upper") or 1.0),
        "ev_predicted": float(bet.get("ev_predicted") or 0.0),
        "shap_top5": bet.get("shap_top5") or [],
        "llm_analysis_original": bet.get("llm_analysis") or {},
        "outcome_result": bet["status"],
        "outcome_binary": outcome_binary,
        "final_score": final_score,
        "pnl_units": float(bet["pnl_units"] or 0.0),
        "clv": bet.get("clv"),
        "key_events": actual_events,
        "actual_lineups": {},
        "prediction_error": discrepancy_dict.get("prediction_error"),
        "calibration_miss": discrepancy_dict.get("calibration_miss"),
        "llm_alignment_score": discrepancy_dict.get("llm_alignment_score"),
    }

    # Respeta LLM_BACKEND
    from apuestas.config import get_settings as _gs

    backend = _gs().llm.llm_backend
    if backend == "deepseek":
        from apuestas.llm.deepseek_client import DeepSeekClient

        llm_cls: Any = DeepSeekClient
    else:
        llm_cls = LlamaClient

    try:
        async with llm_cls() as llm:
            narrative: PostMortemNarrative = await run_task(  # type: ignore[assignment]
                task_kind="post_mortem",
                version="v1",
                client=llm,
                render_vars=render_vars,
            )
    except Exception as exc:
        logger.warning("post_mortem.llm_failed", bet_id=bet["bet_id"], error=str(exc))
        return _fallback_narrative(bet)

    return {
        "outcome": narrative.outcome,
        "prediction_quality": narrative.prediction_quality,
        "what_went_right": list(narrative.what_went_right),
        "what_went_wrong": list(narrative.what_went_wrong),
        "unexpected_factors": list(narrative.unexpected_factors),
        "if_we_had_known": narrative.if_we_had_known,
        "transferable_lesson": narrative.transferable_lesson,
        "tag_for_pattern_detection": list(narrative.tag_for_pattern_detection),
    }


def _fallback_narrative(bet: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": bet["status"],
        "prediction_quality": "off",
        "what_went_right": [],
        "what_went_wrong": ["llm_unavailable_fallback"],
        "unexpected_factors": [],
        "if_we_had_known": "LLM unavailable para analysis",
        "transferable_lesson": "pendiente manual review",
        "tag_for_pattern_detection": ["llm_fallback"],
    }


@task
async def persist_post_mortem(
    *,
    bet: dict[str, Any],
    actual_events: list[dict[str, Any]],
    discrepancy_dict: dict[str, Any],
    narrative: dict[str, Any],
) -> int:
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                INSERT INTO post_mortems
                  (bet_id, event_id,
                   prediction_snapshot, features_snapshot, shap_top5,
                   llm_analysis_snapshot,
                   ev_predicted, kelly_predicted,
                   outcome, pnl_units, clv,
                   actual_final_score, actual_key_events,
                   prediction_error, calibration_miss,
                   ev_realized, ev_realized_vs_predicted,
                   llm_alignment_score, shap_attribution_check,
                   line_movement_assessment_correct,
                   discrepancy_score,
                   narrative, pattern_tags)
                VALUES
                  (:bet_id, :event_id,
                   :prediction_snapshot, :features_snapshot, :shap_top5,
                   :llm_analysis_snapshot,
                   :ev_predicted, :kelly_predicted,
                   :outcome, :pnl_units, :clv,
                   :actual_final_score, :actual_key_events,
                   :prediction_error, :calibration_miss,
                   :ev_realized, :ev_realized_vs_predicted,
                   :llm_alignment_score, :shap_attribution_check,
                   :line_movement_assessment_correct,
                   :discrepancy_score,
                   :narrative, :pattern_tags)
                ON CONFLICT (bet_id) DO NOTHING
                RETURNING id
                """
            ),
            {
                "bet_id": bet["bet_id"],
                "event_id": bet["match_id"],
                "prediction_snapshot": {
                    "p_model": float(bet.get("p_model") or 0),
                    "p_lower": float(bet.get("p_lower") or 0),
                    "p_upper": float(bet.get("p_upper") or 0),
                    "odds_placed": float(bet["odds_placed"]),
                    "model": bet.get("model_name"),
                    "version": bet.get("model_version"),
                },
                "features_snapshot": bet.get("features_snapshot") or {},
                "shap_top5": bet.get("shap_top5") or [],
                "llm_analysis_snapshot": bet.get("llm_analysis") or {},
                "ev_predicted": float(bet.get("ev_predicted") or 0),
                "kelly_predicted": float(bet.get("kelly_predicted") or 0),
                "outcome": bet["status"],
                "pnl_units": float(bet["pnl_units"] or 0),
                "clv": float(bet["clv"]) if bet.get("clv") is not None else None,
                "actual_final_score": {
                    "home": bet.get("home_score"),
                    "away": bet.get("away_score"),
                },
                "actual_key_events": actual_events,
                **discrepancy_dict,
                "narrative": narrative,
                "pattern_tags": narrative.get("tag_for_pattern_detection", []),
            },
        )
        row = result.first()
    return int(row[0]) if row else 0


@task
async def register_in_memory(
    *,
    bet: dict[str, Any],
    narrative: dict[str, Any],
) -> None:
    """Feedback a cuba-memorys: eco + cronica + alarma si patrón."""
    await mcp_memory.record_bet_outcome(
        bet_id=int(bet["bet_id"]),
        result=str(bet["status"]),
        delta_units=float(bet["pnl_units"] or 0),
        clv=float(bet["clv"]) if bet.get("clv") is not None else None,
    )
    await mcp_memory.record_post_mortem_episode(
        bet_id=int(bet["bet_id"]),
        event_id=int(bet["match_id"]),
        narrative=narrative,
        actors=[f"team:{bet['match_id']}"],
    )

    # Si pattern tag recurrente, dispara alarma
    tags = narrative.get("tag_for_pattern_detection", [])
    for tag in tags[:3]:
        repetition = await mcp_memory.check_repetition(pattern_tag=tag, days=30)
        if repetition and isinstance(repetition, dict):
            # heurística: si el resultado incluye count >= 3 dispara alarma
            count = repetition.get("count") if isinstance(repetition.get("count"), int) else None
            if count and count >= 3:
                await mcp_memory.alarma(
                    trigger=f"pattern_{tag}_recurrent",
                    details={"count": count, "bet_id": bet["bet_id"]},
                )


@flow(name="apuestas-post-mortem", log_prints=True)
async def post_mortem_flow(*, batch_size: int = 50) -> dict[str, int]:
    bets = await fetch_settled_bets_without_postmortem(limit=batch_size)
    processed = 0
    for bet in bets:
        try:
            actual_events = await fetch_actual_events(int(bet["match_id"]))
            disc = compute_discrepancy(
                p_model=float(bet.get("p_model") or 0.5),
                outcome_binary=1 if bet["status"] == "won" else 0,
                ev_predicted=float(bet.get("ev_predicted") or 0),
                pnl_units=float(bet["pnl_units"] or 0),
                stake_units=float(bet["stake_units"]),
                llm_analysis=bet.get("llm_analysis"),
                shap_top5=bet.get("shap_top5"),
                actual_key_events=actual_events,
            )
            disc_dict = disc.to_dict()
            narrative = await generate_narrative(
                bet=bet, actual_events=actual_events, discrepancy_dict=disc_dict
            )
            await persist_post_mortem(
                bet=bet,
                actual_events=actual_events,
                discrepancy_dict=disc_dict,
                narrative=narrative,
            )
            await register_in_memory(bet=bet, narrative=narrative)
            processed += 1
        except Exception as exc:
            logger.exception("post_mortem.fail", bet_id=bet["bet_id"], error=str(exc))
    return {"checked": len(bets), "processed": processed}


if __name__ == "__main__":
    asyncio.run(post_mortem_flow())
