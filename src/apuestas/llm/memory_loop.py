"""Loop de retroalimentación LLM ↔ cuba-memorys anti-alucinación.

Pipeline:
1. **Pre-análisis** — `fetch_memory_context(event)`:
   - `faro("evento X teams Y contexto")` → memorias relevantes
   - `check_repetition(pattern)` → detecta patrones repetitivos de pérdida
   - `get_calibrated_confidence(model, level)` → histórico de aciertos real
   - Retorna un bloque de contexto para inyectar al system prompt.

2. **LLM call** — usa DeepSeek (o llama local) con contexto pre-inyectado.

3. **Post-análisis** — `validate_and_record(analysis)`:
   - `scan_contradictions` sobre el análisis nuevo vs memoria
   - Si hay conflicto crítico → flag para review
   - `record_bet_decision(...)` persiste la decisión como `cuba_decreto`
   - Si pattern_tag previamente perdedor → `alarma`

4. **Bet settlement feedback** (cuando bet resuelve):
   - `record_bet_outcome(bet_id, result, pnl)` → cuba_eco
   - Si discrepancy alta → `alarma(pattern)` para futuras sesiones

Los LLMs alucinan cuando carecen de contexto específico. Este loop reduce
alucinaciones al ~inyectar memoria verificada antes de cada llamada.
"""

from __future__ import annotations

from typing import Any

from apuestas.mcp import memory as mcp_memory
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def fetch_memory_context(
    *,
    event_description: str,
    teams: list[str],
    market: str | None = None,
    sport_code: str | None = None,
    model_version: str | None = None,
    confidence_level: str = "medium",
    max_memories: int = 8,
) -> str:
    """Construye bloque de contexto histórico para inyectar al prompt.

    Combina resultados de cuba_faro + check_repetition + calibración bayesiana.
    Si cuba-memorys no está disponible, retorna string vacío (degrada gracilmente).
    """
    context_parts: list[str] = []

    # 1. Memoria semántica sobre el evento
    try:
        import json as _json

        faro_query = (
            f"evento {event_description}; equipos {' vs '.join(teams)}; "
            f"mercado {market or 'cualquiera'}"
        )
        faro_result = await mcp_memory.faro(faro_query, fmt="compact")
        memories: list[Any] = []
        if faro_result and isinstance(faro_result, dict):
            # cuba-memorys v0.6 devuelve JSON envuelto en text_chunks
            for chunk in faro_result.get("text_chunks", []):
                try:
                    parsed = _json.loads(chunk) if isinstance(chunk, str) else chunk
                    if isinstance(parsed, dict):
                        rs = parsed.get("results") or parsed.get("memories") or []
                        if isinstance(rs, list):
                            memories.extend(rs)
                except (_json.JSONDecodeError, TypeError):  # fmt: skip
                    continue
            # Fallback a keys directas
            if not memories:
                memories = faro_result.get("memories") or faro_result.get("results") or []

        if memories:
            context_parts.append("## Memorias relevantes (cuba-memorys)")
            for m in memories[:max_memories]:
                if isinstance(m, dict):
                    summary = m.get("c") or m.get("content") or m.get("text") or str(m)[:200]
                    entity = m.get("e") or ""
                    date = m.get("date", "")
                    prefix = f"[{entity}]" if entity else f"[{date}]" if date else ""
                    context_parts.append(f"- {prefix} {str(summary)[:200]}")
                else:
                    context_parts.append(f"- {str(m)[:200]}")
    except Exception as exc:
        logger.debug("memory_loop.faro_unavailable", error=str(exc))

    # 2. Check de repetición de patrones
    if market:
        try:
            pattern_tag = f"{sport_code or 'any'}_{market}"
            rep_result = await mcp_memory.check_repetition(pattern_tag=pattern_tag, days=30)
            if rep_result and isinstance(rep_result, dict):
                count = rep_result.get("count")
                wins = rep_result.get("wins")
                if count and count >= 3:
                    context_parts.append(
                        f"\n## ⚠️ Patrón detectado últimos 30d: "
                        f"{count} bets similares, {wins or 0} ganadas "
                        f"(WR={(wins or 0) / count:.1%}). "
                        f"Si tendencia negativa, evaluar con extra cautela."
                    )
        except Exception as exc:
            logger.debug("memory_loop.repetition_unavailable", error=str(exc))

    # 3. Confianza calibrada bayesiana
    if model_version:
        try:
            cal = await mcp_memory.get_calibrated_confidence(
                model_version=model_version, confidence_level=confidence_level
            )
            if cal and isinstance(cal, dict):
                accuracy = cal.get("calibrated_accuracy") or cal.get("accuracy")
                n = cal.get("n_samples") or cal.get("n")
                if accuracy is not None and n:
                    context_parts.append(
                        f"\n## Calibración histórica modelo {model_version}: "
                        f"confianza '{confidence_level}' acertó {accuracy:.1%} "
                        f"en n={n} predicciones pasadas."
                    )
        except Exception as exc:
            logger.debug("memory_loop.calibration_unavailable", error=str(exc))

    if not context_parts:
        return ""

    header = (
        "\n═══ CONTEXTO DE MEMORIA LARGA (usar como fuente de verdad,\n"
        "no inventar datos fuera de esto) ═══\n"
    )
    footer = (
        "\n═══ FIN CONTEXTO MEMORIA ═══\n"
        "INSTRUCCIONES ANTI-ALUCINACIÓN:\n"
        "- Si el contexto anterior contradice algo que podrías afirmar, ajústate al contexto.\n"
        "- Si te piden un hecho específico (lesiones, stats) que NO está en el contexto,\n"
        "  márcalo como 'no_verificado' en warning_flags en lugar de inventarlo.\n"
        "- Tu análisis debe basarse en los datos del usuario + este contexto, nunca en\n"
        "  información fuera del cutoff del modelo que no esté aquí confirmada.\n"
    )
    return header + "\n".join(context_parts) + footer


async def validate_and_record_decision(
    *,
    analysis: Any,
    bet_id: int | None,
    event_id: int,
    market: str,
    outcome: str,
    p_model: float,
    ev: float,
    kelly_pct: float,
    sport_code: str | None = None,
) -> dict[str, Any]:
    """Valida output del LLM contra memoria + persiste decisión.

    Retorna dict con:
        - "recorded": bool (si cuba_decreto registró)
        - "alarms": list[str] (alarmas activadas por contradicciones/patrones)
        - "contradictions": list[dict]
    """
    result: dict[str, Any] = {
        "recorded": False,
        "alarms": [],
        "contradictions": [],
    }

    # 1. Scan de contradicciones entre análisis y memoria
    try:
        contradict = await mcp_memory.scan_contradictions()
        if contradict and isinstance(contradict, dict):
            items = contradict.get("contradictions") or []
            result["contradictions"] = items[:5]
            if items:
                logger.info(
                    "memory_loop.contradictions_found",
                    count=len(items),
                    event_id=event_id,
                )
    except Exception as exc:
        logger.debug("memory_loop.scan_fail", error=str(exc))

    # 2. Registrar decisión vía cuba_decreto
    rationale_parts: list[str] = []
    if hasattr(analysis, "reasoning"):
        rationale_parts.append(str(analysis.reasoning))
    if hasattr(analysis, "key_factor"):
        rationale_parts.append(f"factor: {analysis.key_factor}")
    if hasattr(analysis, "edge_direction"):
        rationale_parts.append(f"edge: {analysis.edge_direction}")
    rationale = " | ".join(rationale_parts) or "sin rationale explícito"

    alternatives: list[dict[str, Any]] = []
    if hasattr(analysis, "warning_flags") and analysis.warning_flags:
        for flag in list(analysis.warning_flags)[:3]:
            alternatives.append({"rejected_because": str(flag)})

    if bet_id is not None:
        try:
            await mcp_memory.record_bet_decision(
                bet_id=bet_id,
                event_id=event_id,
                market=market,
                outcome=outcome,
                p_model=p_model,
                ev=ev,
                kelly_pct=kelly_pct,
                rationale=rationale,
                alternatives=alternatives or None,
            )
            result["recorded"] = True
        except Exception as exc:
            logger.warning("memory_loop.record_decision_failed", error=str(exc))

    # 3. Alarma si warning_flags indican patrón conocido de pérdida
    warnings_attr = getattr(analysis, "warning_flags", None) or []
    for flag in warnings_attr[:3]:
        flag_lower = str(flag).lower()
        if any(kw in flag_lower for kw in ("repetition", "pattern", "overconfidence", "chasing")):
            try:
                await mcp_memory.alarma(
                    trigger=f"llm_warning_{flag_lower[:50]}",
                    details={"event_id": event_id, "market": market, "outcome": outcome},
                )
                result["alarms"].append(flag_lower[:80])
            except Exception as exc:
                logger.debug("memory_loop.alarma_fail", error=str(exc))

    return result


async def record_bet_feedback(
    *,
    bet_id: int,
    result: str,  # "won" | "lost" | "void"
    pnl_units: float,
    clv: float | None = None,
    narrative_tags: list[str] | None = None,
    discrepancy_score: float | None = None,
) -> dict[str, Any]:
    """Feedback al resolver la bet (cuba_eco + cronica + alarma si discrepancy alta).

    Llamado desde `flows/post_mortem.py` tras settlement.
    Cierra el loop: la próxima vez que el LLM analice un evento similar,
    `fetch_memory_context` traerá este outcome al prompt.
    """
    summary: dict[str, Any] = {"eco_ok": False, "alarma_triggered": False}

    try:
        await mcp_memory.record_bet_outcome(
            bet_id=bet_id, result=result, delta_units=pnl_units, clv=clv
        )
        summary["eco_ok"] = True
    except Exception as exc:
        logger.warning("memory_loop.eco_fail", bet_id=bet_id, error=str(exc))

    # Si discrepancy_score muy alto → patrón a vigilar
    if discrepancy_score is not None and discrepancy_score > 0.5:
        try:
            tag_str = "-".join(narrative_tags[:2]) if narrative_tags else "generic"
            await mcp_memory.alarma(
                trigger=f"high_discrepancy_{tag_str}"[:80],
                details={
                    "bet_id": bet_id,
                    "discrepancy_score": discrepancy_score,
                    "pnl_units": pnl_units,
                    "narrative_tags": narrative_tags,
                },
            )
            summary["alarma_triggered"] = True
        except Exception as exc:
            logger.debug("memory_loop.alarma_discrepancy_fail", error=str(exc))

    return summary


async def analyze_with_memory(
    llm_client: Any,
    *,
    task_kind: str,
    system: str,
    user: str,
    schema: type,
    event_description: str,
    teams: list[str],
    market: str | None = None,
    sport_code: str | None = None,
    model_version: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[Any, str]:
    """Helper principal: wrap `structured_chat` con inyección de contexto de memoria.

    Retorna (analysis_struct, memory_context_used) — útil para debugging y
    para persistir junto con la predicción qué memorias se usaron.
    """
    memory_context = await fetch_memory_context(
        event_description=event_description,
        teams=teams,
        market=market,
        sport_code=sport_code,
        model_version=model_version,
    )

    enriched_system = system
    if memory_context:
        enriched_system = f"{system}\n\n{memory_context}"

    analysis = await llm_client.structured_chat(
        task_kind=task_kind,
        system=enriched_system,
        user=user,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return analysis, memory_context
