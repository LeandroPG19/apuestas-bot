"""Helpers de alto nivel para cuba-memorys (§8.5).

Mapea los 19 tools de cuba-memorys a operaciones del bot:
- record_bet_decision / record_bet_outcome
- query_past_patterns / check_repetition
- get_calibrated_confidence
- scan_news_contradictions
- session lifecycle (jornada start/end)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from apuestas.mcp.client import MCPClient
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def jornada_start() -> dict[str, Any] | None:
    """Arranca sesión de memoria. Llamar al start de `make up`."""
    client = MCPClient.get()
    return await client.call("memorys", "cuba_jornada", {"action": "start"})


async def jornada_end() -> dict[str, Any] | None:
    """Cierra sesión (computa diff). Llamar en graceful shutdown."""
    client = MCPClient.get()
    return await client.call("memorys", "cuba_jornada", {"action": "end"})


async def faro(
    query: str,
    *,
    before: str | None = None,
    after: str | None = None,
    tags: list[str] | None = None,
    fmt: str = "compact",
) -> dict[str, Any] | None:
    """Búsqueda semántica sobre memoria. Format compact ahorra ~35% tokens."""
    args: dict[str, Any] = {"query": query, "format": fmt}
    if before:
        args["before"] = before
    if after:
        args["after"] = after
    if tags:
        args["tags"] = tags
    client = MCPClient.get()
    return await client.call("memorys", "cuba_faro", args)


async def record_bet_decision(
    *,
    bet_id: int,
    event_id: int,
    market: str,
    outcome: str,
    p_model: float,
    ev: float,
    kelly_pct: float,
    rationale: str,
    alternatives: list[dict[str, Any]] | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any] | None:
    """Registra decisión como cuba_decreto.

    Signature cuba-memorys v0.6/v0.7: action + chosen + rationale + alternatives.
    `chosen` es la decisión tomada; `alternatives` la lista descartada.
    """
    client = MCPClient.get()
    chosen = (
        f"bet #{bet_id} · {market}:{outcome} · p={p_model:.3f} · "
        f"ev={ev:+.3%} · kelly={kelly_pct:.2%} · event {event_id}"
    )
    alts_strs: list[str] = []
    if alternatives:
        for alt in alternatives[:5]:
            if isinstance(alt, dict):
                alts_strs.append(str(alt.get("rejected_because") or alt)[:120])
            else:
                alts_strs.append(str(alt)[:120])
    args: dict[str, Any] = {
        "action": "record",
        "title": f"bet_{bet_id}_{market}_{outcome}"[:80],
        "chosen": chosen,
        "rationale": rationale[:500],
        "context": (
            f"ts={datetime.now(tz=UTC).isoformat()} · correlation_id={correlation_id or '-'}"
        ),
    }
    if alts_strs:
        args["alternatives"] = alts_strs
    return await client.call("memorys", "cuba_decreto", args)


async def record_bet_outcome(
    *,
    bet_id: int,
    result: str,
    delta_units: float,
    clv: float | None = None,
) -> dict[str, Any] | None:
    """Feedback post-match: cuba_eco con corrección Oja-like.

    Signature v0.6/v0.7: action + correction/entity_name/observation_id.
    Usamos action=positive/negative/correct según el resultado de la bet.
    """
    client = MCPClient.get()
    action_map = {
        "won": "positive",
        "halfwon": "positive",
        "lost": "negative",
        "halflost": "negative",
        "void": "correct",
        "cashed": "correct",
    }
    action = action_map.get(result, "correct")
    correction = f"bet #{bet_id} result={result} delta={delta_units:+.3f}u" + (
        f" clv={clv:+.4f}" if clv is not None else ""
    )
    return await client.call(
        "memorys",
        "cuba_eco",
        {
            "action": action,
            "entity_name": f"bet_{bet_id}",
            "correction": correction[:400],
        },
    )


async def record_post_mortem_episode(
    *,
    bet_id: int,
    event_id: int,
    narrative: dict[str, Any],
    actors: list[str],
) -> dict[str, Any] | None:
    """Timeline entry tras settlement.

    Signature v0.6/v0.7: cuba_cronica action=episode_add + entity_name + content.
    """
    client = MCPClient.get()
    content = str(narrative.get("summary_es") or narrative.get("transferable_lesson") or narrative)[
        :800
    ]
    return await client.call(
        "memorys",
        "cuba_cronica",
        {
            "action": "episode_add",
            "entity_name": f"bet_{bet_id}",
            "observation_type": "bet_settled",
            "content": content,
            "actors": actors[:6],
            "source": "apuestas-bot",
        },
    )


async def check_repetition(
    *,
    pattern_tag: str,
    sport_code: str | None = None,
    days: int = 30,
) -> dict[str, Any] | None:
    """§21.6: anti-repetición — cuba_expediente busca errores resueltos
    relacionados al pattern. Signature v0.6/v0.7: query + resolved_only + project.
    """
    client = MCPClient.get()
    project_tag = sport_code or "apuestas"
    return await client.call(
        "memorys",
        "cuba_expediente",
        {
            "query": pattern_tag[:200],
            "project": project_tag,
            "resolved_only": False,
            "proposed_action": f"bet_pattern_{pattern_tag[:50]}",
        },
    )


async def get_calibrated_confidence(
    *,
    model_name: str,
    confidence_level: str,
) -> dict[str, Any] | None:
    """§17.6: P(correcto | confianza_modelo) histórico bayesiano.

    Signature v0.6/v0.7: cuba_calibrar action=stats + outcome + limit.
    Mapeamos confidence_level → outcome string.
    """
    client = MCPClient.get()
    # Algunos helpers aliasan: model_version → model_name
    _ = model_name
    return await client.call(
        "memorys",
        "cuba_calibrar",
        {
            "action": "stats",
            "outcome": f"confidence_{confidence_level}",
            "limit": 100,
        },
    )


# Alias compatibilidad: el memory_loop.py llama con kwarg model_version
async def get_calibrated_confidence_v(
    *, model_version: str, confidence_level: str
) -> dict[str, Any] | None:
    return await get_calibrated_confidence(
        model_name=model_version, confidence_level=confidence_level
    )


async def scan_contradictions(*, date: str | None = None) -> dict[str, Any] | None:
    """§16.2 paso 3: detecta info contradictoria entre fuentes (news cross)."""
    client = MCPClient.get()
    args: dict[str, Any] = {"action": "scan"}
    if date:
        args["date"] = date
    return await client.call("memorys", "cuba_contradiccion", args)


async def explain_drawdown(*, effect: str) -> dict[str, Any] | None:
    """§17.12: hipótesis causal backward ante bad runs."""
    client = MCPClient.get()
    return await client.call("memorys", "cuba_hipotesis", {"action": "explain", "effect": effect})


async def analyze_gaps() -> dict[str, Any] | None:
    """§21.4 audit semanal: detecta entidades aisladas, hubs underconnected."""
    client = MCPClient.get()
    return await client.call("memorys", "cuba_reflexion", {"action": "analyze"})


async def weekly_decay() -> dict[str, Any] | None:
    """§17.6 mantenimiento: facts 30d, errors 14d, context 7d decay."""
    client = MCPClient.get()
    return await client.call("memorys", "cuba_zafra", {"action": "decay"})


async def health_check() -> dict[str, Any] | None:
    """Vigia: null embeddings, table sizes, triggers, entropy."""
    client = MCPClient.get()
    return await client.call("memorys", "cuba_vigia", {"action": "health"})


async def alarma(*, trigger: str, details: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """§21.6: alerta por patrón detectado.

    Signature v0.6/v0.7: cuba_alarma requiere error_type + error_message.
    error_type es la categoría ("pattern_loss_streak", "data_drift", ...).
    """
    client = MCPClient.get()
    # Inferir error_type del prefix del trigger si no vino explícito
    error_type = "betting_pattern"
    if details and isinstance(details, dict):
        error_type = str(details.get("error_type", error_type))
    ctx_parts: list[str] = []
    if details:
        ctx_parts.extend(f"{k}={v!s:.60}" for k, v in list(details.items())[:8])
    context_str = " · ".join(ctx_parts) if ctx_parts else ""
    return await client.call(
        "memorys",
        "cuba_alarma",
        {
            "error_type": error_type,
            "error_message": trigger[:400],
            "project": "apuestas",
            "context": context_str[:400],
        },
    )


async def remedio(
    *, error_id: int | str | None = None, solution: str, issue: str | None = None
) -> dict[str, Any] | None:
    """Guardar lesson-learned post-drawdown.

    Signature v0.6/v0.7: cuba_remedio requiere error_id + solution.
    `issue` es alias legacy — si viene solo, se usa como hash para error_id
    sintético (el server v0.7 acepta string como error_id).
    """
    client = MCPClient.get()
    eid: Any = error_id if error_id is not None else (issue or "unknown")
    return await client.call(
        "memorys",
        "cuba_remedio",
        {"error_id": eid, "solution": solution[:800]},
    )
