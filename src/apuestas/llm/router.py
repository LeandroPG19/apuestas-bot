"""LLM router — un solo entry point por task_kind.

Combina: carga de prompt → cliente llama.cpp → schema msgspec → guardrails.
"""

from __future__ import annotations

import importlib
from typing import Any, TypeVar

import msgspec

from apuestas.llm.client import LlamaClient, LLMGuardrailError
from apuestas.llm.prompts import PromptTemplate, load_prompt
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=msgspec.Struct)


def _resolve_schema(dotted: str) -> type[msgspec.Struct]:
    """Resuelve 'apuestas.schemas.llm.PreMatchAnalysis' → clase."""
    module_path, _, cls_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    if not issubclass(cls, msgspec.Struct):
        msg = f"{dotted} no es msgspec.Struct"
        raise TypeError(msg)
    return cls


async def run_task(
    *,
    task_kind: str,
    version: str = "v1",
    client: LlamaClient,
    render_vars: dict[str, Any] | None = None,
) -> msgspec.Struct:
    """Ejecuta una task LLM completa: carga prompt, renderiza, valida, devuelve struct.

    Args:
        task_kind: nombre del prompt YAML (ej. 'pre_match', 'post_mortem', 'nlp/ner').
        version: versión del prompt (default 'v1').
        client: instancia activa de LlamaClient.
        render_vars: variables para formatear user_template.

    Returns:
        Instancia msgspec.Struct del tipo declarado en el YAML del prompt.
    """
    prompt = load_prompt(task_kind, version)

    if prompt.schema is None:
        msg = f"Prompt {prompt.full_id} no declara 'schema'; usa client.chat() directo"
        raise ValueError(msg)

    schema_cls = _resolve_schema(prompt.schema)
    rendered_user = prompt.render(**(render_vars or {}))

    logger.debug(
        "llm.router.task",
        task_kind=task_kind,
        version=version,
        prompt_id=prompt.full_id,
        schema=prompt.schema,
        grammar=prompt.grammar,
    )

    try:
        result = await client.structured_chat(
            task_kind=task_kind,
            system=prompt.system,
            user=rendered_user,
            schema=schema_cls,
            grammar_name=prompt.grammar,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
            prompt_version=prompt.version,
        )
        return result  # type: ignore[no-any-return]
    except LLMGuardrailError:
        logger.exception("llm.router.guardrail_failed", task_kind=task_kind)
        raise


async def run_chat(
    *,
    task_kind: str,
    version: str,
    client: LlamaClient,
    render_vars: dict[str, Any] | None = None,
) -> tuple[str, PromptTemplate]:
    """Variante raw: devuelve texto sin validación de schema.

    Útil para summarización, traducción, query expansion.
    """
    prompt = load_prompt(task_kind, version)
    rendered_user = prompt.render(**(render_vars or {}))

    from apuestas.llm.client import ChatMessage

    messages = [
        ChatMessage(role="system", content=prompt.system),
        ChatMessage(role="user", content=rendered_user),
    ]
    content = await client.chat(
        task_kind=task_kind,
        messages=messages,
        temperature=prompt.temperature,
        max_tokens=prompt.max_tokens,
        grammar_name=prompt.grammar,
        prompt_version=prompt.version,
    )
    return content, prompt
