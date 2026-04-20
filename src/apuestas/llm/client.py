"""Cliente llama.cpp (OpenAI-compatible) con retry, tracking y guardrails.

Usage:
    async with LlamaClient() as llm:
        analysis = await llm.structured_chat(
            task_kind="pre_match",
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=PreMatchAnalysis,
            grammar_name="pre_match_analysis",
        )
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
import msgspec
import stamina
from sqlalchemy import text

from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.llm.grammars import get_grammar
from apuestas.obs.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

T = TypeVar("T", bound=msgspec.Struct)


class LLMError(Exception):
    """Error genérico del LLM."""


class LLMGuardrailError(LLMError):
    """El output del LLM falló validación msgspec tras retry."""


@dataclass(slots=True)
class ChatMessage:
    role: str  # system|user|assistant
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LlamaClient:
    """Cliente async para llama.cpp server. API compatible OpenAI v1."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.llm.llama_server_url).rstrip("/")
        self.model = model or settings.llm.llama_model
        self.ctx_size = settings.llm.llama_ctx_size
        self.default_temperature = settings.llm.llama_temperature
        self.default_max_tokens = settings.llm.llama_max_tokens
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def __aenter__(self) -> LlamaClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "LlamaClient usado fuera de async context"
            raise RuntimeError(msg)
        return self._client

    async def health(self) -> bool:
        """Verifica si el server responde."""
        try:
            resp = await self.client.get("/health", timeout=5.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    @stamina.retry(
        on=(httpx.HTTPError, httpx.ReadTimeout),
        attempts=3,
        wait_initial=0.5,
        wait_max=5.0,
        wait_jitter=0.5,
    )
    async def _post_chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int,
        grammar: str | None,
    ) -> tuple[str, dict[str, int]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if grammar is not None:
            payload["grammar"] = grammar

        resp = await self.client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

    async def chat(
        self,
        *,
        task_kind: str,
        messages: Sequence[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        grammar_name: str | None = None,
        correlation_id: str | None = None,
        prompt_version: str | None = None,
    ) -> str:
        """Llamada chat raw. Devuelve contenido texto. Registra en llm_calls."""
        correlation_id = correlation_id or uuid.uuid4().hex
        grammar = get_grammar(grammar_name) if grammar_name else None
        t0 = time.perf_counter()
        success = False
        tokens_in = 0
        tokens_out = 0

        try:
            content, usage = await self._post_chat(
                messages=messages,
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                grammar=grammar,
            )
            tokens_in = usage["prompt_tokens"]
            tokens_out = usage["completion_tokens"]
            success = True
            return content
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            await self._persist_call(
                task_kind=task_kind,
                prompt_version=prompt_version,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                success=success,
                correlation_id=correlation_id,
            )
            logger.info(
                "llm.call",
                task_kind=task_kind,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                success=success,
                correlation_id=correlation_id,
            )

    async def structured_chat(
        self,
        *,
        task_kind: str,
        system: str,
        user: str,
        schema: type[T],
        grammar_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        prompt_version: str | None = None,
    ) -> T:
        """Chat con validación msgspec estricta. Retry una vez con prompt correctivo."""
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]

        raw = await self.chat(
            task_kind=task_kind,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            grammar_name=grammar_name,
            prompt_version=prompt_version,
        )
        try:
            return msgspec.json.decode(raw.encode(), type=schema, strict=True)
        except msgspec.ValidationError as exc:
            logger.warning("llm.structured.validation_fail", error=str(exc), retry=True)
            # Retry con prompt correctivo
            messages.append(ChatMessage(role="assistant", content=raw))
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Tu respuesta falló validación JSON. Error: "
                        f"{exc}. Responde SOLO con JSON válido según el schema requerido."
                    ),
                )
            )
            raw2 = await self.chat(
                task_kind=f"{task_kind}_retry",
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                grammar_name=grammar_name,
                prompt_version=prompt_version,
            )
            try:
                return msgspec.json.decode(raw2.encode(), type=schema, strict=True)
            except msgspec.ValidationError as exc2:
                msg = f"LLM output failed validation after retry: {exc2}"
                raise LLMGuardrailError(msg) from exc2

    @staticmethod
    async def _persist_call(
        *,
        task_kind: str,
        prompt_version: str | None,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        success: bool,
        correlation_id: str,
    ) -> None:
        """Inserta row en llm_calls. Fail-safe: no rompe el flow si DB caída."""
        try:
            async with session_scope() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO llm_calls
                          (task_kind, model, prompt_version, tokens_in, tokens_out,
                           latency_ms, cost_usd, success, correlation_id)
                        VALUES
                          (:task_kind, :model, :prompt_version, :tokens_in, :tokens_out,
                           :latency_ms, 0, :success, :correlation_id)
                        """
                    ),
                    {
                        "task_kind": task_kind,
                        "model": get_settings().llm.llama_model,
                        "prompt_version": prompt_version,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "latency_ms": latency_ms,
                        "success": success,
                        "correlation_id": correlation_id,
                    },
                )
        except Exception as exc:
            logger.debug("llm.persist_call.failed", error=str(exc))
