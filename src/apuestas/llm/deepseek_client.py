"""Cliente DeepSeek API (OpenAI-compatible) con misma interfaz que LlamaClient.

Drop-in replacement cuando `LLM_BACKEND=deepseek` en `.env`. Usa `deepseek-chat`
(V3.2 general, $0.27/$1.10 por 1M tok input/output) o `deepseek-reasoner` (R1
para razonamiento, más caro pero mejor en analíticos).

DeepSeek no soporta el campo `grammar` de llama.cpp; usamos `response_format=
{"type":"json_object"}` + validación msgspec con retry corrector.

Usage:
    async with DeepSeekClient() as llm:
        analysis = await llm.structured_chat(
            task_kind="pre_match",
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=PreMatchAnalysis,
            grammar_name=None,  # DeepSeek ignora grammar
        )
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
import msgspec
import stamina
from sqlalchemy import text

from apuestas.config import get_settings
from apuestas.db import session_scope
from apuestas.llm.client import (
    ChatMessage,
    LLMError,
    LLMGuardrailError,
)
from apuestas.obs.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

T = TypeVar("T", bound=msgspec.Struct)


class DeepSeekClient:
    """Cliente async para DeepSeek API. Drop-in con LlamaClient."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.llm.deepseek_base_url).rstrip("/")
        self.model = model or settings.llm.deepseek_model
        self.default_temperature = settings.llm.deepseek_temperature
        self.default_max_tokens = settings.llm.deepseek_max_tokens

        key = api_key or (
            settings.llm.deepseek_api_key.get_secret_value()
            if settings.llm.deepseek_api_key
            else None
        )
        if not key:
            msg = "DEEPSEEK_API_KEY requerida cuando LLM_BACKEND=deepseek"
            raise LLMError(msg)
        self._api_key = key
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout

    async def __aenter__(self) -> DeepSeekClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "DeepSeekClient usado fuera de async context"
            raise RuntimeError(msg)
        return self._client

    async def health(self) -> bool:
        """DeepSeek no expone /health; pruebo con /v1/models."""
        try:
            resp = await self.client.get("/v1/models", timeout=5.0)
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
        json_mode: bool,
    ) -> tuple[str, dict[str, int]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = await self.client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        # DeepSeek expone cache hit/miss en usage:
        #   prompt_cache_hit_tokens  : input tokens facturados a $0.07/M (-74%)
        #   prompt_cache_miss_tokens : input tokens facturados a $0.27/M (full)
        # Un sistema con system prompt estable + alto volumen mismo task_kind
        # debería ver hit_ratio >50% tras unos minutos de calentamiento.
        return content, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
            "cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens", 0)),
        }

    async def structured_chat(
        self,
        *,
        task_kind: str,
        system: str,
        user: str,
        schema: type[T],
        grammar_name: str | None = None,  # ignorado en DeepSeek
        temperature: float | None = None,
        max_tokens: int | None = None,
        prompt_version: str | None = None,  # metadata, no afecta request
        **_extra: Any,  # tolera kwargs extra del router sin romper
    ) -> T:
        """Chat con response_format JSON + validación msgspec + retry corrector."""
        _ = grammar_name  # no aplica en DeepSeek API
        _ = prompt_version  # se usa en tracking, no en el payload
        call_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                content, usage = await self._post_chat(
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    json_mode=True,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "deepseek.http_failed",
                    task_kind=task_kind,
                    attempt=attempt,
                    error=str(exc),
                )
                continue

            try:
                normalized = _coerce_common_mistakes(content)
                validated = msgspec.json.decode(normalized.encode("utf-8"), type=schema)
            except msgspec.ValidationError as exc:
                last_error = exc
                logger.warning(
                    "deepseek.schema_invalid",
                    task_kind=task_kind,
                    attempt=attempt,
                    error=str(exc),
                )
                # Retry con corrección explícita
                messages.append(ChatMessage(role="assistant", content=content))
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"Your previous response did not match the required schema. "
                            f"Error: {exc}. Return ONLY valid JSON matching the schema."
                        ),
                    )
                )
                continue
            else:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                await self._log_call(
                    call_id=call_id,
                    task_kind=task_kind,
                    usage=usage,
                    latency_ms=elapsed_ms,
                )
                cache_hit = usage.get("cache_hit_tokens", 0)
                cache_miss = usage.get("cache_miss_tokens", 0)
                cache_ratio = (
                    cache_hit / (cache_hit + cache_miss) if (cache_hit + cache_miss) > 0 else 0.0
                )
                logger.info(
                    "deepseek.structured_ok",
                    task_kind=task_kind,
                    model=self.model,
                    latency_ms=elapsed_ms,
                    tokens=usage.get("total_tokens", 0),
                    cache_hit_ratio=round(cache_ratio, 3),
                )
                return validated

        msg = f"DeepSeek failed after 3 attempts: {last_error}"
        raise LLMGuardrailError(msg) from last_error

    async def _log_call(
        self,
        *,
        call_id: str,
        task_kind: str,
        usage: dict[str, int],
        latency_ms: int,
    ) -> None:
        """Persiste llamada en llm_calls para tracking de costos.

        Costos DeepSeek V3.2 (abril 2026):
          - input cache MISS: $0.27/M
          - input cache HIT : $0.07/M  (caching automático en system idéntico)
          - output         : $1.10/M
        """
        cache_hit = usage.get("cache_hit_tokens", 0)
        cache_miss = usage.get("cache_miss_tokens", 0)
        # Si DeepSeek devolvió tracking de cache, factura por buckets exactos.
        # Si no (modelos viejos o respuesta sin metadata), usar prompt_tokens
        # full price como fallback conservador.
        if cache_hit + cache_miss > 0:
            cost_in = cache_hit * 0.07e-6 + cache_miss * 0.27e-6
        else:
            cost_in = usage.get("prompt_tokens", 0) * 0.27e-6
        cost_out = usage.get("completion_tokens", 0) * 1.10e-6
        try:
            async with session_scope() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO llm_calls
                            (task_kind, model, tokens_in, tokens_out,
                             latency_ms, cost_usd, success, correlation_id)
                        VALUES
                            (:tk, :m, :ti, :to, :lt, :cu, true, :cid)
                        """
                    ),
                    {
                        "tk": task_kind,
                        "m": f"deepseek:{self.model}",
                        "ti": usage.get("prompt_tokens", 0),
                        "to": usage.get("completion_tokens", 0),
                        "lt": latency_ms,
                        "cu": round(cost_in + cost_out, 6),
                        "cid": call_id,
                    },
                )
        except Exception as exc:
            # Logging del costo es best-effort; no romper el pipeline si falla
            logger.debug("deepseek.log_call_failed", error=str(exc))


def _coerce_common_mistakes(content: str) -> str:
    """Normaliza errores recurrentes del LLM antes del schema strict.

    - sentiment: number → mapea a positive/neutral/negative según signo.
    - sentiment_score: string "0.5" → float.
    - teams/suspensions/transfers: list[dict{name}] → list[str] (DeepSeek hábito).
    - persons[].role: 'team'/'coach_staff' → 'other' (valores fuera del enum).
    """
    import json

    try:
        data = json.loads(content)
    except Exception:
        return content
    if not isinstance(data, dict):
        return content

    sent = data.get("sentiment")
    if isinstance(sent, int | float) and not isinstance(sent, bool):
        if sent > 0.15:
            data["sentiment"] = "positive"
        elif sent < -0.15:
            data["sentiment"] = "negative"
        else:
            data["sentiment"] = "neutral"
        data.setdefault("sentiment_score", float(sent))

    score = data.get("sentiment_score")
    if isinstance(score, str):
        try:
            data["sentiment_score"] = float(score)
        except ValueError:
            data["sentiment_score"] = 0.0

    # Coerce list[str] fields: DeepSeek frecuentemente devuelve [{"name": "X"}]
    for key in ("teams", "suspensions", "transfers"):
        val = data.get(key)
        if isinstance(val, list):
            coerced: list[str] = []
            for item in val:
                if isinstance(item, str):
                    coerced.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("team") or item.get("player")
                    if isinstance(name, str) and name:
                        coerced.append(name)
            data[key] = coerced

    # persons[].role fuera del enum → 'other'
    allowed_roles = {"player", "coach", "referee", "executive", "other"}
    persons = data.get("persons")
    if isinstance(persons, list):
        for p in persons:
            if isinstance(p, dict) and p.get("role") not in allowed_roles:
                p["role"] = "other"

    # injuries[]: coerce null → "" en player/team/impact + enum mapping severity
    allowed_severity = {"out", "doubtful", "questionable", "probable", "active"}
    injuries = data.get("injuries")
    if isinstance(injuries, list):
        for inj in injuries:
            if not isinstance(inj, dict):
                continue
            for k in ("player", "team", "impact"):
                if inj.get(k) is None:
                    inj[k] = ""
            sev = inj.get("severity")
            if sev is None or sev not in allowed_severity:
                inj["severity"] = "questionable"
            # Además coerce dict→str en player/team (DeepSeek a veces)
            for k in ("player", "team", "impact"):
                v = inj.get(k)
                if isinstance(v, dict):
                    inj[k] = v.get("name") or v.get("value") or ""

    # team_analysis.team_name omitido por LLM (schema_invalid recurrente).
    # Cinturón + tirantes:
    #   1. Si home_/away_team_analysis es dict y le falta team_name → ""
    #   2. Si es None / faltante completo → crear estructura mínima vacía
    #   3. Si es string (LLM colapsó la estructura) → wrap en dict mínimo
    # El detector usa team_name solo para logging, no para decisiones, así
    # que placeholders evitan el retry cycle (que cuesta tokens del quota).
    # Estructura mínima alineada con `apuestas.schemas.llm.TeamAnalysis`
    # (msgspec). Solo `team_name` es required; el resto tienen defaults pero
    # los completamos explícitos para evitar surprises si el schema cambia.
    _MIN_TA: dict = {
        "team_name": "",
        "narrative_momentum": "neutral",
        "rest_days": 0,
        "back_to_back": False,
        "key_injuries": [],
        "lineup_changes": [],
        "recent_transfers_impact": [],
        "coaching_change_flags": [],
        "streak_home_away": None,
        "streak_overall": None,
        "player_streaks_notable": [],
    }
    for side in ("home_team_analysis", "away_team_analysis"):
        ta = data.get(side)
        if not isinstance(ta, dict):
            data[side] = dict(_MIN_TA)
            continue
        for k, v in _MIN_TA.items():
            if k not in ta:
                ta[k] = v if not isinstance(v, list) else []

    return json.dumps(data, ensure_ascii=False)


def get_llm_client() -> Any:
    """Factory: retorna el cliente correcto según LLM_BACKEND.

    Permite al resto del código ser agnóstico al backend:
        async with get_llm_client() as llm:
            ...
    """
    from apuestas.llm.client import LlamaClient

    backend = get_settings().llm.llm_backend.lower()
    if backend == "deepseek":
        return DeepSeekClient()
    return LlamaClient()
