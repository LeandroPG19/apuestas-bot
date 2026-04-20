"""Tests del cliente llama.cpp con HTTP mockeado."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import msgspec
import pytest

from apuestas.llm.client import ChatMessage, LlamaClient, LLMGuardrailError
from apuestas.schemas.llm import NERExtraction


def _build_openai_response(
    content: str, tokens_in: int = 10, tokens_out: int = 20
) -> dict[str, Any]:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


@pytest.mark.asyncio
async def test_chat_raw_returns_content() -> None:
    expected_content = '{"persons":[],"teams":[],"injuries":[],"suspensions":[],"transfers":[],"sentiment":"neutral","sentiment_score":0.0}'
    mock_response = httpx.Response(
        status_code=200,
        json=_build_openai_response(expected_content),
        request=httpx.Request("POST", "http://llm:8080/v1/chat/completions"),
    )

    async with LlamaClient(base_url="http://llm:8080") as client:
        with (
            patch.object(client.client, "post", AsyncMock(return_value=mock_response)),
            patch.object(LlamaClient, "_persist_call", AsyncMock()),
        ):
            result = await client.chat(
                task_kind="test",
                messages=[ChatMessage(role="user", content="hola")],
            )
    assert result == expected_content


@pytest.mark.asyncio
async def test_structured_chat_valid_json() -> None:
    valid_json = json.dumps(
        {
            "persons": [{"name": "LeBron James", "role": "player", "team": "Lakers"}],
            "teams": ["Lakers"],
            "injuries": [],
            "suspensions": [],
            "transfers": [],
            "sentiment": "positive",
            "sentiment_score": 0.5,
        }
    )
    mock_response = httpx.Response(
        status_code=200,
        json=_build_openai_response(valid_json),
        request=httpx.Request("POST", "http://llm:8080/v1/chat/completions"),
    )

    async with LlamaClient(base_url="http://llm:8080") as client:
        with (
            patch.object(client.client, "post", AsyncMock(return_value=mock_response)),
            patch.object(LlamaClient, "_persist_call", AsyncMock()),
        ):
            result = await client.structured_chat(
                task_kind="ner_test",
                system="extrae",
                user="LeBron juega hoy",
                schema=NERExtraction,
            )
    assert isinstance(result, NERExtraction)
    assert result.persons[0].name == "LeBron James"
    assert result.sentiment == "positive"


@pytest.mark.asyncio
async def test_structured_chat_invalid_then_retry_fails() -> None:
    bad_json = '{"persons":[],"sentiment":"muy_positivo"}'  # severity inválida + falta campos
    mock_response = httpx.Response(
        status_code=200,
        json=_build_openai_response(bad_json),
        request=httpx.Request("POST", "http://llm:8080/v1/chat/completions"),
    )

    async with LlamaClient(base_url="http://llm:8080") as client:
        with (
            patch.object(client.client, "post", AsyncMock(return_value=mock_response)),
            patch.object(LlamaClient, "_persist_call", AsyncMock()),
        ):
            with pytest.raises(LLMGuardrailError):
                await client.structured_chat(
                    task_kind="ner_test",
                    system="extrae",
                    user="texto",
                    schema=NERExtraction,
                )


@pytest.mark.asyncio
async def test_structured_chat_retry_recovers() -> None:
    bad_json = '{"not":"compliant"}'
    good_json = json.dumps(
        {
            "persons": [],
            "teams": [],
            "injuries": [],
            "suspensions": [],
            "transfers": [],
            "sentiment": "neutral",
            "sentiment_score": 0.0,
        }
    )
    responses = [
        httpx.Response(
            200,
            json=_build_openai_response(bad_json),
            request=httpx.Request("POST", "http://llm:8080/v1/chat/completions"),
        ),
        httpx.Response(
            200,
            json=_build_openai_response(good_json),
            request=httpx.Request("POST", "http://llm:8080/v1/chat/completions"),
        ),
    ]
    mock_post = AsyncMock(side_effect=responses)

    async with LlamaClient(base_url="http://llm:8080") as client:
        with (
            patch.object(client.client, "post", mock_post),
            patch.object(LlamaClient, "_persist_call", AsyncMock()),
        ):
            result = await client.structured_chat(
                task_kind="ner_test",
                system="extrae",
                user="texto",
                schema=NERExtraction,
            )
    assert isinstance(result, NERExtraction)
    assert mock_post.await_count == 2


@pytest.mark.asyncio
async def test_health_check_ok() -> None:
    mock_response = httpx.Response(
        status_code=200,
        json={"status": "ok"},
        request=httpx.Request("GET", "http://llm:8080/health"),
    )
    async with LlamaClient(base_url="http://llm:8080") as client:
        with patch.object(client.client, "get", AsyncMock(return_value=mock_response)):
            assert await client.health() is True


@pytest.mark.asyncio
async def test_health_check_failure() -> None:
    async with LlamaClient(base_url="http://llm:8080") as client:
        with patch.object(
            client.client,
            "get",
            AsyncMock(
                side_effect=httpx.ConnectError(
                    "down", request=httpx.Request("GET", "http://llm:8080/health")
                )
            ),
        ):
            assert await client.health() is False


def test_msgspec_parses_enum_validation() -> None:
    """Sanity check: msgspec rechaza enums inválidos."""
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            b'{"persons":[],"teams":[],"injuries":[],"suspensions":[],"transfers":[],"sentiment":"invalid","sentiment_score":0}',
            type=NERExtraction,
        )
