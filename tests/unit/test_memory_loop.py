"""Tests del loop de retroalimentación LLM ↔ cuba-memorys."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import msgspec

from apuestas.llm.memory_loop import (
    analyze_with_memory,
    fetch_memory_context,
    record_bet_feedback,
    validate_and_record_decision,
)


class MockAnalysis(msgspec.Struct, frozen=True):
    outcome: str
    reasoning: str
    key_factor: str
    edge_direction: str
    warning_flags: list[str]


async def test_fetch_memory_context_empty_when_cuba_down() -> None:
    """Si cuba-memorys no responde, retorna string vacío (degrada gracilmente)."""
    with patch("apuestas.llm.memory_loop.mcp_memory.faro", side_effect=Exception("mcp down")):
        ctx = await fetch_memory_context(
            event_description="Test match",
            teams=["A", "B"],
            market="h2h",
        )
    assert ctx == ""


async def test_fetch_memory_context_includes_memories() -> None:
    """Si cuba_faro devuelve memorias, se inyectan al context."""
    with (
        patch(
            "apuestas.llm.memory_loop.mcp_memory.faro",
            new=AsyncMock(
                return_value={
                    "memories": [
                        {
                            "content": "Past bet on Liga MX favorito home perdió",
                            "date": "2026-03-01",
                        },
                        {"content": "Lesion player X confirmada", "date": "2026-04-15"},
                    ]
                }
            ),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.check_repetition",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.get_calibrated_confidence",
            new=AsyncMock(return_value=None),
        ),
    ):
        ctx = await fetch_memory_context(
            event_description="LigaMX favorito home",
            teams=["América", "Chivas"],
            market="h2h",
        )
    assert "Memorias relevantes" in ctx
    assert "Liga MX favorito" in ctx
    assert "ANTI-ALUCINACIÓN" in ctx


async def test_fetch_memory_context_flags_repetition() -> None:
    """Si check_repetition encuentra pattern con >=3 bets, genera warning en context."""
    with (
        patch("apuestas.llm.memory_loop.mcp_memory.faro", new=AsyncMock(return_value=None)),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.check_repetition",
            new=AsyncMock(return_value={"count": 5, "wins": 1}),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.get_calibrated_confidence",
            new=AsyncMock(return_value=None),
        ),
    ):
        ctx = await fetch_memory_context(
            event_description="Test",
            teams=["A", "B"],
            market="h2h",
            sport_code="soccer",
        )
    assert "Patrón detectado" in ctx
    assert "5 bets" in ctx
    assert "20.0%" in ctx  # 1/5


async def test_fetch_memory_context_includes_calibration() -> None:
    """Calibración bayesiana se inyecta si existe."""
    with (
        patch("apuestas.llm.memory_loop.mcp_memory.faro", new=AsyncMock(return_value=None)),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.check_repetition",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.get_calibrated_confidence",
            new=AsyncMock(return_value={"calibrated_accuracy": 0.63, "n_samples": 40}),
        ),
    ):
        ctx = await fetch_memory_context(
            event_description="Test",
            teams=["A", "B"],
            market="h2h",
            model_version="lgbm_v3",
            confidence_level="high",
        )
    assert "Calibración histórica modelo lgbm_v3" in ctx
    assert "63.0%" in ctx
    assert "n=40" in ctx


async def test_validate_and_record_persists_decision() -> None:
    analysis = MockAnalysis(
        outcome="home",
        reasoning="home team strong form",
        key_factor="rest advantage",
        edge_direction="favors_home",
        warning_flags=["small_sample"],
    )
    with (
        patch(
            "apuestas.llm.memory_loop.mcp_memory.scan_contradictions",
            new=AsyncMock(return_value={"contradictions": []}),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.record_bet_decision",
            new=AsyncMock(return_value={"ok": True}),
        ) as mock_record,
        patch("apuestas.llm.memory_loop.mcp_memory.alarma", new=AsyncMock(return_value=None)),
    ):
        result = await validate_and_record_decision(
            analysis=analysis,
            bet_id=42,
            event_id=100,
            market="h2h",
            outcome="home",
            p_model=0.58,
            ev=0.06,
            kelly_pct=0.02,
            sport_code="soccer",
        )

    assert result["recorded"] is True
    mock_record.assert_called_once()
    call_kwargs = mock_record.call_args.kwargs
    assert call_kwargs["bet_id"] == 42
    assert "home team strong form" in call_kwargs["rationale"]
    assert call_kwargs["alternatives"] == [{"rejected_because": "small_sample"}]


async def test_validate_triggers_alarma_on_repetition_flag() -> None:
    analysis = MockAnalysis(
        outcome="home",
        reasoning="test",
        key_factor="test",
        edge_direction="neutral",
        warning_flags=["repetition_3_consecutive_losses"],
    )
    with (
        patch(
            "apuestas.llm.memory_loop.mcp_memory.scan_contradictions",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.record_bet_decision",
            new=AsyncMock(return_value={"ok": True}),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.alarma",
            new=AsyncMock(return_value={"ok": True}),
        ) as mock_alarma,
    ):
        result = await validate_and_record_decision(
            analysis=analysis,
            bet_id=1,
            event_id=1,
            market="h2h",
            outcome="home",
            p_model=0.5,
            ev=0.03,
            kelly_pct=0.01,
        )

    mock_alarma.assert_called_once()
    assert len(result["alarms"]) == 1


async def test_record_bet_feedback_triggers_alarma_on_high_discrepancy() -> None:
    with (
        patch(
            "apuestas.llm.memory_loop.mcp_memory.record_bet_outcome",
            new=AsyncMock(return_value={"ok": True}),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.alarma",
            new=AsyncMock(return_value={"ok": True}),
        ) as mock_alarma,
    ):
        summary = await record_bet_feedback(
            bet_id=10,
            result="lost",
            pnl_units=-1.0,
            clv=-0.02,
            narrative_tags=["high_altitude_away", "b2b"],
            discrepancy_score=0.75,
        )

    assert summary["eco_ok"] is True
    assert summary["alarma_triggered"] is True
    mock_alarma.assert_called_once()


async def test_record_bet_feedback_no_alarma_low_discrepancy() -> None:
    with (
        patch(
            "apuestas.llm.memory_loop.mcp_memory.record_bet_outcome",
            new=AsyncMock(return_value={"ok": True}),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.alarma",
            new=AsyncMock(return_value=None),
        ) as mock_alarma,
    ):
        summary = await record_bet_feedback(
            bet_id=11,
            result="won",
            pnl_units=1.5,
            discrepancy_score=0.15,
        )

    assert summary["eco_ok"] is True
    assert summary["alarma_triggered"] is False
    mock_alarma.assert_not_called()


async def test_analyze_with_memory_injects_context_into_system() -> None:
    """El wrapper principal concatena memoria al system prompt antes de llamar LLM."""
    captured_systems: list[str] = []

    class FakeClient:
        async def structured_chat(self, **kwargs: Any) -> Any:
            captured_systems.append(kwargs["system"])
            return MockAnalysis(
                outcome="home",
                reasoning="test",
                key_factor="test",
                edge_direction="neutral",
                warning_flags=[],
            )

    with (
        patch(
            "apuestas.llm.memory_loop.mcp_memory.faro",
            new=AsyncMock(
                return_value={
                    "memories": [{"content": "memoria X relevante", "date": "2026-04-10"}]
                }
            ),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.check_repetition",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "apuestas.llm.memory_loop.mcp_memory.get_calibrated_confidence",
            new=AsyncMock(return_value=None),
        ),
    ):
        analysis, ctx = await analyze_with_memory(
            FakeClient(),
            task_kind="test",
            system="Original system prompt",
            user="user q",
            schema=MockAnalysis,
            event_description="EPL test match",
            teams=["Team A", "Team B"],
            market="h2h",
        )

    assert len(captured_systems) == 1
    assert "Original system prompt" in captured_systems[0]
    assert "CONTEXTO DE MEMORIA LARGA" in captured_systems[0]
    assert "memoria X relevante" in captured_systems[0]
    assert "ANTI-ALUCINACIÓN" in captured_systems[0]
    assert analysis.outcome == "home"
    assert "memoria X relevante" in ctx
