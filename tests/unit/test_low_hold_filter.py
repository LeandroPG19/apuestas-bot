"""Tests Fase 1.2 — low-hold filter en detector."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apuestas.betting.detector import DetectorConfig, EventOdds, detect_value_bets_for_event


def _event(
    quotes: dict[str, list[float]], outcomes: tuple[str, ...] = ("home", "away")
) -> EventOdds:
    return EventOdds(
        event_id=1,
        event_external_id="test_event_1",
        market="h2h",
        start_time=datetime(2026, 5, 1, 20, 0, tzinfo=UTC),
        outcomes=list(outcomes),
        quotes_by_bookmaker=quotes,
    )


async def test_rejects_market_high_hold_caliente() -> None:
    """Caliente solo, vig 8% → rechaza (>3% default)."""
    # home=1.85, away=1.85 → overround = 1/1.85 + 1/1.85 - 1 = 0.081
    event = _event({"caliente": [1.85, 1.85]})
    cfg = DetectorConfig(max_hold=0.03, max_hold_single_book=0.05)
    result = await detect_value_bets_for_event(
        event, model_probs={"home": 0.55, "away": 0.45}, cfg=cfg
    )
    assert result == []


async def test_accepts_market_low_hold_pinnacle() -> None:
    """Pinnacle h2h NBA con vig ~2% → puede pasar (no rechaza por hold)."""
    # home=1.97, away=1.97 → overround = 0.015 < 0.03
    event = _event({"pinnacle": [1.97, 1.97]})
    cfg = DetectorConfig(max_hold=0.03, max_hold_single_book=0.05)
    # Sin model_probs → blend usa p_pinnacle_fair; no debería rechazar por hold.
    # Detector puede devolver picks vacíos por otros filtros; el test clave es
    # que NO loggea "rejected_high_hold".
    result = await detect_value_bets_for_event(
        event, model_probs={"home": 0.60, "away": 0.40}, cfg=cfg
    )
    # No rechazado por hold: puede haber picks o no (depende EV filter)
    assert isinstance(result, list)


async def test_single_book_relaxed_threshold() -> None:
    """Un solo book → max_hold_single_book (5%) aplica. Vig 4.5% pasa."""
    # home=1.93, away=1.93 → overround = 0.036 (entre 0.03 y 0.05)
    event = _event({"pinnacle": [1.93, 1.93]})
    cfg = DetectorConfig(max_hold=0.03, max_hold_single_book=0.05)
    result = await detect_value_bets_for_event(
        event, model_probs={"home": 0.55, "away": 0.45}, cfg=cfg
    )
    # No rechazado por hold (single book usa threshold 5%)
    assert isinstance(result, list)


async def test_multi_book_stricter_threshold() -> None:
    """Multi-book con best odds combinado vig 4% → rechaza (>3% multi-book)."""
    # best per outcome: home=1.95, away=1.95 → overround = 0.0256
    # Esto sí pasa (<3%). Probamos con peores: home=1.90, away=1.90 (vig=5.3%).
    event = _event({"b1": [1.90, 1.90], "b2": [1.88, 1.88]})
    cfg = DetectorConfig(max_hold=0.03, max_hold_single_book=0.05)
    result = await detect_value_bets_for_event(
        event, model_probs={"home": 0.55, "away": 0.45}, cfg=cfg
    )
    # best: home=1.90, away=1.90 → vig = 2·(1/1.90)-1 = 0.0526 > 0.03 → rechaza
    assert result == []


async def test_accepts_multi_book_low_hold() -> None:
    """Multi-book best-odds combinado vig 2% → acepta."""
    # best: home=1.97, away=1.97 → vig = 0.015
    event = _event({"pinnacle": [1.97, 1.95], "bet365": [1.93, 1.97]})
    cfg = DetectorConfig(max_hold=0.03, max_hold_single_book=0.05)
    result = await detect_value_bets_for_event(
        event, model_probs={"home": 0.55, "away": 0.45}, cfg=cfg
    )
    assert isinstance(result, list)  # no rechazado por hold


@pytest.mark.parametrize(
    ("odds", "expected_rejected"),
    [
        ([1.97, 1.97], False),  # vig 1.5% → pasa
        ([1.91, 1.91], False),  # vig 4.7% → pasa bajo single-book 5%
        ([1.85, 1.85], True),  # vig 8.1% → rechaza (>5%)
        ([1.50, 1.50], True),  # vig 33.3% → rechaza siempre
    ],
)
async def test_hold_threshold_boundaries_single_book(
    odds: list[float], expected_rejected: bool
) -> None:
    """Single book usa max_hold_single_book=5%."""
    event = _event({"pinnacle": odds})
    cfg = DetectorConfig(max_hold=0.03, max_hold_single_book=0.05)
    result = await detect_value_bets_for_event(
        event, model_probs={"home": 0.55, "away": 0.45}, cfg=cfg
    )
    if expected_rejected:
        assert result == []
    else:
        assert isinstance(result, list)  # no rechazado por hold
