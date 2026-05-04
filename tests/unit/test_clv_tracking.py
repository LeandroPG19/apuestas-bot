"""Tests CLV tracking — Sprint 12."""

from __future__ import annotations

import pytest

from apuestas.flows.capture_closing_lines import compute_clv_pct


def test_clv_positive_when_picked_better_odds() -> None:
    """Tomaste @2.10, cerró @2.00 → +5% CLV."""
    clv = compute_clv_pct(odds_at_pick=2.10, odds_closing=2.00)
    assert clv == pytest.approx(0.05, abs=1e-6)


def test_clv_negative_when_picked_worse_odds() -> None:
    """Tomaste @1.90, cerró @2.00 → negativo."""
    clv = compute_clv_pct(odds_at_pick=1.90, odds_closing=2.00)
    assert clv < 0
    assert clv == pytest.approx(-0.05, abs=1e-6)


def test_clv_zero_when_same_odds() -> None:
    assert compute_clv_pct(odds_at_pick=2.00, odds_closing=2.00) == 0.0


def test_clv_handles_invalid_odds() -> None:
    assert compute_clv_pct(odds_at_pick=1.0, odds_closing=2.0) == 0.0
    assert compute_clv_pct(odds_at_pick=2.0, odds_closing=1.0) == 0.0
    assert compute_clv_pct(odds_at_pick=-1.0, odds_closing=2.0) == 0.0


def test_clv_buchdahl_reference_case() -> None:
    """Buchdahl 2023: 3% CLV consistente = skill profesional."""
    # Escenario: tomas @2.25, cerró @2.18 → (2.25/2.18 - 1) ≈ +3.2%
    clv = compute_clv_pct(odds_at_pick=2.25, odds_closing=2.18)
    assert 0.028 < clv < 0.035


def test_clv_longshot_picks() -> None:
    """Picks longshot: odds altas, CLV relativo grande."""
    # Tomaste @5.00, cerró @4.50 → +11.1% CLV
    clv = compute_clv_pct(odds_at_pick=5.00, odds_closing=4.50)
    assert clv == pytest.approx(0.1111, abs=1e-3)
