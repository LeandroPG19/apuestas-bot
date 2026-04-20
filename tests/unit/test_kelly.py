"""Tests de Kelly Criterion y variante correlation-aware."""

from __future__ import annotations

import pytest

from apuestas.risk.kelly import (
    KellyBet,
    correlation_aware_kelly,
    correlation_matrix,
    implied_correlation,
    kelly_fraction,
)


def test_kelly_zero_when_no_edge() -> None:
    """Si p = 1/odds, no hay edge → Kelly = 0."""
    assert kelly_fraction(p=0.5, odds=2.0) == 0.0


def test_kelly_zero_when_negative_edge() -> None:
    assert kelly_fraction(p=0.4, odds=2.0) == 0.0


def test_kelly_positive_and_capped() -> None:
    """Edge real + Kelly debe ser positivo y ≤ cap."""
    # p=0.6 @ odds=2.0 → b=1, b*p - q = 0.6 - 0.4 = 0.2; f* = 0.2; × 0.25 = 0.05
    f = kelly_fraction(p=0.6, odds=2.0, fraction=0.25, cap=0.10)
    assert 0 < f <= 0.10
    assert f == pytest.approx(0.05, abs=1e-6)


def test_kelly_respects_cap() -> None:
    """Edge enorme debe quedar capado."""
    f = kelly_fraction(p=0.95, odds=3.0, fraction=1.0, cap=0.05)
    assert f == pytest.approx(0.05)


def test_kelly_zero_on_degenerate_prob() -> None:
    assert kelly_fraction(p=0.0, odds=2.0) == 0.0
    assert kelly_fraction(p=1.0, odds=2.0) == 0.0


def test_implied_correlation_same_event_same_market() -> None:
    a = KellyBet(p=0.55, odds=1.9, event_id=123, market="h2h")
    b = KellyBet(p=0.55, odds=1.9, event_id=123, market="h2h")
    assert implied_correlation(a, b) == pytest.approx(0.9)


def test_implied_correlation_same_event_different_market() -> None:
    a = KellyBet(p=0.55, odds=1.9, event_id=123, market="h2h")
    b = KellyBet(p=0.55, odds=1.9, event_id=123, market="totals")
    assert implied_correlation(a, b) == pytest.approx(0.6)


def test_implied_correlation_different_events() -> None:
    a = KellyBet(p=0.55, odds=1.9, event_id=1, market="h2h")
    b = KellyBet(p=0.55, odds=1.9, event_id=2, market="h2h")
    assert implied_correlation(a, b) < 0.2


def test_correlation_matrix_is_symmetric_and_unit_diagonal() -> None:
    bets = [
        KellyBet(p=0.55, odds=1.9, event_id=1, market="h2h"),
        KellyBet(p=0.55, odds=1.9, event_id=1, market="totals"),
        KellyBet(p=0.55, odds=1.9, event_id=2, market="h2h"),
    ]
    c = correlation_matrix(bets)
    assert c.shape == (3, 3)
    assert all(c[i, i] == 1.0 for i in range(3))
    for i in range(3):
        for j in range(i + 1, 3):
            assert c[i, j] == c[j, i]


def test_correlation_aware_kelly_respects_daily_cap() -> None:
    """Suma de stakes debe respetar daily_cap."""
    bets = [KellyBet(p=0.60, odds=2.0, event_id=i, market="h2h") for i in range(10)]
    stakes = correlation_aware_kelly(bets, fraction=0.25, cap_per_bet=0.05, daily_cap=0.15)
    assert len(stakes) == 10
    # Small tolerance for QP numerics
    assert sum(stakes) <= 0.15 + 1e-3
    assert all(0.0 <= s <= 0.05 + 1e-6 for s in stakes)


def test_correlation_aware_kelly_single_bet() -> None:
    bets = [KellyBet(p=0.60, odds=2.0, event_id=1, market="h2h")]
    stakes = correlation_aware_kelly(bets, fraction=0.25, cap_per_bet=0.05)
    assert len(stakes) == 1
    assert 0 < stakes[0] <= 0.05
