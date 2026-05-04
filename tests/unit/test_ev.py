"""Tests motor EV + line shopping (post-pivote detector puro).

Tests eliminados en 2026-04-23:
  - kelly_stake / bankroll / stake (subsistema removido)
  - compute_clv (ya no se trackea; Sprint 5 vuelve con snapshot inmutable)
"""

from __future__ import annotations

from typing import Any

import pytest

from apuestas.betting.ev import (
    BookmakerQuote,
    blend_probabilities,
    compute_ev,
    edge,
    evaluate_offer,
    find_best_price,
    implied_probability,
    line_shopping,
)


def test_compute_ev_break_even() -> None:
    """p=0.5 @ odds=2.0 → EV=0."""
    assert compute_ev(0.5, 2.0) == pytest.approx(0.0)


def test_compute_ev_positive() -> None:
    """p=0.55 @ odds=2.0 → EV=+10%."""
    assert compute_ev(0.55, 2.0) == pytest.approx(0.10)


def test_compute_ev_negative() -> None:
    assert compute_ev(0.45, 2.0) == pytest.approx(-0.10)


def test_implied_probability() -> None:
    assert implied_probability(2.0) == 0.5
    assert implied_probability(1.91) == pytest.approx(1 / 1.91)
    assert implied_probability(0.5) == 0.0  # protect invalid


def test_edge_positive() -> None:
    assert edge(0.55, 2.0) == pytest.approx(0.05)


def test_find_best_price_excludes_sharp() -> None:
    quotes = [
        BookmakerQuote(bookmaker="pinnacle", odds=2.00),
        BookmakerQuote(bookmaker="caliente", odds=1.95),
        BookmakerQuote(bookmaker="draftkings", odds=1.98),
    ]
    best = find_best_price(quotes, exclude_sharp=True)
    assert best is not None
    assert best.bookmaker == "draftkings"
    assert best.odds == 1.98


def test_find_best_price_respects_allowed_books() -> None:
    quotes = [
        BookmakerQuote(bookmaker="caliente", odds=1.95),
        BookmakerQuote(bookmaker="strendus", odds=1.97),
        BookmakerQuote(bookmaker="draftkings", odds=2.00),
    ]
    best = find_best_price(quotes, allowed_books=frozenset({"caliente", "strendus"}))
    assert best is not None
    assert best.bookmaker == "strendus"


def test_find_best_price_empty() -> None:
    assert find_best_price([], exclude_sharp=True) is None


def test_evaluate_offer_rejects_below_threshold() -> None:
    """Edge pequeño < threshold → None."""
    # p=0.505 @ odds=1.95 → EV=1.48%, bajo threshold 3% default
    q = BookmakerQuote(bookmaker="caliente", odds=1.95)
    offer = evaluate_offer(p_fair=0.505, quote=q)
    assert offer is None


def test_evaluate_offer_rejects_out_of_range(monkeypatch: Any) -> None:
    """Odds fuera del rango configurado → None."""
    monkeypatch.setenv("MIN_ODDS", "1.5")
    monkeypatch.setenv("MAX_ODDS", "4.0")
    from apuestas.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        q = BookmakerQuote(bookmaker="caliente", odds=10.0)
        offer = evaluate_offer(p_fair=0.20, quote=q)
        assert offer is None
    finally:
        get_settings.cache_clear()  # type: ignore[attr-defined]


def test_evaluate_offer_accepts_value_bet() -> None:
    """p=0.58 @ odds=1.95 → EV~+13%, pasa."""
    q = BookmakerQuote(bookmaker="caliente", odds=1.95)
    offer = evaluate_offer(p_fair=0.58, quote=q)
    assert offer is not None
    assert offer.ev > 0.10
    assert offer.edge > 0


def test_line_shopping_picks_best_and_evaluates() -> None:
    quotes = [
        BookmakerQuote(bookmaker="pinnacle", odds=1.85),  # excluido
        BookmakerQuote(bookmaker="caliente", odds=1.90),
        BookmakerQuote(bookmaker="strendus", odds=1.95),
        BookmakerQuote(bookmaker="draftkings", odds=1.92),
    ]
    offer = line_shopping(quotes, p_fair=0.58)
    assert offer is not None
    assert offer.bookmaker == "strendus"  # mejor precio soft


def test_blend_probabilities_ensemble() -> None:
    """Blend 0.4 model + 0.6 pinnacle."""
    blended = blend_probabilities(p_model=0.60, p_pinnacle_fair=0.50, weight_model=0.4)
    assert blended == pytest.approx(0.54)


def test_blend_clamps_weight() -> None:
    assert blend_probabilities(0.5, 0.7, weight_model=-0.5) == pytest.approx(0.7)
    assert blend_probabilities(0.5, 0.7, weight_model=1.5) == pytest.approx(0.5)
