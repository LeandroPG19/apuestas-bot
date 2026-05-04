"""Tests para apuestas.betting.market_consensus (Sprint 6c)."""

from __future__ import annotations

import pytest

from apuestas.betting.market_consensus import (
    compute_consensus_sharp,
    consensus_delta,
    is_significant_disagreement,
)


def test_consensus_all_three_sources() -> None:
    r = compute_consensus_sharp(
        pinnacle_devigged=0.60,
        polymarket_mid=0.58,
        kalshi_mid=0.61,
        polymarket_volume_usd=50_000,
    )
    # ponderado: 0.5*0.60 + 0.3*0.58 + 0.2*0.61 = 0.596
    assert r.p_consensus == pytest.approx(0.596, rel=1e-3)
    assert r.sources == 3
    assert r.dispersion > 0


def test_consensus_skips_polymarket_low_volume() -> None:
    r = compute_consensus_sharp(
        pinnacle_devigged=0.60,
        polymarket_mid=0.58,
        kalshi_mid=None,
        polymarket_volume_usd=500,
    )
    # Polymarket excluido por volumen bajo; solo Pinnacle queda
    assert r.sources == 1
    assert r.polymarket is None
    assert r.p_consensus == pytest.approx(0.60)


def test_consensus_only_pinnacle() -> None:
    r = compute_consensus_sharp(
        pinnacle_devigged=0.55,
        polymarket_mid=None,
        kalshi_mid=None,
    )
    assert r.sources == 1
    assert r.p_consensus == pytest.approx(0.55)
    assert r.dispersion == 0.0


def test_consensus_fallback_when_no_sources() -> None:
    r = compute_consensus_sharp(pinnacle_devigged=None, polymarket_mid=None, kalshi_mid=None)
    assert r.sources == 0
    assert r.p_consensus == pytest.approx(0.5)


def test_delta_simple() -> None:
    r = compute_consensus_sharp(pinnacle_devigged=0.60, polymarket_mid=0.60, kalshi_mid=0.60)
    assert consensus_delta(0.62, r) == pytest.approx(0.02, abs=1e-6)


def test_is_significant_disagreement_needs_two_sources() -> None:
    r_one = compute_consensus_sharp(pinnacle_devigged=0.50, polymarket_mid=None, kalshi_mid=None)
    # Con una sola fuente, aunque delta sea alto, no marcamos disagreement.
    assert is_significant_disagreement(0.70, r_one) is False


def test_is_significant_disagreement_triggers_above_threshold() -> None:
    r = compute_consensus_sharp(
        pinnacle_devigged=0.50,
        polymarket_mid=0.52,
        kalshi_mid=0.49,
        polymarket_volume_usd=50_000,
    )
    assert is_significant_disagreement(0.62, r, threshold_pp=0.08) is True
    assert is_significant_disagreement(0.54, r, threshold_pp=0.08) is False
