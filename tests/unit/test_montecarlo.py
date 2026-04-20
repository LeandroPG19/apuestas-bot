"""Tests de Monte Carlo risk simulator."""

from __future__ import annotations

import pytest

from apuestas.risk.montecarlo import MCConfig, recommend_kelly_adjustment, simulate


@pytest.mark.slow
def test_positive_edge_yields_positive_expected_bankroll() -> None:
    """Con edges positivos muestreados, bankroll esperado >= inicial."""
    cfg = MCConfig(
        n_simulations=500,
        n_bets_per_path=200,
        initial_bankroll=1000.0,
        edge_distribution=[0.03, 0.04, 0.05],
        odds_distribution=[1.90, 2.00, 2.10],
        seed=42,
    )
    result = simulate(cfg)
    assert result.expected_final_bankroll > 500  # no ruinoso con edge pequeno
    assert result.p50_final_bankroll > 0


def test_probabilities_bounded() -> None:
    cfg = MCConfig(
        n_simulations=200,
        n_bets_per_path=100,
        seed=7,
    )
    result = simulate(cfg)
    for prob in (
        result.prob_dd_25pct,
        result.prob_dd_40pct,
        result.prob_ruin,
        result.prob_double,
    ):
        assert 0.0 <= prob <= 1.0


def test_recommend_kelly_adjustment_triggers_on_high_dd40() -> None:
    from apuestas.risk.montecarlo import MCResult

    result = MCResult(
        prob_dd_25pct=0.5,
        prob_dd_40pct=0.10,  # > 5% threshold
        prob_ruin=0.05,
        prob_double=0.40,
        expected_final_bankroll=1200,
        p10_final_bankroll=400,
        p50_final_bankroll=1100,
        p90_final_bankroll=2000,
        median_max_drawdown_pct=0.35,
        params={},
    )
    recommendation = recommend_kelly_adjustment(result)
    assert recommendation is not None
    assert "reducir kelly_fraction" in recommendation.lower()


def test_recommend_kelly_adjustment_no_trigger_on_safe() -> None:
    from apuestas.risk.montecarlo import MCResult

    result = MCResult(
        prob_dd_25pct=0.10,
        prob_dd_40pct=0.01,
        prob_ruin=0.001,
        prob_double=0.60,
        expected_final_bankroll=2000,
        p10_final_bankroll=900,
        p50_final_bankroll=1800,
        p90_final_bankroll=3500,
        median_max_drawdown_pct=0.10,
        params={},
    )
    assert recommend_kelly_adjustment(result) is None
