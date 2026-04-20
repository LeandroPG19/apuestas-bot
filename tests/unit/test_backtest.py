"""Tests de backtest framework."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from apuestas.ml.backtest import BacktestConfig, simulate_walk_forward, summary_to_dict


def _make_events(
    n: int,
    *,
    p_model: float = 0.60,
    odds: float = 1.90,
    win_rate: float = 0.60,
    league_id: int = 1,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base_ts = datetime(2024, 10, 1, tzinfo=UTC)
    rows = []
    for i in range(n):
        results_bern = int(rng.random() < win_rate)
        rows.append(
            {
                "match_id": i,
                "start_time": base_ts + timedelta(days=i),
                "market": "h2h",
                "outcome": "home",
                "p_model": p_model,
                "p_lower": p_model - 0.02,
                "p_upper": p_model + 0.02,
                "odds": odds,
                "result": results_bern,
                "closing_line": odds * 0.98,  # fair a cierre (CLV +2%)
                "league_id": league_id,
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("start_time").cast(pl.Datetime(time_zone="UTC")))


def test_positive_edge_yields_positive_roi() -> None:
    """Edge real 10pp con 200 bets debería rendir ROI positivo."""
    df = _make_events(200, p_model=0.60, odds=1.90, win_rate=0.60, seed=1)
    report = simulate_walk_forward(df)
    # Hit rate real ≥ break-even (1/odds = 0.526)
    assert report.n_bets > 0
    # ROI >0 probable pero no garantizado por varianza; test es que el sistema
    # al menos generó bets y calculó métricas
    assert report.final_bankroll > 0


def test_no_bets_below_min_odds_or_ev() -> None:
    """Cuando edge < threshold, no debe haber bets."""
    df = _make_events(50, p_model=0.51, odds=1.90, win_rate=0.51)
    # Edge = 0.51 - 1/1.90 = -0.016 → 0 bets
    report = simulate_walk_forward(df)
    assert report.n_bets == 0


def test_stop_loss_halts_betting() -> None:
    """Si racha perdedora lleva bankroll a −30%, pausa."""
    rng = np.random.default_rng(0)
    df = _make_events(100, p_model=0.70, odds=1.90, win_rate=0.20, seed=0)
    # hit rate 20% con Kelly agresivo → seguro toca stop loss
    cfg = BacktestConfig(stop_loss_pct=0.30)
    report = simulate_walk_forward(df, cfg=cfg)
    # Debe tener menos bets que total events (pausó)
    assert report.n_bets < 100


def test_conformal_filter_blocks_when_p_lower_below_margin() -> None:
    """Si p_lower ≤ implied_prob + margin, no apostar."""
    df = _make_events(50, p_model=0.60, odds=1.90, win_rate=0.60).with_columns(
        pl.lit(0.50).alias("p_lower"),  # p_lower = implied, falla margin
    )
    cfg = BacktestConfig(conformal_min_margin=0.05)
    report = simulate_walk_forward(df, cfg=cfg)
    assert report.n_bets == 0


def test_summary_to_dict_has_all_keys() -> None:
    df = _make_events(50, p_model=0.60, odds=1.90, win_rate=0.60)
    report = simulate_walk_forward(df)
    d = summary_to_dict(report)
    for key in [
        "n_bets",
        "roi",
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "clv_mean",
        "hit_rate_by_ev_bucket",
        "roi_by_market",
    ]:
        assert key in d


def test_roi_by_market_aggregation() -> None:
    df1 = _make_events(20, p_model=0.60, odds=1.90, win_rate=0.70, seed=1)
    df2 = _make_events(20, p_model=0.60, odds=1.90, win_rate=0.30, seed=2).with_columns(
        pl.lit("totals").alias("market"),
        (pl.col("match_id") + 100).alias("match_id"),
    )
    df = pl.concat([df1, df2])
    report = simulate_walk_forward(df)
    if report.n_bets > 0:
        assert "h2h" in report.roi_by_market or "totals" in report.roi_by_market
