"""Tests Sprint 11: Venn-Abers, closing_line, book_power, xT, clutch, kalman, timing, weather."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest


# ───────── Fase A — Venn-Abers wrapper ─────────
def test_venn_abers_wrapper_interface() -> None:
    from apuestas.ml.calibrate import _VennAbersWrapper

    class _Base:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])

    class _FakeVA:
        def predict_proba(self, p_2d: np.ndarray):
            # API real: retorna (p_prime (n, 2), p0_p1 (n, 2))
            n = p_2d.shape[0]
            p_prime = np.column_stack([np.full(n, 0.42), np.full(n, 0.58)])
            p0p1 = np.column_stack([np.full(n, 0.40), np.full(n, 0.60)])
            return p_prime, p0p1

    w = _VennAbersWrapper(base_estimator=_Base(), va=_FakeVA())
    probs = w.predict_proba(np.zeros((3, 2)))
    assert probs.shape == (3, 2)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert probs[0, 1] == pytest.approx(0.58, abs=1e-6)


# ───────── Fase A — Focal loss flag ─────────
def test_stacker_focal_loss_flag() -> None:
    from apuestas.ml.stacker import MarketAwareStacker

    s = MarketAwareStacker(focal_loss=True, focal_alpha=0.5, focal_gamma=1.0)
    assert s.focal_loss is True
    assert s.focal_alpha == 0.5
    assert s.focal_gamma == 1.0


# ───────── Fase B — TabPFN stacker ─────────
def test_tabpfn_stacker_fallback_logreg() -> None:
    from apuestas.ml.tabpfn_stacker import TabPFNStacker

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 3))
    y = (X[:, 0] > 0).astype(int)
    s = TabPFNStacker()
    s.fit(X, y)
    probs = s.predict_proba(X)
    assert probs.shape == (100, 2)


# ───────── Fase C — Closing line predictor ─────────
def test_closing_line_predictor_default_without_fit() -> None:
    from apuestas.betting.closing_line_predictor import (
        ClosingLineFeatures,
        ClosingLinePredictor,
    )

    pred = ClosingLinePredictor(sport="nba")
    feats = ClosingLineFeatures(
        current_odds=1.95,
        line_movement_4h=0.0,
        line_movement_1h=0.0,
        n_updates_4h=5,
        n_books_tracking=3,
        sharp_book_consensus=1.90,
        public_pct=0.5,
        hours_until_start=3.0,
        sport_code="nba",
        league_id=None,
    )
    # Sin fit → heurística weighted
    p = pred.predict(feats)
    assert 1.85 < p < 2.0


def test_closing_line_predictor_fit_and_predict() -> None:
    from apuestas.betting.closing_line_predictor import (
        ClosingLineFeatures,
        ClosingLinePredictor,
    )

    rng = np.random.default_rng(7)
    X = []
    y = []
    for _ in range(50):
        X.append(
            ClosingLineFeatures(
                current_odds=float(rng.uniform(1.5, 3.0)),
                line_movement_4h=float(rng.normal(0, 0.02)),
                line_movement_1h=float(rng.normal(0, 0.01)),
                n_updates_4h=int(rng.integers(1, 10)),
                n_books_tracking=int(rng.integers(1, 8)),
                sharp_book_consensus=float(rng.uniform(1.5, 3.0)),
                public_pct=float(rng.uniform(0.3, 0.7)),
                hours_until_start=float(rng.uniform(0.5, 6.0)),
                sport_code="nba",
                league_id=None,
            )
        )
        y.append(float(rng.uniform(1.4, 3.2)))
    pred = ClosingLinePredictor(sport="nba").fit(X, y)
    out = pred.predict(X[0])
    assert out > 0
    clv = pred.anticipated_clv(X[0])
    assert isinstance(clv, float)


def test_closing_line_empty_fit_raises() -> None:
    from apuestas.betting.closing_line_predictor import ClosingLinePredictor

    with pytest.raises(ValueError, match="vacío"):
        ClosingLinePredictor(sport="nba").fit([], [])


# ───────── Fase D — Book power ratings ─────────
def test_book_power_get_cached_edge_default_zero() -> None:
    from apuestas.betting.book_power_ratings import get_cached_edge

    assert get_cached_edge("nonexistent_book", "fake_league") == 0.0


def test_book_power_rank_books_empty() -> None:
    from apuestas.betting.book_power_ratings import rank_books_for

    result = rank_books_for("fake_league", "nba")
    assert result == []


# ───────── Fase E — Soccer xT ─────────
def test_approximate_xt_returns_positive() -> None:
    from apuestas.features.soccer_xt import MatchThreatStats, approximate_xt

    stats = MatchThreatStats(
        possession_pct=0.6,
        shots_total=15,
        shots_on_target=5,
        progressive_passes=20,
        progressive_carries=10,
        avg_position_x=0.65,
    )
    xt = approximate_xt(stats)
    assert xt > 0


def test_add_xt_rolling_preserves_frame() -> None:
    from apuestas.features.soccer_xt import add_xt_rolling

    df = pl.DataFrame(
        {
            "team_id": [1, 1, 2, 2] * 3,
            "start_time": [datetime(2025, 1, d, tzinfo=UTC) for d in range(1, 13)],
            "possession_pct": [0.55] * 12,
            "shots_total": [10] * 12,
            "shots_on_target": [4] * 12,
        }
    )
    out = add_xt_rolling(df)
    assert "xt_raw" in out.columns
    assert "xt_mean_roll_5" in out.columns


# ───────── Fase F — NBA clutch ─────────
def test_clutch_stats_ortg() -> None:
    from apuestas.features.nba_clutch import ClutchStats

    s = ClutchStats(
        team_id=1,
        points_clutch=20,
        possessions_clutch=18,
        ft_made_clutch=5,
        ft_att_clutch=6,
        turnovers_clutch=2,
        minutes_clutch=5.0,
    )
    assert s.clutch_ortg == pytest.approx(111.111, abs=0.01)
    assert s.clutch_ft_pct == pytest.approx(5 / 6, abs=0.01)


def test_lineup_efficiency_net_rating() -> None:
    from apuestas.features.nba_clutch import LineupEfficiency

    lu = LineupEfficiency(
        lineup_hash="1-2-3-4-5",
        team_id=1,
        minutes=100.0,
        points_for=220,
        points_against=200,
        possessions=100,
    )
    assert lu.net_rating == pytest.approx(20.0, abs=0.01)


# ───────── Fase G — MLB Stuff+ ─────────
def test_estimate_stuff_plus_league_avg() -> None:
    from apuestas.features.mlb_pitching_plus import (
        PitcherStuffMetrics,
        estimate_stuff_plus,
    )

    m = PitcherStuffMetrics(
        pitcher_id=1,
        spin_rate_avg=2300.0,
        velo_avg=93.8,
        whiff_pct=0.115,
        csw_pct=0.28,
        chase_pct=0.30,
        release_consistency=0.08,
        n_pitches=500,
    )
    stuff = estimate_stuff_plus(m)
    assert 95 <= stuff <= 105


def test_estimate_pitching_plus_elite() -> None:
    from apuestas.features.mlb_pitching_plus import (
        PitcherStuffMetrics,
        estimate_pitching_plus,
    )

    m = PitcherStuffMetrics(
        pitcher_id=1,
        spin_rate_avg=2600.0,  # elite spin
        velo_avg=97.0,
        whiff_pct=0.15,
        csw_pct=0.32,
        chase_pct=0.36,
        release_consistency=0.04,
        n_pitches=500,
    )
    pp = estimate_pitching_plus(m)
    assert pp > 110


# ───────── Fase H — FT-Transformer ─────────
def test_ft_transformer_fallback_when_not_fitted() -> None:
    from apuestas.ml.ft_transformer import FTTransformerClassifier

    clf = FTTransformerClassifier(n_features=5)
    X = np.zeros((3, 5))
    probs = clf.predict_proba(X)
    assert probs.shape == (3, 2)
    assert np.allclose(probs[:, 1], 0.5)


# ───────── Fase I — Kalman live ─────────
def test_kalman_initial_p_home() -> None:
    from apuestas.betting.live_kalman import LiveKalmanFilter

    kf = LiveKalmanFilter(sport="soccer", initial_p_home=0.55)
    assert kf.p_home_win() == pytest.approx(0.55, abs=1e-3)


def test_kalman_goal_shifts_posterior() -> None:
    from apuestas.betting.live_kalman import LiveKalmanFilter

    kf = LiveKalmanFilter(sport="soccer", initial_p_home=0.50)
    kf.observe_goal(team="home", minute=30)
    assert kf.p_home_win() > 0.50
    kf.observe_goal(team="away", minute=60)
    # Goal al away debería bajar p_home (puede quedar > 0.5 por prior)
    before = kf.p_home_win()
    kf.observe_goal(team="away", minute=80)
    assert kf.p_home_win() < before


def test_kalman_score_delta() -> None:
    from apuestas.betting.live_kalman import LiveKalmanFilter

    kf = LiveKalmanFilter(sport="nba", initial_p_home=0.50)
    kf.observe_score_delta(home_score=90, away_score=80, minute=40)
    # NBA con ventaja home al 83% del game → p_home debe subir
    assert kf.p_home_win() > 0.55


# ───────── Fase J — Execution timing ─────────
def test_score_timing_optimal_window_nba() -> None:
    from apuestas.betting.execution_timing import score_timing

    now = datetime(2025, 4, 24, 7, 30, tzinfo=UTC)  # 07:30 UTC
    kickoff = now + timedelta(hours=12)
    score = score_timing(sport_code="nba", kickoff_utc=kickoff, now_utc=now)
    assert score.in_optimal_window is True
    assert score.edge_multiplier > 1.0


def test_score_timing_late_kickoff() -> None:
    from apuestas.betting.execution_timing import score_timing

    now = datetime(2025, 4, 24, 20, 0, tzinfo=UTC)
    kickoff = now + timedelta(minutes=20)  # < 30 min
    score = score_timing(sport_code="nfl", kickoff_utc=kickoff, now_utc=now)
    assert score.edge_multiplier < 1.0


# ───────── Fase J — Weather ─────────
def test_weather_mlb_wind_out_boosts_totals() -> None:
    from apuestas.betting.information_edge import compute_weather_adjustment_mlb

    adj = compute_weather_adjustment_mlb(
        wind_speed_mph=15.0,
        wind_direction="out",
        temperature_f=78.0,
        humidity_pct=50.0,
        precip_prob=0.0,
    )
    assert adj.total_runs_delta > 0
    assert "wind_out" in adj.reason


def test_weather_nfl_high_wind_penalizes() -> None:
    from apuestas.betting.information_edge import compute_weather_adjustment_nfl

    adj = compute_weather_adjustment_nfl(
        wind_speed_mph=25.0,
        precip_prob=0.6,
        temperature_f=35.0,
    )
    assert adj.total_runs_delta < 0


def test_sharp_divergence_detects_reverse_line_move() -> None:
    from apuestas.betting.information_edge import SharpDivergence

    div = SharpDivergence.compute(
        public_pct_home=0.80,
        line_movement_home=+0.05,  # línea se mueve a away (contra público)
    )
    assert div.is_divergence is True
    assert div.signal_strength > 0


def test_sharp_divergence_no_divergence_when_aligned() -> None:
    from apuestas.betting.information_edge import SharpDivergence

    div = SharpDivergence.compute(
        public_pct_home=0.80,
        line_movement_home=-0.05,  # línea se mueve a home (alineado)
    )
    assert div.is_divergence is False
