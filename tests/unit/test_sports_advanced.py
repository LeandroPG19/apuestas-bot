"""Tests features avanzadas multi-deporte — Sprint 11 completamiento."""

from __future__ import annotations

import pytest

from apuestas.features.sports_advanced import (
    AssistStats,
    LineInjuryImpact,
    OnOffSplit,
    UmpireProfile,
    approximate_xa,
    contrarian_signal,
    line_injury_ev_adjustment,
    pdo_regression_signal,
    star_out_adjustment,
    umpire_k_adjustment,
)


def test_xa_positive() -> None:
    stats = AssistStats(key_passes=8, shots_assisted=5, xg_assisted=1.2)
    assert approximate_xa(stats) == pytest.approx(1.2)


def test_xa_clamp_negative() -> None:
    stats = AssistStats(key_passes=0, shots_assisted=0, xg_assisted=-0.5)
    assert approximate_xa(stats) == 0.0


def test_onoff_split_impact_positive() -> None:
    s = OnOffSplit(player_id=1, minutes_on=100, team_net_rating_on=5.0, team_net_rating_off=-3.0)
    assert s.impact == 8.0


def test_star_out_adjustment_negative() -> None:
    # Team peor sin estrella
    delta = star_out_adjustment(team_avg_on_off=5.0, team_avg_without_star=-2.0)
    assert delta < 0  # team worse → prob home win down


def test_umpire_k_adjustment_pitcher_friendly() -> None:
    profile = UmpireProfile(
        umpire_id=1,
        called_strike_pct_out_of_zone=0.25,  # expansive zone
        called_ball_pct_in_zone=0.08,
        consistency_score=0.85,
    )
    adj = umpire_k_adjustment(profile)
    assert adj > 0  # more Ks expected


def test_umpire_k_adjustment_tight_zone() -> None:
    profile = UmpireProfile(
        umpire_id=1,
        called_strike_pct_out_of_zone=0.08,
        called_ball_pct_in_zone=0.12,
        consistency_score=0.85,
    )
    adj = umpire_k_adjustment(profile)
    assert adj < 0


def test_line_injury_negative_with_starters_out() -> None:
    impact = LineInjuryImpact(
        team_id=1,
        starters_out_oline=2,
        starters_out_dline=1,
        pff_grade_drop_oline=5.0,
        pff_grade_drop_dline=3.0,
    )
    delta = line_injury_ev_adjustment(impact)
    assert delta < 0


def test_pdo_regression_high_pdo_negative_signal() -> None:
    sig = pdo_regression_signal(pdo_last_10=1.05)
    assert sig < 0  # lucky team → regression DOWN


def test_pdo_regression_low_pdo_positive_signal() -> None:
    sig = pdo_regression_signal(pdo_last_10=0.95)
    assert sig > 0  # unlucky → regression UP


def test_pdo_regression_at_mean_zero() -> None:
    sig = pdo_regression_signal(pdo_last_10=1.000)
    assert abs(sig) < 1e-6


def test_contrarian_sharp_beats_public() -> None:
    # 80% public un lado, pero sharp money solo 30% → sharp está contra público
    sig = contrarian_signal(public_pct_side=0.80, sharp_pct_side=0.30)
    assert sig < 0  # No confiar en ese lado


def test_contrarian_aligned_zero() -> None:
    sig = contrarian_signal(public_pct_side=0.50, sharp_pct_side=0.50)
    assert sig == 0.0
