"""Tests para el classifier de resultados de pick_alerts (Sprint 3).

Prueba `_classify_alert` — la función pura que mapea (market, outcome,
line, score) → won/lost/void/None. No requiere DB.
"""

from __future__ import annotations

import pytest

from apuestas.flows.live_scores import _classify_alert

# ─────────────── h2h / moneyline ───────────────


def test_h2h_home_wins() -> None:
    assert (
        _classify_alert(
            market="h2h", outcome="home", line=None, home_score=5, away_score=2, sport="mlb"
        )
        == "won"
    )


def test_h2h_away_loses_when_home_wins() -> None:
    assert (
        _classify_alert(
            market="h2h", outcome="away", line=None, home_score=5, away_score=2, sport="mlb"
        )
        == "lost"
    )


def test_h2h_soccer_draw_is_won_when_bet_draw() -> None:
    assert (
        _classify_alert(
            market="h2h", outcome="draw", line=None, home_score=1, away_score=1, sport="soccer"
        )
        == "won"
    )


def test_h2h_soccer_draw_lost_when_bet_home() -> None:
    assert (
        _classify_alert(
            market="h2h", outcome="home", line=None, home_score=1, away_score=1, sport="soccer"
        )
        == "lost"
    )


def test_h2h_mlb_tie_is_void_for_home_bet() -> None:
    # MLB no admite empate en regla. Un empate stored es anomalía → void.
    assert (
        _classify_alert(
            market="h2h", outcome="home", line=None, home_score=3, away_score=3, sport="mlb"
        )
        == "void"
    )


# ─────────────── spreads / handicap ───────────────


def test_spreads_home_covers() -> None:
    # home -1.5; score 5-2 (diff +3) → home cubre
    assert (
        _classify_alert(
            market="spreads", outcome="home", line=-1.5, home_score=5, away_score=2, sport="mlb"
        )
        == "won"
    )


def test_spreads_away_does_not_cover() -> None:
    # away +1.5; score 5-2 away perdió por 3 → pick away pierde
    assert (
        _classify_alert(
            market="spreads", outcome="away", line=1.5, home_score=5, away_score=2, sport="mlb"
        )
        == "lost"
    )


def test_spreads_away_covers_with_positive_line() -> None:
    # Regresión bug 2026-04-23: NYM 3-2 MIN, away+1.5 → away pierde por 1 pero cubre.
    # Antes del fix: `inv = away - home - line` daba -2.5 (lost). Correcto: +0.5 (won).
    assert (
        _classify_alert(
            market="spreads", outcome="away", line=1.5, home_score=3, away_score=2, sport="mlb"
        )
        == "won"
    )


def test_spreads_away_exact_push_positive_line() -> None:
    # away+3 con home=5 away=2 → diff (desde away) = -3 + 3 = 0 → push/void
    assert (
        _classify_alert(
            market="spreads", outcome="away", line=3.0, home_score=5, away_score=2, sport="nba"
        )
        == "void"
    )


def test_spreads_away_favorite_negative_line_wins() -> None:
    # away -1.5 con away=5 home=2 → away favorito gana por 3, cubre.
    assert (
        _classify_alert(
            market="spreads", outcome="away", line=-1.5, home_score=2, away_score=5, sport="nba"
        )
        == "won"
    )


def test_spreads_exact_push_is_void() -> None:
    # home -3; score 5-2 → diff = 0 → push
    assert (
        _classify_alert(
            market="spreads", outcome="home", line=-3.0, home_score=5, away_score=2, sport="nba"
        )
        == "void"
    )


def test_spreads_missing_line_returns_none() -> None:
    assert (
        _classify_alert(
            market="spreads", outcome="home", line=None, home_score=5, away_score=2, sport="nba"
        )
        is None
    )


# ─────────────── totals / over-under ───────────────


def test_totals_over_wins() -> None:
    # total 7 > 6.5 → over gana
    assert (
        _classify_alert(
            market="totals", outcome="over", line=6.5, home_score=5, away_score=2, sport="mlb"
        )
        == "won"
    )


def test_totals_under_wins() -> None:
    # total 7 < 9.5 → under gana
    assert (
        _classify_alert(
            market="totals", outcome="under", line=9.5, home_score=5, away_score=2, sport="mlb"
        )
        == "won"
    )


def test_totals_push_is_void() -> None:
    # total 7 == 7.0 → void
    assert (
        _classify_alert(
            market="totals", outcome="over", line=7.0, home_score=5, away_score=2, sport="mlb"
        )
        == "void"
    )
    assert (
        _classify_alert(
            market="totals", outcome="under", line=7.0, home_score=5, away_score=2, sport="mlb"
        )
        == "void"
    )


# ─────────────── markets no cubiertos ───────────────


@pytest.mark.parametrize(
    "market",
    [
        "player_points",
        "pitcher_strikeouts",
        "first_scorer",
        "team_totals",
    ],
)
def test_unsupported_markets_return_none(market: str) -> None:
    assert (
        _classify_alert(
            market=market, outcome="over", line=20.5, home_score=5, away_score=2, sport="nba"
        )
        is None
    )


def test_spreads_invalid_outcome_returns_none() -> None:
    assert (
        _classify_alert(
            market="spreads",
            outcome="draw",
            line=-1.5,
            home_score=5,
            away_score=2,
            sport="soccer",
        )
        is None
    )
