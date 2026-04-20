"""Tests del normalizador de outcomes."""

from __future__ import annotations

import pytest

from apuestas.ingest.outcome_normalizer import (
    normalize_btts_outcome,
    normalize_h2h_outcome,
    normalize_outcome,
    normalize_spread_outcome,
    normalize_totals_outcome,
)


def test_h2h_home_team_name() -> None:
    assert (
        normalize_h2h_outcome(
            raw_outcome="Crystal Palace",
            home_team_name="Crystal Palace FC",
            away_team_name="West Ham United FC",
        )
        == "home"
    )


def test_h2h_away_team_name() -> None:
    assert (
        normalize_h2h_outcome(
            raw_outcome="West Ham United",
            home_team_name="Crystal Palace FC",
            away_team_name="West Ham United FC",
        )
        == "away"
    )


def test_h2h_draw_aliases() -> None:
    for alias in ("Draw", "DRAW", "draw", "Empate", "X"):
        assert (
            normalize_h2h_outcome(raw_outcome=alias, home_team_name="A", away_team_name="B")
            == "draw"
        )


def test_h2h_special_tokens() -> None:
    assert (
        normalize_h2h_outcome(raw_outcome="HOME_TEAM", home_team_name="X", away_team_name="Y")
        == "home"
    )
    assert (
        normalize_h2h_outcome(raw_outcome="AWAY", home_team_name="X", away_team_name="Y") == "away"
    )


def test_h2h_fuzzy_strip_fc_suffix() -> None:
    """Real-world case: odds devuelven 'Crystal Palace' pero roster dice 'Crystal Palace FC'."""
    assert (
        normalize_h2h_outcome(
            raw_outcome="Manchester City",
            home_team_name="Manchester City FC",
            away_team_name="Liverpool FC",
        )
        == "home"
    )


def test_h2h_unknown_returns_none() -> None:
    assert (
        normalize_h2h_outcome(
            raw_outcome="Tottenham",
            home_team_name="Arsenal",
            away_team_name="Chelsea",
        )
        is None
    )


def test_totals_normalizer() -> None:
    assert normalize_totals_outcome(raw_outcome="Over") == "over"
    assert normalize_totals_outcome(raw_outcome="Under") == "under"
    assert normalize_totals_outcome(raw_outcome="Over 2.5") == "over"
    assert normalize_totals_outcome(raw_outcome="U") == "under"
    assert normalize_totals_outcome(raw_outcome="garbage") is None


def test_spread_normalizer_delegates_h2h() -> None:
    assert (
        normalize_spread_outcome(
            raw_outcome="Crystal Palace",
            home_team_name="Crystal Palace FC",
            away_team_name="West Ham FC",
        )
        == "home"
    )


def test_btts_normalizer() -> None:
    assert normalize_btts_outcome(raw_outcome="Yes") == "yes"
    assert normalize_btts_outcome(raw_outcome="No") == "no"
    assert normalize_btts_outcome(raw_outcome="GG") == "yes"


def test_dispatcher_h2h() -> None:
    assert (
        normalize_outcome(
            market="h2h",
            raw_outcome="Crystal Palace",
            home_team_name="Crystal Palace FC",
            away_team_name="West Ham FC",
        )
        == "home"
    )


def test_dispatcher_totals() -> None:
    assert normalize_outcome(market="totals", raw_outcome="Over 2.5") == "over"


def test_dispatcher_unknown_market() -> None:
    assert normalize_outcome(market="specials", raw_outcome="X") is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Home", "home"),
        ("Away", "away"),
        ("Draw", "draw"),
        ("Tie", "draw"),
        ("Empate", "draw"),
    ],
)
def test_common_patterns(raw: str, expected: str) -> None:
    assert (
        normalize_h2h_outcome(raw_outcome=raw, home_team_name="A", away_team_name="B") == expected
    )
