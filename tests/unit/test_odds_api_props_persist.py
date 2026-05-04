"""Tests para _persist_props_market / _markets_for_sport de odds_api_optimized."""

from __future__ import annotations

from apuestas.ingest.odds_api_optimized import _markets_for_sport


def test_markets_for_sport_default_no_props(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("APUESTAS_ENABLE_PROPS", raising=False)
    m = _markets_for_sport("nba")
    assert "h2h" in m
    assert "player_points" not in m


def test_markets_for_sport_with_props_enabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("APUESTAS_ENABLE_PROPS", "true")
    m = _markets_for_sport("nba")
    assert "player_points" in m
    assert "player_rebounds" in m


def test_markets_for_sport_no_props_when_flag_off(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("APUESTAS_ENABLE_PROPS", "false")
    m = _markets_for_sport("nba")
    assert "player_points" not in m


def test_markets_for_sport_soccer_epl_props(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("APUESTAS_ENABLE_PROPS", "true")
    m = _markets_for_sport("soccer_epl")
    assert "player_goal_scorer_anytime" in m


def test_markets_for_sport_unknown_sport_stays_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("APUESTAS_ENABLE_PROPS", "true")
    m = _markets_for_sport("mma")
    # mma no está en _PROPS_ENABLED_SPORTS ni en _PROPS_MARKETS_BY_SPORT → sin props
    assert "player_" not in m
