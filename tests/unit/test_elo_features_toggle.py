"""Tests para add_elo_features — toggle env flag + anti-leakage."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from apuestas.features.common import add_elo_features


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "home_team_id": [1, 2, 1, 3],
            "away_team_id": [2, 1, 3, 1],
            "home_score": [100, 95, 110, 90],
            "away_score": [95, 100, 100, 100],
            "start_time": [
                datetime(2025, 1, 1, tzinfo=UTC),
                datetime(2025, 1, 5, tzinfo=UTC),
                datetime(2025, 1, 10, tzinfo=UTC),
                datetime(2025, 1, 15, tzinfo=UTC),
            ],
        }
    )


def test_add_elo_features_adds_columns() -> None:
    df = _sample_df()
    out = add_elo_features(df, sport="nba")
    for col in ("elo_home", "elo_away", "elo_diff", "elo_p_home"):
        assert col in out.columns


def test_env_flag_disables_elo(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("APUESTAS_ELO_FEATURES_DISABLED", "true")
    df = _sample_df()
    out = add_elo_features(df, sport="nba")
    assert "elo_home" not in out.columns


def test_first_match_has_default_rating() -> None:
    df = _sample_df()
    out = add_elo_features(df, sport="nba").sort("start_time")
    # El primer match tiene ambos ratings = 1500 (default)
    assert out["elo_home"][0] == 1500.0
    assert out["elo_away"][0] == 1500.0


def test_elo_updates_between_matches() -> None:
    df = _sample_df()
    out = add_elo_features(df, sport="nba").sort("start_time")
    # Match #2 home=2 away=1. Como el match #1 home=1 ganó a 2, en match #2
    # el rating away (team 1) debe ser > 1500, y home (team 2) < 1500.
    assert out["elo_away"][1] > 1500.0
    assert out["elo_home"][1] < 1500.0
