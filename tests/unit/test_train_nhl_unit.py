"""Tests unitarios para train_nhl (feature engineering isolado de DB)."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from apuestas.ml.train_nhl import (
    FEATURE_SET_NAME,
    NHLTrainConfig,
    _team_rolling_basic,
    build_nhl_feature_frame,
)


def _team_games_fixture() -> pl.DataFrame:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for team_id in (1, 2):
        for day in range(10):
            rows.append(
                {
                    "team_id": team_id,
                    "game_date": base.replace(day=1 + day),
                    "goals_for": 3 + (day % 4),
                    "goals_against": 2 + (day % 3),
                    "win": 1 if (day % 2 == 0) else 0,
                    "margin": 1 if (day % 2 == 0) else -1,
                }
            )
    return pl.DataFrame(rows)


def _matches_fixture() -> pl.DataFrame:
    base = datetime(2025, 1, 15, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "id": 1,
                "home_team_id": 1,
                "away_team_id": 2,
                "start_time": base,
                "home_score": 3,
                "away_score": 2,
                "season": "2024-25",
                "status": "finished",
            }
        ]
    )


def test_feature_set_name() -> None:
    assert FEATURE_SET_NAME == "nhl_basic_v1"


def test_nhl_train_config_defaults() -> None:
    cfg = NHLTrainConfig(seasons=["2024-25"])
    assert cfg.target == "moneyline"
    assert cfg.stage == "shadow"
    assert cfg.experiment_name == "nhl_moneyline"


def test_team_rolling_basic_produces_roll_columns() -> None:
    df = _team_games_fixture()
    out = _team_rolling_basic(df)
    # Debe tener rolling de 5/10/20 para goals_for
    assert any(c.startswith("goals_for_roll_") for c in out.columns)
    assert "rest_days" in out.columns
    assert "back_to_back" in out.columns


def test_build_nhl_feature_frame_joins_and_diffs() -> None:
    games = _team_games_fixture()
    matches = _matches_fixture()
    out = build_nhl_feature_frame(matches, games)
    # Debe contener columnas *_home y *_away para la match
    roll_home = [c for c in out.columns if c.endswith("_home") and "roll" in c]
    roll_away = [c for c in out.columns if c.endswith("_away") and "roll" in c]
    assert len(roll_home) > 0
    assert len(roll_away) > 0
    # Al menos un *_diff derivado
    assert any(c.endswith("_diff") for c in out.columns)
