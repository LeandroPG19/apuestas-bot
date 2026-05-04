"""Tests unitarios para train_tennis (Elo + augmentation isolado de DB)."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from apuestas.ml.train_tennis import (
    FEATURE_SET_NAME,
    TennisTrainConfig,
    _compute_elo_history,
    build_tennis_feature_frame,
)


def _matches_fixture() -> pl.DataFrame:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(10):
        h_id = 1 + (i % 3)
        a_id = 1 + ((i + 1) % 3)
        if h_id == a_id:
            a_id = (a_id + 1) % 3 + 1
        rows.append(
            {
                "id": i + 1,
                "home_team_id": h_id,
                "away_team_id": a_id,
                "start_time": base.replace(day=1 + i),
                "home_score": 2 if (i % 2 == 0) else 1,
                "away_score": 1 if (i % 2 == 0) else 2,
                "season": "2025",
                "status": "finished",
                "surface": "hard",
            }
        )
    return pl.DataFrame(rows)


def _player_games_from_matches(matches: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in matches.iter_rows(named=True):
        hg = int(r["home_score"])
        ag = int(r["away_score"])
        rows.append(
            {
                "team_id": r["home_team_id"],
                "game_date": r["start_time"],
                "win": 1 if hg > ag else 0,
            }
        )
        rows.append(
            {
                "team_id": r["away_team_id"],
                "game_date": r["start_time"],
                "win": 1 if ag > hg else 0,
            }
        )
    return pl.DataFrame(rows)


def test_feature_set_name() -> None:
    assert FEATURE_SET_NAME == "tennis_v1"


def test_tennis_train_config_defaults() -> None:
    cfg = TennisTrainConfig(seasons=["2024", "2025"])
    assert cfg.experiment_name == "tennis_moneyline"
    assert cfg.stage == "shadow"


def test_compute_elo_history_returns_pre_match_ratings() -> None:
    matches = _matches_fixture()
    history = _compute_elo_history(matches)
    # Pre-match del primer match = baseline (_INITIAL_ELO=1500)
    first_row = matches.row(0, named=True)
    assert history[(int(first_row["home_team_id"]), first_row["id"])] == 1500.0
    assert history[(int(first_row["away_team_id"]), first_row["id"])] == 1500.0
    # Después del primer partido, el elo debe haber cambiado (no siendo 1500)
    # para los equipos que jugaron en match 1 cuando aparezcan otra vez.


def test_build_tennis_feature_frame_has_elo_columns() -> None:
    matches = _matches_fixture()
    games = _player_games_from_matches(matches)
    out = build_tennis_feature_frame(matches, games)
    assert "elo_home" in out.columns
    assert "elo_away" in out.columns
    assert "elo_diff" in out.columns
    # Al menos una rolling column exists
    assert any(c.startswith("win_roll_") and c.endswith("_home") for c in out.columns)
