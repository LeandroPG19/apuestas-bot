"""Tests para `apuestas.features.runtime_features.build_match_features_from_raw`.

Mockea la query SQL (`_fetch_history`) para inyectar matches sintéticos y
verifica:
- Shape correcto del vector `(len(feature_names),)`.
- Las feature_names del modelo aparecen como columnas del DataFrame intermedio
  (acoplamiento estricto con el pipeline de training: cero skew).
- Política fail-safe: <_MIN_TEAM_HISTORY matches → None.
- Sport no soportado → None.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from apuestas.features import runtime_features


def _make_history(
    *, n_matches: int, home_id: int = 1001, away_id: int = 1002
) -> list[dict[str, Any]]:
    """Genera N matches finished alternando home/away entre los 2 teams + un
    rival sparring para que el rolling tenga muestras de contexto."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed=42)
    sparring = 9999  # rival neutro adicional
    for i in range(n_matches):
        ts = base + timedelta(days=i * 2)
        # Alternar quién es home: home_id, away_id y sparring
        if i % 3 == 0:
            h, a = home_id, sparring
        elif i % 3 == 1:
            h, a = sparring, away_id
        else:
            h, a = home_id, away_id
        hs = int(rng.integers(85, 130))
        as_ = int(rng.integers(85, 130))
        rows.append(
            {
                "id": i + 1,
                "external_id": f"ext_{i}",
                "home_team_id": h,
                "away_team_id": a,
                "start_time": ts,
                "venue_id": None,
                "home_score": hs,
                "away_score": as_,
                "status": "finished",
                "season": "2025-26",
            }
        )
    return rows


async def test_unsupported_sport_returns_none() -> None:
    out = await runtime_features.build_match_features_from_raw(
        sport_code="hockey_random",
        home_team_id=1,
        away_team_id=2,
        match_start=datetime(2026, 4, 25, tzinfo=UTC),
        feature_names=["foo", "bar"],
        use_cache=False,
    )
    assert out is None


async def test_empty_feature_names_returns_none() -> None:
    out = await runtime_features.build_match_features_from_raw(
        sport_code="nba",
        home_team_id=1,
        away_team_id=2,
        match_start=datetime(2026, 4, 25, tzinfo=UTC),
        feature_names=[],
        use_cache=False,
    )
    assert out is None


async def _stub_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass DB para resolución canónica en tests (identity passthrough)."""

    async def _fake_canonical(team_id: int, sport_code: str) -> int:
        return team_id

    monkeypatch.setattr(runtime_features, "_resolve_canonical_team_id", _fake_canonical)


async def test_insufficient_history_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solo 2 matches → bajo el umbral _MIN_TEAM_HISTORY (5) → None."""
    await _stub_canonical(monkeypatch)

    async def _fake_fetch(**kwargs: Any) -> list[dict[str, Any]]:
        return _make_history(n_matches=2)

    monkeypatch.setattr(runtime_features, "_fetch_history", _fake_fetch)

    out = await runtime_features.build_match_features_from_raw(
        sport_code="nba",
        home_team_id=1001,
        away_team_id=1002,
        match_start=datetime(2026, 4, 25, tzinfo=UTC),
        feature_names=["pts_for_avg_roll_5_home", "pts_for_avg_roll_5_away"],
        use_cache=False,
    )
    assert out is None


async def test_nba_returns_correct_shape_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipeline raw produce las columnas esperadas + shape correcto."""
    await _stub_canonical(monkeypatch)
    history = _make_history(n_matches=30, home_id=1001, away_id=1002)

    async def _fake_fetch(**kwargs: Any) -> list[dict[str, Any]]:
        return history

    monkeypatch.setattr(runtime_features, "_fetch_history", _fake_fetch)

    # Feature names que `build_nba_feature_frame` produce de hecho:
    # (rolling sobre win_margin/total_points + diff + elo)
    feature_names = [
        "win_margin_roll_5_home",
        "win_margin_roll_10_home",
        "win_margin_roll_5_away",
        "win_margin_roll_10_away",
        "total_points_roll_5_home",
        "total_points_roll_5_away",
        "elo_home",
        "elo_away",
        "elo_diff",
        "elo_p_home",
        "rest_days_home",
        "rest_days_away",
    ]

    out = await runtime_features.build_match_features_from_raw(
        sport_code="nba",
        home_team_id=1001,
        away_team_id=1002,
        match_start=datetime(2026, 4, 25, tzinfo=UTC),
        feature_names=feature_names,
        use_cache=False,
    )
    assert out is not None
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(feature_names),)
    assert out.dtype == np.float64
    # Sanity: elo_p_home debe estar en [0, 1] (logistic-Elo)
    p_home_idx = feature_names.index("elo_p_home")
    assert 0.0 <= out[p_home_idx] <= 1.0
    # elo_home/elo_away deben ser != 0 (EloBuilder iniciliza ~1500 + decay)
    elo_home = out[feature_names.index("elo_home")]
    assert elo_home > 0.0


async def test_mlb_returns_correct_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLB pipeline (build_mlb_feature_frame) corre y produce vector con shape."""
    await _stub_canonical(monkeypatch)
    history = _make_history(n_matches=40, home_id=2001, away_id=2002)

    async def _fake_fetch(**kwargs: Any) -> list[dict[str, Any]]:
        return history

    monkeypatch.setattr(runtime_features, "_fetch_history", _fake_fetch)

    feature_names = [
        "runs_scored_roll_5_home",
        "runs_scored_roll_10_home",
        "runs_scored_roll_5_away",
        "runs_allowed_roll_5_home",
        "runs_allowed_roll_5_away",
        "elo_home",
        "elo_away",
    ]

    out = await runtime_features.build_match_features_from_raw(
        sport_code="mlb",
        home_team_id=2001,
        away_team_id=2002,
        match_start=datetime(2026, 4, 25, tzinfo=UTC),
        feature_names=feature_names,
        use_cache=False,
    )
    assert out is not None
    assert out.shape == (len(feature_names),)


async def test_low_coverage_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si pides 100 features inventadas, coverage < 0.5 → None."""
    await _stub_canonical(monkeypatch)
    history = _make_history(n_matches=30)

    async def _fake_fetch(**kwargs: Any) -> list[dict[str, Any]]:
        return history

    monkeypatch.setattr(runtime_features, "_fetch_history", _fake_fetch)

    fake_features = [f"nonexistent_feature_{i}" for i in range(100)]
    out = await runtime_features.build_match_features_from_raw(
        sport_code="nba",
        home_team_id=1001,
        away_team_id=1002,
        match_start=datetime(2026, 4, 25, tzinfo=UTC),
        feature_names=fake_features,
        use_cache=False,
    )
    assert out is None


def test_feature_set_hash_stable() -> None:
    """Hash es determinístico para mismos features y sensible al orden."""
    h1 = runtime_features._feature_set_hash(["a", "b", "c"])
    h2 = runtime_features._feature_set_hash(["a", "b", "c"])
    h3 = runtime_features._feature_set_hash(["c", "b", "a"])
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 12
