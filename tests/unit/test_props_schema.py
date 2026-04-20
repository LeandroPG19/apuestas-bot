"""Tests del catálogo props §23.1 + PropPrediction msgspec."""

from __future__ import annotations

import msgspec
import pytest

from apuestas.schemas.props import (
    ALL_PROPS,
    BOXING_PROPS,
    MLB_PROPS,
    NBA_PROPS,
    NFL_PROPS,
    SOCCER_PROPS,
    PropCategory,
    PropDistribution,
    PropPrediction,
    get_prop,
    props_for_sport,
)


def test_all_sports_have_props() -> None:
    for sport in ("nba", "mlb", "nfl", "soccer", "boxing"):
        props = props_for_sport(sport)
        assert len(props) > 0, f"{sport} sin props"


def test_get_prop_known() -> None:
    p = get_prop("nba_points")
    assert p.sport_code == "nba"
    assert p.stat_key == "points"
    assert p.distribution == PropDistribution.NEG_BINOMIAL


def test_get_prop_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_prop("nba_nonexistent_prop")


def test_mlb_home_run_uses_monte_carlo() -> None:
    p = get_prop("mlb_home_run")
    assert p.distribution == PropDistribution.MONTE_CARLO
    assert p.category == PropCategory.BINARY
    assert p.needs_park_factor


def test_nba_pra_is_combo() -> None:
    p = get_prop("nba_pra")
    assert p.category == PropCategory.COMBO
    assert "nba_points" in p.correlated_props
    assert p.needs_minutes_projection


def test_nfl_qb_pass_yds_gamma() -> None:
    p = get_prop("nfl_qb_pass_yds")
    assert p.distribution == PropDistribution.GAMMA
    assert p.category == PropCategory.CONTINUOUS
    assert p.role == "qb"


def test_typical_lines_sorted() -> None:
    """Typical lines deben estar en orden ascendente."""
    for p in ALL_PROPS.values():
        if p.typical_lines:
            sorted_lines = sorted(p.typical_lines)
            assert list(p.typical_lines) == sorted_lines, f"{p.code} lines no ordenadas"


def test_prop_prediction_msgspec_valid() -> None:
    pred = PropPrediction(
        prop_code="nba_points",
        player_id=1,
        player_name="Test Player",
        event_id=100,
        line=24.5,
        mean=25.8,
        std=5.2,
        p_over=0.57,
        p_under=0.43,
        p_exact=None,
        p_over_lower=0.52,
        p_over_upper=0.62,
        distribution=PropDistribution.NEG_BINOMIAL,
        n_samples_training=100,
        model_name="props_v1",
        model_version="shadow",
    )
    # Serialización round-trip
    b = msgspec.json.encode(pred)
    decoded = msgspec.json.decode(b, type=PropPrediction)
    assert decoded.player_name == "Test Player"
    assert decoded.line == 24.5


def test_prop_prediction_rejects_invalid_distribution() -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            b'{"prop_code":"x","player_id":1,"player_name":"X","event_id":1,'
            b'"line":null,"mean":0,"std":0,"p_over":null,"p_under":null,'
            b'"p_exact":null,"p_over_lower":null,"p_over_upper":null,'
            b'"distribution":"INVALID_DIST","n_samples_training":0,'
            b'"model_name":"v","model_version":"s"}',
            type=PropPrediction,
        )


def test_boxing_rounds_weibull() -> None:
    p = get_prop("boxing_over_rounds")
    assert p.distribution == PropDistribution.WEIBULL


def test_counts_overview() -> None:
    """Al menos N props por deporte (smoke test)."""
    assert len(NBA_PROPS) >= 7
    assert len(MLB_PROPS) >= 5
    assert len(NFL_PROPS) >= 5
    assert len(SOCCER_PROPS) >= 3
    assert len(BOXING_PROPS) >= 2
