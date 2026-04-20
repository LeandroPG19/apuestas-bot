"""Tests de discrepancy metrics."""

from __future__ import annotations

import pytest

from apuestas.ml.discrepancy import compute_discrepancy


def test_perfect_prediction() -> None:
    """Predicción 0.99 y resultado 1 debe dar discrepancy baja."""
    m = compute_discrepancy(
        p_model=0.99,
        outcome_binary=1,
        ev_predicted=0.10,
        pnl_units=0.95,
        stake_units=1.0,
    )
    assert m.prediction_error < 0.05
    assert m.discrepancy_score < 0.10


def test_catastrophic_miss() -> None:
    """Predicción 0.90 y resultado 0 debe dar discrepancy alta."""
    m = compute_discrepancy(
        p_model=0.90,
        outcome_binary=0,
        ev_predicted=0.15,
        pnl_units=-1.0,
        stake_units=1.0,
    )
    assert m.prediction_error > 0.80
    assert m.discrepancy_score > 0.30


def test_ev_realized_computation() -> None:
    m = compute_discrepancy(
        p_model=0.55,
        outcome_binary=1,
        ev_predicted=0.05,
        pnl_units=0.85,
        stake_units=1.0,
    )
    assert m.ev_realized == pytest.approx(0.85, abs=1e-9)
    assert m.ev_realized_vs_predicted == pytest.approx(0.80, abs=1e-9)


def test_calibration_miss_optional() -> None:
    m = compute_discrepancy(
        p_model=0.60,
        outcome_binary=1,
        ev_predicted=0.05,
        pnl_units=1.0,
        stake_units=1.0,
    )
    assert m.calibration_miss is None
    m2 = compute_discrepancy(
        p_model=0.60,
        outcome_binary=1,
        ev_predicted=0.05,
        pnl_units=1.0,
        stake_units=1.0,
        empirical_rate_at_bucket=0.55,
    )
    assert m2.calibration_miss == pytest.approx(0.05, abs=1e-9)


def test_llm_alignment_and_shap_check() -> None:
    llm_analysis = {
        "home_team_analysis": {
            "key_injuries": [{"player": "Star Player", "severity": "out"}],
            "lineup_changes": [],
            "contextual_factors": [],
        },
        "away_team_analysis": {
            "key_injuries": [],
            "lineup_changes": [],
            "contextual_factors": [],
        },
        "matchup_context": {},
    }
    actual_events = [{"description": "Star Player was out due to injury impacting offense"}]
    shap_top5 = [
        {"feature": "rest_days_home", "value": 1, "shap": 0.3, "direction": "down"},
        {"feature": "ortg_roll_10_home", "value": 110, "shap": -0.2, "direction": "down"},
    ]
    m = compute_discrepancy(
        p_model=0.70,
        outcome_binary=0,
        ev_predicted=0.10,
        pnl_units=-1.0,
        stake_units=1.0,
        llm_analysis=llm_analysis,
        shap_top5=shap_top5,
        actual_key_events=actual_events,
    )
    assert m.llm_alignment_score is not None
    assert 0 <= m.llm_alignment_score <= 1
    assert m.shap_attribution_check is not None


def test_line_movement_assessment_correct() -> None:
    m_correct = compute_discrepancy(
        p_model=0.55,
        outcome_binary=1,
        ev_predicted=0.03,
        pnl_units=1.0,
        stake_units=1.0,
        line_movement_assessment="sharp",
        actual_line_movement_was_sharp=True,
    )
    assert m_correct.line_movement_assessment_correct is True

    m_wrong = compute_discrepancy(
        p_model=0.55,
        outcome_binary=0,
        ev_predicted=0.03,
        pnl_units=-1.0,
        stake_units=1.0,
        line_movement_assessment="sharp",
        actual_line_movement_was_sharp=False,
    )
    assert m_wrong.line_movement_assessment_correct is False
