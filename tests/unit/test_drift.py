"""Tests de drift detection."""

from __future__ import annotations

import numpy as np

from apuestas.ml.drift import (
    cbpe_accuracy_estimate,
    classify_psi,
    detect_feature_drift,
    full_drift_report,
    population_stability_index,
)


def test_psi_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 10000)
    cur = rng.normal(0, 1, 10000)
    psi = population_stability_index(ref, cur)
    assert psi < 0.1, f"PSI de mismas distribuciones debe ser < 0.1, got {psi}"


def test_psi_shifted_distribution() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 10000)
    cur = rng.normal(2, 1, 10000)  # shift grande
    psi = population_stability_index(ref, cur)
    assert psi > 0.25, f"PSI con shift grande debe indicar drift severo, got {psi}"


def test_classify_psi_thresholds() -> None:
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "drift"
    assert classify_psi(0.30) == "severe_drift"


def test_detect_feature_drift_multicolumn() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (1000, 3))
    # Cambiar feature 2 drásticamente
    cur = rng.normal(0, 1, (1000, 3))
    cur[:, 1] = rng.normal(5, 1, 1000)
    alerts = detect_feature_drift(ref, cur, ["a", "b", "c"])
    # b (índice 1) debería ser el más alto
    assert alerts[0].feature == "b"
    assert alerts[0].severity in {"drift", "severe_drift"}


def test_cbpe_point_estimate_in_bounds() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 1000)
    point, (lo, hi) = cbpe_accuracy_estimate(p)
    assert 0.5 <= point <= 1.0  # max(p, 1-p) siempre ≥0.5
    assert lo <= point <= hi


def test_full_drift_report_recommends_retrain_on_severe() -> None:
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, (500, 2))
    cur = rng.normal(5, 1, (500, 2))  # drift severo en ambas features
    report = full_drift_report(ref, cur, ["f1", "f2"])
    assert report.needs_retrain is True
    assert any("severe_drift" in r for r in report.reasons)


def test_full_drift_report_stable() -> None:
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, (1000, 3))
    cur = rng.normal(0, 1, (1000, 3))
    report = full_drift_report(ref, cur, ["a", "b", "c"])
    assert report.overall_drift_score < 0.15
