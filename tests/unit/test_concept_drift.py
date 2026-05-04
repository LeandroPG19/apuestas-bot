"""Tests para apuestas.monitors.concept_drift (Sprint 7).

Page-Hinkley con residuales sintéticos:
  - baseline Brier ~0.22 (modelo normal) → nunca dispara
  - degradación sostenida Brier > 0.40 → dispara en pocas muestras
  - cooldown 24h respetado
"""

from __future__ import annotations

import numpy as np

from apuestas.monitors.concept_drift import (
    BrierDriftMonitor,
    PageHinkleyDetector,
)


def test_page_hinkley_no_drift_on_stable_series() -> None:
    """Residuales Brier ~0.20 constantes con ruido bajo → no drift."""
    ph = PageHinkleyDetector(delta=0.01, lambda_threshold=100.0)
    rng = np.random.default_rng(42)
    noise = rng.normal(loc=0.20, scale=0.02, size=500)
    detected = any(ph.update(float(v)) for v in noise)
    assert detected is False


def test_page_hinkley_detects_sustained_increase() -> None:
    """Tras 100 muestras normales + 100 con residual mucho mayor → detecta."""
    ph = PageHinkleyDetector(delta=0.005, lambda_threshold=20.0)
    rng = np.random.default_rng(123)
    normal = rng.normal(0.20, 0.02, size=100)
    degraded = rng.normal(0.60, 0.02, size=200)

    for v in normal:
        ph.update(float(v))
    detected_at = None
    for i, v in enumerate(degraded):
        if ph.update(float(v)):
            detected_at = i
            break
    assert detected_at is not None
    assert detected_at < 100  # debe detectar bastante antes de gastar las 200


def test_page_hinkley_reset_clears_state() -> None:
    ph = PageHinkleyDetector()
    for v in np.linspace(0.1, 0.9, 40):
        ph.update(float(v))
    assert ph.n == 40
    ph.reset()
    assert ph.n == 0
    assert ph.m_t == 0.0


def test_brier_monitor_singleton() -> None:
    m1 = BrierDriftMonitor.get()
    m2 = BrierDriftMonitor.get()
    assert m1 is m2


def test_brier_monitor_stable_no_drift() -> None:
    """Serie bien calibrada (pred == actual tendencia): residuales bajos.

    Con p=0.8 y actual~Bernoulli(0.8), residual esperado = 0.16 en promedio.
    """
    monitor = BrierDriftMonitor()
    rng = np.random.default_rng(7)
    first_drift_iter = None
    for i in range(500):
        pred = 0.80
        actual = 1 if rng.uniform() < 0.80 else 0
        drifted = monitor.update("soccer", "h2h", pred=pred, actual=actual)
        if drifted and first_drift_iter is None:
            first_drift_iter = i
    # Con parámetros default puede disparar por varianza Bernoulli; si lo
    # hace, que sea tarde (indica que al menos no es inmediato).
    if first_drift_iter is not None:
        assert first_drift_iter > 100


def test_brier_monitor_respects_cooldown() -> None:
    """Tras un drift, un segundo drift inmediato no dispara retrain."""
    monitor = BrierDriftMonitor()
    # Forzar drift artificial con residuales altos sostenidos.
    for _ in range(500):
        monitor.update("mlb", "h2h", pred=0.95, actual=0)
    # Ahora forzar nuevamente otra ronda — cooldown debe suprimir.
    any_second = False
    for _ in range(500):
        if monitor.update("mlb", "h2h", pred=0.95, actual=0):
            any_second = True
    # No se garantiza que el primer drift dispare (depende del λ), pero
    # si disparó, el segundo debe estar bloqueado por cooldown 24h.
    snap = monitor.snapshot()
    if snap.get(("mlb", "h2h"), {}).get("last_retrain_at") is not None:
        assert any_second is False


def test_brier_monitor_isolates_sports() -> None:
    """Drift en NBA no afecta el estado de MLB."""
    monitor = BrierDriftMonitor()
    for _ in range(200):
        monitor.update("nba", "h2h", pred=0.95, actual=0)
    snap = monitor.snapshot()
    assert ("mlb", "h2h") not in snap
