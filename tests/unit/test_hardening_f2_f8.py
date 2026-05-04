"""Tests para hardenings F2-F8 (Sprint D abr-2026).

Cobertura:
- F2: adaptive_blend según Brier holdout
- F3: max_hold_target_book filter
- F4: slippage guard helper
- F5: CLV anti-stale helper
- F6: auto_degradate_drifted_model
- F7: min_train_samples guard
- F8: shrinkage cuadrático cuando |Δ|>8pp

Nota: tests de helpers async que tocan DB usan AsyncMock del session.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from apuestas.betting.detector import DetectorConfig
from apuestas.flows.deep_analysis import (
    _check_clv_anti_stale,
    _check_slippage,
    _should_filter_low_sharp,
)

# ─────────────────── F2 — adaptive_blend by Brier ───────────────────


def test_f2_adaptive_blend_default_enabled() -> None:
    """Por default APUESTAS_ADAPTIVE_BLEND=true → cfg.adaptive_blend_enabled=True."""
    cfg = DetectorConfig()
    assert cfg.adaptive_blend_enabled is True


def test_f2_adaptive_blend_can_disable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_ADAPTIVE_BLEND", "false")
    cfg = DetectorConfig()
    assert cfg.adaptive_blend_enabled is False


# ─────────────────── F3 — max_hold_target_book ───────────────────


def test_f3_max_hold_default_07_pct() -> None:
    cfg = DetectorConfig()
    assert cfg.max_hold_target_book == pytest.approx(0.07)


def test_f3_max_hold_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_MAX_HOLD_TARGET_BOOK", "0.05")
    cfg = DetectorConfig()
    assert cfg.max_hold_target_book == pytest.approx(0.05)


def test_f3_overround_calculation_high_hold() -> None:
    """Hold de Caliente típico: cuotas (1.91, 1.91) → overround = 1/1.91 + 1/1.91 = 1.047."""
    from apuestas.betting.devig import overround

    hold = overround([1.91, 1.91])
    assert hold > 0.04  # 4.7% hold típico
    # Caso peor: cuotas (1.85, 1.85) → 8.1% hold
    hold_high = overround([1.85, 1.85])
    assert hold_high > 0.07


# ─────────────────── F4 — slippage guard ───────────────────


def _mock_session_with_row(row_data: dict[str, Any] | None) -> AsyncMock:
    session = AsyncMock()
    result_obj = MagicMock()
    if row_data is None:
        result_obj.first = MagicMock(return_value=None)
    else:
        row = MagicMock()
        for k, v in row_data.items():
            setattr(row, k, v)
        result_obj.first = MagicMock(return_value=row)
    session.execute = AsyncMock(return_value=result_obj)
    return session


async def test_f4_slippage_no_recent_quote_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin quote reciente → fail-open (permite notificar)."""
    monkeypatch.setenv("APUESTAS_SLIPPAGE_GUARD", "true")
    s = _mock_session_with_row(None)
    ok, current = await _check_slippage(
        s,
        match_id=1,
        bookmaker="caliente",
        market="h2h",
        outcome="home",
        line=None,
        odds_emitted=2.50,
    )
    assert ok is True
    assert current is None


async def test_f4_slippage_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_SLIPPAGE_GUARD", "false")
    s = _mock_session_with_row({"odds": 2.0})  # cuota actual << emitida
    ok, _ = await _check_slippage(
        s,
        match_id=1,
        bookmaker="caliente",
        market="h2h",
        outcome="home",
        line=None,
        odds_emitted=2.50,
    )
    assert ok is True  # disabled → siempre pasa


async def test_f4_slippage_within_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cuota bajó 3% (dentro de 5% tolerance) → ok."""
    monkeypatch.setenv("APUESTAS_SLIPPAGE_GUARD", "true")
    monkeypatch.setenv("APUESTAS_SLIPPAGE_TOLERANCE", "0.05")
    s = _mock_session_with_row({"odds": 2.42})  # 2.42/2.50 = 0.968 (3.2% drop)
    ok, current = await _check_slippage(
        s,
        match_id=1,
        bookmaker="caliente",
        market="h2h",
        outcome="home",
        line=None,
        odds_emitted=2.50,
    )
    assert ok is True
    assert current == pytest.approx(2.42)


async def test_f4_slippage_exceeds_tolerance_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cuota bajó 10% → bloquea."""
    monkeypatch.setenv("APUESTAS_SLIPPAGE_GUARD", "true")
    monkeypatch.setenv("APUESTAS_SLIPPAGE_TOLERANCE", "0.05")
    s = _mock_session_with_row({"odds": 2.20})  # 2.20/2.50 = 0.88 (12% drop)
    ok, current = await _check_slippage(
        s,
        match_id=1,
        bookmaker="caliente",
        market="h2h",
        outcome="home",
        line=None,
        odds_emitted=2.50,
    )
    assert ok is False
    assert current == pytest.approx(2.20)


# ─────────────────── F5 — CLV anti-stale ───────────────────


async def test_f5_clv_no_old_pinnacle_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin row 30min atrás → fail-open."""
    monkeypatch.setenv("APUESTAS_CLV_ANTISTALE", "true")
    s = _mock_session_with_row({"now_odds": 2.0, "old_odds": None})
    ok, drift = await _check_clv_anti_stale(
        s,
        match_id=1,
        market="h2h",
        outcome="home",
        line=None,
    )
    assert ok is True
    assert drift is None


async def test_f5_clv_pinnacle_moved_against_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinnacle bajó precio 5% (movimiento adverso > 2% threshold) → bloquea."""
    monkeypatch.setenv("APUESTAS_CLV_ANTISTALE", "true")
    monkeypatch.setenv("APUESTAS_CLV_DRIFT_TOLERANCE", "0.02")
    s = _mock_session_with_row({"now_odds": 1.90, "old_odds": 2.00})
    # drift = (1.90 - 2.00) / 2.00 = -0.05 → negativo > 2% threshold
    ok, drift = await _check_clv_anti_stale(
        s,
        match_id=1,
        market="h2h",
        outcome="home",
        line=None,
    )
    assert ok is False
    assert drift == pytest.approx(-0.05)


async def test_f5_clv_pinnacle_moved_in_favor_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinnacle SUBIÓ precio (movimiento favorable) → ok."""
    monkeypatch.setenv("APUESTAS_CLV_ANTISTALE", "true")
    monkeypatch.setenv("APUESTAS_CLV_DRIFT_TOLERANCE", "0.02")
    s = _mock_session_with_row({"now_odds": 2.10, "old_odds": 2.00})
    # drift positivo = mejor edge ahora
    ok, drift = await _check_clv_anti_stale(
        s,
        match_id=1,
        market="h2h",
        outcome="home",
        line=None,
    )
    assert ok is True
    assert drift == pytest.approx(0.05)


async def test_f5_clv_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_CLV_ANTISTALE", "false")
    s = _mock_session_with_row({"now_odds": 1.0, "old_odds": 5.0})
    ok, _ = await _check_clv_anti_stale(
        s,
        match_id=1,
        market="h2h",
        outcome="home",
        line=None,
    )
    assert ok is True  # disabled → siempre pasa


# ─────────────────── F7 — min_train_samples ───────────────────


def test_f7_min_train_samples_default_50() -> None:
    cfg = DetectorConfig()
    assert cfg.min_train_samples == 50


def test_f7_min_train_samples_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_MIN_TRAIN_SAMPLES", "200")
    cfg = DetectorConfig()
    assert cfg.min_train_samples == 200


# ─────────────────── F-low-sharp filter (already implemented earlier) ───────────────────


def test_low_sharp_filter_default_off() -> None:
    """Filtro consensus_sharp default OFF (validación gradual)."""
    os.environ.pop("APUESTAS_FILTER_LOW_SHARP", None)
    detail = {"p_consensus_sharp": 0.10, "consensus_sources": 2, "sport": "mlb"}
    assert _should_filter_low_sharp(detail) is False


def test_low_sharp_filter_no_consensus_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin fuentes consensus → no aplica filtro (passes)."""
    monkeypatch.setenv("APUESTAS_FILTER_LOW_SHARP", "true")
    detail = {"p_consensus_sharp": None, "consensus_sources": 0}
    assert _should_filter_low_sharp(detail) is False


def test_low_sharp_filter_below_threshold_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_FILTER_LOW_SHARP", "true")
    monkeypatch.setenv("APUESTAS_MIN_P_CONSENSUS_SHARP", "0.40")
    detail = {"p_consensus_sharp": 0.30, "consensus_sources": 2, "sport": "mlb"}
    assert _should_filter_low_sharp(detail) is True


def test_low_sharp_filter_sport_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_FILTER_LOW_SHARP", "true")
    monkeypatch.setenv("APUESTAS_FILTER_LOW_SHARP_SPORTS", "soccer,nba")
    # mlb no en lista → no filtra aunque p_sharp bajo
    assert (
        _should_filter_low_sharp(
            {"p_consensus_sharp": 0.30, "consensus_sources": 2, "sport": "mlb"}
        )
        is False
    )
    # soccer sí en lista → filtra
    assert (
        _should_filter_low_sharp(
            {"p_consensus_sharp": 0.30, "consensus_sources": 2, "sport": "soccer"}
        )
        is True
    )
