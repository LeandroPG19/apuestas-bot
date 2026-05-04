"""Tests del validator de integridad histórica."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from apuestas.validators.historical_odds_integrity import (
    HistoricalOddsRow,
    batch_validate,
    validate_odds_row,
)

START = datetime(2026, 4, 1, 20, 0, tzinfo=UTC)


def _row(
    odds: dict[str, float], *, closing: bool = False, ts_offset_min: int = -60
) -> HistoricalOddsRow:
    return HistoricalOddsRow(
        match_id=1,
        bookmaker="b365",
        market="h2h",
        outcomes_odds=odds,
        ts=START + timedelta(minutes=ts_offset_min),
        start_time=START,
        is_closing=closing,
    )


def test_valid_h2h_3way_soccer() -> None:
    row = _row({"home": 2.10, "draw": 3.40, "away": 3.50})
    assert validate_odds_row(row) == "ok"


def test_valid_h2h_2way_nba() -> None:
    row = _row({"home": 1.85, "away": 1.95})
    assert validate_odds_row(row) == "ok"


def test_invalid_odds_below_101() -> None:
    row = _row({"home": 1.00, "away": 100.0})
    assert validate_odds_row(row) == "invalid_range"


def test_invalid_odds_above_50() -> None:
    row = _row({"home": 60.0, "away": 1.01})
    assert validate_odds_row(row) == "invalid_range"


def test_invalid_overround_too_high() -> None:
    # Mercados exóticos MX con vig 25% deben rechazarse
    row = _row({"home": 1.50, "away": 1.50})  # vig ~33%
    assert validate_odds_row(row) == "invalid_overround"


def test_invalid_overround_too_low_negative() -> None:
    # Arb (vig negativo) no debería aparecer en histórico limpio
    row = _row({"home": 2.10, "away": 2.10})  # vig ~-4.76%
    assert validate_odds_row(row) == "invalid_overround"


def test_invalid_inversion_both_sides_longshot() -> None:
    # Ambos probabilidad <30% = algo mal (falta outcome draw?)
    row = _row({"home": 5.00, "away": 5.00})
    assert validate_odds_row(row) == "invalid_overround"  # overround catch first


def test_missing_odds() -> None:
    row = HistoricalOddsRow(
        match_id=1,
        bookmaker="b365",
        market="h2h",
        outcomes_odds={"home": None, "away": 2.0},  # type: ignore[dict-item]
        ts=START - timedelta(hours=1),
        start_time=START,
    )
    assert validate_odds_row(row) == "invalid_missing"


def test_empty_odds() -> None:
    row = HistoricalOddsRow(
        match_id=1,
        bookmaker="b365",
        market="h2h",
        outcomes_odds={},
        ts=START,
        start_time=START,
    )
    assert validate_odds_row(row) == "invalid_missing"


def test_closing_valid_timing() -> None:
    # Closing 10 min antes de start → ok
    row = _row({"home": 1.90, "away": 1.90}, closing=True, ts_offset_min=-10)
    assert validate_odds_row(row) == "ok"


def test_closing_invalid_timing_too_late() -> None:
    # Closing 30 segundos antes del start (muy tarde, probable error feed)
    row = _row({"home": 1.90, "away": 1.90}, closing=True, ts_offset_min=0)
    assert validate_odds_row(row) == "invalid_timing"


def test_closing_invalid_timing_too_early() -> None:
    # Closing 2 horas antes (no es closing real, es opening)
    row = _row({"home": 1.90, "away": 1.90}, closing=True, ts_offset_min=-120)
    assert validate_odds_row(row) == "invalid_timing"


def test_batch_validate_separates_valid_invalid() -> None:
    rows = [
        _row({"home": 2.10, "draw": 3.40, "away": 3.50}),  # ok
        _row({"home": 0.50, "away": 2.00}),  # invalid_range
        _row({"home": 1.85, "away": 1.85}),  # ok (vig ~8% aceptable)
        _row({"home": 1.20, "away": 1.20}),  # invalid_overround
    ]
    valid, counter = batch_validate(rows)
    assert len(valid) == 2
    assert counter["ok"] == 2
    assert counter.get("invalid_range", 0) >= 1
    assert counter.get("invalid_overround", 0) >= 1


@pytest.mark.parametrize(
    ("odds", "expected"),
    [
        ({"home": 1.91, "away": 1.91}, "ok"),  # NBA típico
        ({"home": 2.10, "draw": 3.30, "away": 3.50}, "ok"),  # soccer
        ({"over": 1.90, "under": 1.95}, "ok"),  # totals
    ],
)
def test_various_valid_markets(odds: dict[str, float], expected: str) -> None:
    row = _row(odds)
    assert validate_odds_row(row) == expected
