"""Tests para los enrich helpers de deep_analysis (Deuda closure).

`_enrich_with_consensus`, `_enrich_with_regional`, `_enrich_with_weather`
usan session mockeada para probar la lógica sin DB real.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from apuestas.flows import deep_analysis


def _mock_session(rows_by_call: list) -> AsyncMock:
    """Retorna un AsyncSession mock cuyo execute().first()/all() devuelven
    los valores de `rows_by_call` en orden.
    """
    call_iter = iter(rows_by_call)

    async def _execute(*_a, **_kw):  # type: ignore[no-untyped-def]
        try:
            value = next(call_iter)
        except StopIteration:
            value = None
        result = MagicMock()
        if isinstance(value, list):
            result.all = MagicMock(return_value=value)
            result.first = MagicMock(return_value=value[0] if value else None)
        else:
            result.first = MagicMock(return_value=value)
            result.all = MagicMock(return_value=[value] if value else [])
        return result

    session = MagicMock()
    session.execute = AsyncMock(side_effect=_execute)
    return session


# ───────────────────── _enrich_with_weather ─────────────────────


@pytest.mark.asyncio
async def test_weather_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_WEATHER", "false")
    session = _mock_session([])
    detail = {"match_id": 1, "sport": "mlb"}
    out = await deep_analysis._enrich_with_weather(session, detail)
    assert "weather_summary" not in out


@pytest.mark.asyncio
async def test_weather_noop_for_indoor_sport() -> None:
    session = _mock_session([])
    # NBA no dispara weather porque no está en la lista (mlb/nfl/soccer).
    detail = {"match_id": 1, "sport": "nba"}
    out = await deep_analysis._enrich_with_weather(session, detail)
    assert "weather_summary" not in out


@pytest.mark.asyncio
async def test_weather_noop_when_no_forecast() -> None:
    # fetch_match_weather_bucket devuelve None cuando no hay forecast.
    session = _mock_session([None])  # la query SELECT retorna None
    detail = {"match_id": 1, "sport": "mlb"}
    os.environ.pop("ENABLE_WEATHER", None)
    out = await deep_analysis._enrich_with_weather(session, detail)
    assert "weather_summary" not in out


# ───────────────────── _enrich_with_regional ─────────────────────


@pytest.mark.asyncio
async def test_regional_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_REGIONAL", "false")
    session = _mock_session([])
    detail = {"match_id": 1, "market": "h2h", "outcome": "home", "p_blended": 0.55}
    out = await deep_analysis._enrich_with_regional(session, detail)
    assert "regional" not in out


@pytest.mark.asyncio
async def test_regional_noop_when_insufficient_quotes() -> None:
    # Solo 1 quote → no alcanza threshold `< 2`
    row = MagicMock()
    row.bookmaker = "caliente"
    row.odds = 1.95
    row.line = None
    session = _mock_session([[row]])
    detail = {
        "match_id": 1,
        "market": "h2h",
        "outcome": "home",
        "p_blended": 0.55,
    }
    os.environ.pop("ENABLE_REGIONAL", None)
    out = await deep_analysis._enrich_with_regional(session, detail)
    assert "regional" not in out


# ───────────────────── _enrich_with_consensus ─────────────────────


@pytest.mark.asyncio
async def test_consensus_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_CONSENSUS", "false")
    session = _mock_session([])
    detail = {"match_id": 1, "p_blended": 0.60, "p_pinnacle_fair": 0.58}
    out = await deep_analysis._enrich_with_consensus(session, detail)
    assert "p_consensus_sharp" not in out


@pytest.mark.asyncio
async def test_consensus_skip_without_p_blended() -> None:
    session = _mock_session([])
    detail = {"match_id": 1}  # sin p_blended ni p_pinnacle_fair
    os.environ.pop("ENABLE_CONSENSUS", None)
    out = await deep_analysis._enrich_with_consensus(session, detail)
    assert "p_consensus_sharp" not in out


@pytest.mark.asyncio
async def test_consensus_skip_when_no_data() -> None:
    # El primer query SELECT (matches meta) retorna None → compute no arranca.
    session = _mock_session([None])
    detail = {"match_id": 1, "p_blended": 0.60, "p_pinnacle_fair": 0.58}
    os.environ.pop("ENABLE_CONSENSUS", None)
    out = await deep_analysis._enrich_with_consensus(session, detail)
    # No peta aunque no haya datos.
    assert isinstance(out, dict)
