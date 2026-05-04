"""Tests betfair_exchange: fail-soft sin credenciales + constantes."""

from __future__ import annotations

import pytest

from apuestas.ingest.betfair_exchange import (
    BETFAIR_EVENT_TYPE_IDS,
    STRAIGHT_MARKET_TYPES,
    BetfairExchangeClient,
    _credentials_available,
    ingest,
)


def test_event_type_ids_cubre_deportes_core() -> None:
    for s in ("soccer", "tennis", "nba", "mlb", "nfl"):
        assert s in BETFAIR_EVENT_TYPE_IDS


def test_straight_market_types_incluye_match_odds() -> None:
    assert STRAIGHT_MARKET_TYPES["MATCH_ODDS"] == "h2h"
    assert STRAIGHT_MARKET_TYPES["OVER_UNDER_25"] == "totals"
    assert "HANDICAP" in STRAIGHT_MARKET_TYPES


def test_credentials_check_sin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)
    monkeypatch.delenv("BETFAIR_USERNAME", raising=False)
    monkeypatch.delenv("BETFAIR_PASSWORD", raising=False)
    assert _credentials_available() is False


def test_credentials_check_con_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BETFAIR_APP_KEY", "fake_key")
    monkeypatch.setenv("BETFAIR_USERNAME", "u@example.com")
    monkeypatch.setenv("BETFAIR_PASSWORD", "pw")
    assert _credentials_available() is True


@pytest.mark.asyncio
async def test_ingest_sin_credenciales_retorna_lista_vacia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-soft: sin creds no crashea, retorna []."""
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)
    monkeypatch.delenv("BETFAIR_USERNAME", raising=False)
    monkeypatch.delenv("BETFAIR_PASSWORD", raising=False)

    result = await ingest(["nba", "soccer"])
    assert result == []


@pytest.mark.asyncio
async def test_login_sin_creds_retorna_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)
    monkeypatch.delenv("BETFAIR_USERNAME", raising=False)
    monkeypatch.delenv("BETFAIR_PASSWORD", raising=False)

    client = BetfairExchangeClient()
    assert await client.login() is False


@pytest.mark.asyncio
async def test_fetch_events_sin_login_auto_intenta_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si fetch se llama sin login, intenta login una vez; si falla, retorna []."""
    monkeypatch.delenv("BETFAIR_APP_KEY", raising=False)
    client = BetfairExchangeClient()
    result = await client.fetch_events("nba")
    assert result == []
