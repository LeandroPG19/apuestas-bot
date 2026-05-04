"""Tests para `_match_resolver.resolve_or_create_match` Sprint 3.

El resolver ahora tiene:
  - Capa 1: lookup exacto por `external_id_{odds_api,nba,nhl}` si el
    ingester lo provee.
  - Capa 1b: si hit por fuzzy+ventana pero los external_id_* estaban NULL,
    se rellenan con los nuevos valores.
  - Capa 2+: fuzzy match por teams + ventana temporal.

Estos tests usan AsyncMock para simular AsyncSession.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from apuestas.ingest._match_resolver import resolve_or_create_match


def _mock_session(responses: list) -> AsyncMock:
    """Construye AsyncSession.execute mock que devuelve resultados en orden.

    Cada ítem puede ser:
      - None  → `.first()` retorna None
      - un objeto MagicMock con `.id` → `.first()` retorna ese objeto
    """
    call_iter = iter(responses)

    async def _execute(*_a, **_kw):  # type: ignore[no-untyped-def]
        try:
            value = next(call_iter)
        except StopIteration:
            value = None
        result = MagicMock()
        result.first = MagicMock(return_value=value)
        return result

    session = MagicMock()
    session.execute = AsyncMock(side_effect=_execute)
    return session


def _row(id_: int) -> MagicMock:
    row = MagicMock()
    row.id = id_
    return row


@pytest.mark.asyncio
async def test_resolves_by_external_id_odds_api_first() -> None:
    """Si se pasa external_ids['odds_api'] y la capa 1 matchea, no se
    hacen más queries fuzzy.
    """
    # Secuencia esperada:
    #   1) resolve_or_create_team(home) → select fuzzy team home
    #   2) insert team home si falla — asumimos hit
    #   3) resolve_or_create_team(away) → select fuzzy team away
    #   4) layer 1 select external_id_odds_api → HIT match_id=999
    responses = [
        _row(1),  # select team home → existe
        _row(2),  # select team away → existe
        _row(999),  # layer 1 hit
    ]
    session = _mock_session(responses)
    match_id = await resolve_or_create_match(
        session=session,
        sport_code="nba",
        home_name="Lakers",
        away_name="Celtics",
        start_time=datetime(2026, 4, 25, 19, 0, tzinfo=UTC),
        source="theoddsapi",
        external_ids={"odds_api": "abc123"},
    )
    assert match_id == 999
    # Debe haber parado antes de los INSERT de match.
    # 3 llamadas: 2 team fuzzy hits + 1 layer-1 lookup.
    assert session.execute.await_count == 3


@pytest.mark.asyncio
async def test_falls_back_to_fuzzy_window_when_external_id_misses() -> None:
    """Con external_id provisto pero NO hay match por él → cae a fuzzy
    ventana temporal. Si fuzzy hit, se llama a `_fill_null_external_ids`
    para completar columnas NULL.
    """
    responses = [
        _row(1),  # select team home → existe
        _row(2),  # select team away → existe
        None,  # layer 1 lookup → MISS (no match con ese odds_api_id)
        _row(55),  # fuzzy+window → HIT match_id=55
        None,  # _fill_null_external_ids UPDATE — no retorna rows relevantes
    ]
    session = _mock_session(responses)
    match_id = await resolve_or_create_match(
        session=session,
        sport_code="nba",
        home_name="Lakers",
        away_name="Celtics",
        start_time=datetime(2026, 4, 25, 19, 0, tzinfo=UTC),
        source="theoddsapi",
        external_ids={"odds_api": "xyz999"},
    )
    assert match_id == 55
    # Se espera la UPDATE de fill_null_external_ids al cierre.
    assert session.execute.await_count >= 5


@pytest.mark.asyncio
async def test_no_external_ids_skips_layer_1() -> None:
    """Sin `external_ids`, NO se ejecuta el select de capa 1."""
    responses = [
        _row(1),  # home team
        _row(2),  # away team
        _row(77),  # fuzzy+window hit
    ]
    session = _mock_session(responses)
    match_id = await resolve_or_create_match(
        session=session,
        sport_code="nba",
        home_name="Lakers",
        away_name="Celtics",
        start_time=datetime(2026, 4, 25, 19, 0, tzinfo=UTC),
        source="theoddsapi",
    )
    assert match_id == 77
    assert session.execute.await_count == 3


@pytest.mark.asyncio
async def test_same_team_both_sides_returns_none() -> None:
    """Defensa: si fuzzy resuelve home y away al mismo team_id (bug del
    scraper) retorna None sin ejecutar más queries.
    """
    responses = [
        _row(42),  # home
        _row(42),  # away (mismo id → error)
    ]
    session = _mock_session(responses)
    match_id = await resolve_or_create_match(
        session=session,
        sport_code="mlb",
        home_name="Yankees",
        away_name="Yankees",
        start_time=datetime(2026, 4, 25, 19, 0, tzinfo=UTC),
        source="theoddsapi",
    )
    assert match_id is None
    assert session.execute.await_count == 2
