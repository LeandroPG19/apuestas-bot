"""Tests para alert_store.should_emit_or_upgrade.

Usa AsyncMock para simular `AsyncSession.execute()` retornando filas
sintéticas de `pick_alerts`. Así se testea sólo la lógica de decisión
(new/upgrade/skip) sin depender de Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from apuestas.betting.alert_store import (
    UpgradeConfig,
    get_upgrade_config,
    should_emit_or_upgrade,
)


@dataclass(slots=True)
class _FakeVB:
    event_id: int = 100
    market: str = "h2h"
    outcome: str = "home"
    line: float | None = None
    bookmaker: str = "caliente"
    odds: float = 2.05
    p_blended: float = 0.55
    sport_code: str = "nba"
    ev: float = 0.05
    edge: float = 0.02


def _make_session(first_result) -> AsyncMock:
    """Construye un AsyncSession mock que retorna `first_result` al `.first()`."""
    session = AsyncMock()
    result_obj = MagicMock()
    result_obj.first = MagicMock(return_value=first_result)
    session.execute = AsyncMock(return_value=result_obj)
    return session


def _existing_row(
    *,
    id_: int = 42,
    odds_placed: float = 1.95,
    best_odds_seen: float | None = 2.00,
    bookmaker: str = "draftkings",
    upgrade_count: int = 1,
    last_alert_at: datetime | None = None,
    prediction_id: int | None = 7,
) -> MagicMock:
    row = MagicMock()
    row.id = id_
    row.odds_placed = odds_placed
    row.best_odds_seen = best_odds_seen
    row.bookmaker = bookmaker
    row.upgrade_count = upgrade_count
    row.last_alert_at = last_alert_at
    row.prediction_id = prediction_id
    return row


# ─────────────────────── Config ────────────────────────


def test_get_upgrade_config_loads_nba() -> None:
    cfg = get_upgrade_config("nba")
    assert isinstance(cfg, UpgradeConfig)
    assert cfg.ev_delta_min_pp > 0
    assert cfg.odds_delta_min > 0


def test_get_upgrade_config_unknown_sport_falls_back_to_defaults() -> None:
    cfg = get_upgrade_config("unknown-sport-xyz")
    assert isinstance(cfg, UpgradeConfig)
    # Debe coincidir con la entrada "defaults" del YAML.
    default_cfg = get_upgrade_config("defaults")
    assert cfg == default_cfg


# ─────────────────────── Decisión new ────────────────────────


@pytest.mark.asyncio
async def test_should_emit_new_when_no_existing_alert() -> None:
    session = _make_session(None)
    decision = await should_emit_or_upgrade(session, _FakeVB())
    assert decision == "new"
    session.execute.assert_called_once()


# ─────────────────────── Decisión skip ────────────────────────


@pytest.mark.asyncio
async def test_skip_when_odds_delta_below_threshold() -> None:
    # NBA: odds_delta_min=0.07. Le damos delta=0.03 → skip.
    row = _existing_row(best_odds_seen=2.00)
    session = _make_session(row)
    pick = _FakeVB(odds=2.03)
    assert await should_emit_or_upgrade(session, pick) == "skip"


@pytest.mark.asyncio
async def test_skip_when_ev_delta_below_threshold() -> None:
    """Cubre el caso donde odds suben pero EV apenas mejora.

    Construimos un pick con p_blended=0.20 para que el ΔEV quede por debajo
    del ev_delta_min (NBA=1.5pp) aun cuando Δodds supere 0.07.
    """
    row = _existing_row(best_odds_seen=1.80)
    session = _make_session(row)
    pick = _FakeVB(odds=1.88, p_blended=0.20)  # ΔEV ≈ 0.016 → 1.6pp — ~borderline
    # Con p=0.10, ΔEV=0.10 * 0.08 = 0.008 = 0.8pp < 1.5pp
    pick.p_blended = 0.10
    assert await should_emit_or_upgrade(session, pick) == "skip"


@pytest.mark.asyncio
async def test_skip_when_cooldown_active() -> None:
    """Alerta reciente (dentro del cooldown) bloquea upgrade aunque mejore."""
    # NBA cooldown=30 min. Marcamos last_alert_at = hace 5 min.
    recent = datetime.now(tz=UTC) - timedelta(minutes=5)
    row = _existing_row(best_odds_seen=1.90, last_alert_at=recent)
    session = _make_session(row)
    # Delta grande en odds (0.20) y EV (pp alto) → aun así skip por cooldown.
    pick = _FakeVB(odds=2.10, p_blended=0.55)
    assert await should_emit_or_upgrade(session, pick) == "skip"


# ─────────────────────── Decisión upgrade ────────────────────────


@pytest.mark.asyncio
async def test_upgrade_when_all_thresholds_pass_and_no_cooldown() -> None:
    # NBA: odds_delta_min=0.07, ev_delta_min_pp=1.5, cooldown=30.
    # Sin last_alert_at → cooldown no aplica.
    row = _existing_row(best_odds_seen=1.90, last_alert_at=None)
    session = _make_session(row)
    pick = _FakeVB(odds=2.10, p_blended=0.55)  # Δodds=0.20, p*Δodds=0.11 → 11pp
    assert await should_emit_or_upgrade(session, pick) == "upgrade"


@pytest.mark.asyncio
async def test_upgrade_after_cooldown_expires() -> None:
    # NBA cooldown=30 min. last_alert_at = hace 40 min → elapsed > cooldown.
    past = datetime.now(tz=UTC) - timedelta(minutes=40)
    row = _existing_row(best_odds_seen=1.95, last_alert_at=past)
    session = _make_session(row)
    pick = _FakeVB(odds=2.15, p_blended=0.55)
    assert await should_emit_or_upgrade(session, pick) == "upgrade"


@pytest.mark.asyncio
async def test_sport_without_override_uses_defaults() -> None:
    """Un deporte no listado en YAML aplica los umbrales de `defaults`.

    El YAML defaults tiene odds_delta_min=0.10. Probamos que con 0.08 skipea
    aun cuando con la config NBA (0.07) pasaría.
    """
    row = _existing_row(best_odds_seen=2.00)
    session = _make_session(row)
    pick = _FakeVB(odds=2.08, p_blended=0.55, sport_code="curling")
    # Con NBA (0.07): sería upgrade. Con defaults (0.10): skip.
    assert await should_emit_or_upgrade(session, pick) == "skip"
