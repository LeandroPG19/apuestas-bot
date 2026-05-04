"""Tests para pinnacle_scraper con fixtures reales de la API guest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apuestas.ingest.pinnacle_scraper import (
    american_to_decimal,
    parse_market,
    parse_matchup,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pinnacle"


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


def test_pinnacle_sport_ids_cubren_deportes_core() -> None:
    """Los sport_ids de Pinnacle son estables — league_ids se auto-discover."""
    from apuestas.ingest.pinnacle_scraper import PINNACLE_SPORT_IDS

    for required in ("baseball", "basketball", "boxing", "football", "hockey", "soccer", "tennis"):
        assert required in PINNACLE_SPORT_IDS, f"falta sport_id para {required}"


def test_american_to_decimal_favorito_negativo() -> None:
    # -108 → 1.926 (favorito clásico -110 margen estándar sharp)
    assert abs(american_to_decimal(-108) - 1.9259) < 0.001


def test_american_to_decimal_underdog_positivo() -> None:
    assert american_to_decimal(200) == 3.0
    assert american_to_decimal(103) == 2.03


def test_parse_matchup_filtra_over_under_del_totals() -> None:
    """Pinnacle emite 'matchups' con participants=['Over','Under'] para totals.
    Esos no son matchups reales — deben filtrarse."""
    raw = _load("nba_matchups.json")
    parsed = [m for r in raw if (m := parse_matchup(r)) is not None]
    # Cualquier participant con nombre Over/Under se filtra
    for m in parsed:
        assert m.home not in ("Over", "Under")
        assert m.away not in ("Over", "Under")


def test_parse_market_moneyline_devuelve_2_outcomes() -> None:
    raw = _load("nba_markets.json")
    moneylines = [m for m in raw if m.get("type") == "moneyline"]
    if not moneylines:
        pytest.skip("no hay moneylines en fixture")
    m = moneylines[0]
    # Construir participants index fake
    participants = {}
    out = parse_market(m, participants)
    # Un moneyline válido tiene 2 prices (home+away)
    assert len(out) == 2
    assert all(o.market_type == "moneyline" for o in out)
    # outcomes
    outcomes = {o.outcome for o in out}
    assert outcomes == {"home", "away"}


def test_parse_market_total_tiene_points() -> None:
    raw = _load("nba_markets.json")
    totals = [m for m in raw if m.get("type") == "total"]
    if not totals:
        pytest.skip("no hay totales en fixture")
    out = parse_market(totals[0], {})
    assert len(out) == 2
    assert out[0].points is not None
    assert out[0].market_type == "total"
    assert {o.outcome for o in out} == {"over", "under"}


def test_parse_market_ignora_tipos_desconocidos() -> None:
    # Market con type='futures' o 'parlay' no debe generar odds
    fake = {"type": "parlay_adj", "matchupId": 1, "prices": [{"price": 100}]}
    assert parse_market(fake, {}) == []


def test_odds_decimal_rango_sportbook_real() -> None:
    """Odds Pinnacle en moneylines de MATCH (2 prices) deben ser <50.

    Moneylines con N>2 prices son futures/MVP/championship y pueden tener
    longshots +80000; esos se filtran por parse_market pero aquí validamos
    el rango antes de parsear."""
    raw = _load("nba_markets.json")
    for m in raw:
        if m.get("type") != "moneyline":
            continue
        prices = m.get("prices") or []
        # Solo match moneylines (2 prices). Futures se omiten.
        if len(prices) != 2:
            continue
        for price_obj in prices:
            p = price_obj.get("price")
            if p is None:
                continue
            decimal = american_to_decimal(int(p))
            assert 1.01 <= decimal <= 50, f"odds fuera de rango: {decimal} desde {p}"
