"""E2E — pipeline completo con fixtures offline.

Flujo: odds sample → devig → EV → Kelly → regional compare.
No toca red ni DB real. Valida que los componentes encajan correctamente.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"

pytestmark = [pytest.mark.e2e]


def _load(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_odds_api_sample_produces_ev_decision() -> None:
    """odds sample → devig Pinnacle → EV vs DraftKings → decisión."""
    from apuestas.betting.devig import shin
    from apuestas.betting.ev import compute_ev, kelly_stake

    payload = _load("odds_api_sample.json")
    assert isinstance(payload, list)
    event = payload[0]
    pinnacle = next(b for b in event["bookmakers"] if b["key"] == "pinnacle")
    h2h = next(m for m in pinnacle["markets"] if m["key"] == "h2h")
    odds = [float(o["price"]) for o in h2h["outcomes"]]

    fair = shin(odds)
    assert abs(float(sum(fair)) - 1.0) < 1e-5
    assert all(0 < float(p) < 1 for p in fair)

    dk = next(b for b in event["bookmakers"] if b["key"] == "draftkings")
    dk_h2h = next(m for m in dk["markets"] if m["key"] == "h2h")
    home_odds_dk = next(o["price"] for o in dk_h2h["outcomes"] if o["name"] == "Boston Celtics")
    p_home = float(fair[0])
    ev = compute_ev(p=p_home, odds=home_odds_dk)
    stake_abs, kelly_pct = kelly_stake(
        p=p_home, odds=home_odds_dk, bankroll=100.0, fraction=0.25, cap_pct=0.05
    )
    assert ev == pytest.approx((p_home * home_odds_dk) - 1, abs=1e-9)
    assert 0.0 <= kelly_pct <= 0.05
    assert stake_abs == pytest.approx(kelly_pct * 100.0, abs=1e-6)


def test_regional_compare_mx_vs_us() -> None:
    """MX (Caliente/Strendus) vs US (DraftKings/FanDuel) mismo mercado."""
    from apuestas.betting.ev import BookmakerQuote
    from apuestas.betting.regional import compare_regions

    quotes = [
        BookmakerQuote(bookmaker="caliente", odds=1.95),
        BookmakerQuote(bookmaker="strendus", odds=1.92),
        BookmakerQuote(bookmaker="draftkings", odds=1.88),
        BookmakerQuote(bookmaker="fanduel", odds=1.90),
    ]
    rec = compare_regions(
        event_id=1,
        market="h2h",
        outcome="home",
        p_fair=0.55,
        quotes=quotes,
        bankroll=100.0,
    )
    assert rec.cross_recommendation in {"MX", "US", "tie", "neither"}
    assert rec.mx.best_offer is not None
    assert rec.us.best_offer is not None
    assert rec.mx.best_offer.bookmaker == "caliente"
    assert rec.us.best_offer.bookmaker == "fanduel"


def test_api_football_fixture_status_parsing() -> None:
    """Valida que fixtures JSON de API-Football se parsean correctamente."""
    payload = _load("api_football_fixtures_sample.json")
    assert isinstance(payload, dict)
    fixtures = payload["response"]
    assert len(fixtures) == 2

    ft = fixtures[0]
    assert ft["fixture"]["status"]["short"] == "FT"
    assert ft["goals"]["home"] == 2
    assert ft["goals"]["away"] == 1

    ns = fixtures[1]
    assert ns["fixture"]["status"]["short"] == "NS"
    assert ns["goals"]["home"] is None


def test_devig_power_and_multiplicative_agree_within_tolerance() -> None:
    """Sanity check: métodos de-vigging no divergen salvajemente."""
    from apuestas.betting.devig import multiplicative, power, shin

    odds = [1.71, 2.25]
    mult = multiplicative(odds)
    p = power(odds)
    s = shin(odds)

    for i in range(2):
        assert abs(float(mult[i]) - float(p[i])) < 0.02
        assert abs(float(mult[i]) - float(s[i])) < 0.03
    assert abs(float(mult.sum()) - 1.0) < 1e-6
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert abs(float(s.sum()) - 1.0) < 1e-6
