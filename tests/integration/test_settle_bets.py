"""Integration tests — settle_bets contra Postgres real via testcontainers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration]


async def _seed_match_and_bet(
    sessionmaker: Any,
    *,
    market: str,
    outcome: str,
    line: float | None,
    stake: float,
    odds: float,
    home_score: int,
    away_score: int,
    sport: str = "nba",
) -> int:
    async with sessionmaker() as session:
        await session.execute(
            text(
                "INSERT INTO teams(id,sport_code,name) VALUES (1,:s,'HOME'),(2,:s,'AWAY') "
                "ON CONFLICT DO NOTHING"
            ),
            {"s": sport},
        )
        now = datetime.now(tz=UTC)
        r = await session.execute(
            text(
                """INSERT INTO matches(sport_code, home_team_id, away_team_id,
                   start_time, status, home_score, away_score)
                   VALUES(:s, 1, 2, :t, 'finished', :hs, :as_)
                   RETURNING id"""
            ),
            {
                "s": sport,
                "t": now - timedelta(hours=3),
                "hs": home_score,
                "as_": away_score,
            },
        )
        match_id = int(r.scalar_one())
        r2 = await session.execute(
            text(
                """INSERT INTO bets(match_id, bookmaker, market, outcome, line,
                    stake_units, odds_placed, status, is_paper)
                   VALUES(:m,'test',:mk,:oc,:ln,:st,:od,'pending',true)
                   RETURNING id"""
            ),
            {
                "m": match_id,
                "mk": market,
                "oc": outcome,
                "ln": line,
                "st": stake,
                "od": odds,
            },
        )
        await session.commit()
        return int(r2.scalar_one())


async def test_settle_moneyline_home_win(
    integration_engine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sm = integration_engine["sessionmaker"]
    bet_id = await _seed_match_and_bet(
        sm,
        market="h2h",
        outcome="home",
        line=None,
        stake=1.0,
        odds=2.0,
        home_score=110,
        away_score=100,
    )
    # Patch session_scope usado por el flow
    monkeypatch.setattr("apuestas.flows.settle_bets.session_scope", lambda: sm.begin())
    from apuestas.flows.settle_bets import apply_settlement, load_pending_bets_with_final_match

    bets = await load_pending_bets_with_final_match.fn()
    assert len(bets) == 1
    counts = await apply_settlement.fn(bets)
    assert counts["won"] == 1
    async with sm() as session:
        r = await session.execute(
            text("SELECT status, pnl_units FROM bets WHERE id = :id"), {"id": bet_id}
        )
        status, pnl = r.one()
    assert status == "won"
    assert Decimal(str(pnl)) == Decimal("1.000")


async def test_settle_total_push_is_void(
    integration_engine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sm = integration_engine["sessionmaker"]
    bet_id = await _seed_match_and_bet(
        sm,
        market="total",
        outcome="over",
        line=210.0,
        stake=1.0,
        odds=1.9,
        home_score=105,
        away_score=105,
    )
    monkeypatch.setattr("apuestas.flows.settle_bets.session_scope", lambda: sm.begin())
    from apuestas.flows.settle_bets import apply_settlement, load_pending_bets_with_final_match

    bets = await load_pending_bets_with_final_match.fn()
    counts = await apply_settlement.fn(bets)
    assert counts["void"] == 1
    async with sm() as session:
        r = await session.execute(
            text("SELECT status, pnl_units FROM bets WHERE id = :id"), {"id": bet_id}
        )
        status, pnl = r.one()
    assert status == "void"
    assert float(pnl) == 0.0


async def test_settle_asian_handicap_quarter_halfwin(
    integration_engine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sm = integration_engine["sessionmaker"]
    # AH -0.25 home con home gana por 1: mitad win, mitad push → halfwon
    bet_id = await _seed_match_and_bet(
        sm,
        market="asian_handicap",
        outcome="home",
        line=-0.25,
        stake=1.0,
        odds=2.0,
        home_score=2,
        away_score=1,
        sport="soccer",
    )
    monkeypatch.setattr("apuestas.flows.settle_bets.session_scope", lambda: sm.begin())
    from apuestas.flows.settle_bets import apply_settlement, load_pending_bets_with_final_match

    bets = await load_pending_bets_with_final_match.fn()
    counts = await apply_settlement.fn(bets)
    assert counts.get("halfwon") == 1
    async with sm() as session:
        r = await session.execute(
            text("SELECT status, pnl_units FROM bets WHERE id = :id"), {"id": bet_id}
        )
        status, pnl = r.one()
    assert status == "halfwon"
    # 0.5 * stake * (odds - 1) = 0.5 * 1 * 1 = 0.5
    assert float(pnl) == pytest.approx(0.5, abs=1e-3)


async def test_soccer_draw_moneyline_home_loses(
    integration_engine: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sm = integration_engine["sessionmaker"]
    bet_id = await _seed_match_and_bet(
        sm,
        market="1x2",
        outcome="home",
        line=None,
        stake=1.0,
        odds=2.5,
        home_score=1,
        away_score=1,
        sport="soccer",
    )
    monkeypatch.setattr("apuestas.flows.settle_bets.session_scope", lambda: sm.begin())
    from apuestas.flows.settle_bets import apply_settlement, load_pending_bets_with_final_match

    bets = await load_pending_bets_with_final_match.fn()
    counts = await apply_settlement.fn(bets)
    assert counts["lost"] == 1
    async with sm() as session:
        r = await session.execute(text("SELECT status FROM bets WHERE id = :id"), {"id": bet_id})
    assert r.scalar_one() == "lost"
