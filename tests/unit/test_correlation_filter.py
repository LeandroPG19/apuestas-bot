"""Tests para correlation_filter — Sprint 10 Fase 1 (Mejora #3)."""

from __future__ import annotations

from apuestas.betting.correlation_filter import filter_correlated_picks


def _p(event_id: int, market: str, outcome: str, edge: float) -> dict:
    return {"event_id": event_id, "market": market, "outcome": outcome, "edge": edge}


def test_h2h_and_spread_same_side_keeps_higher_edge() -> None:
    picks = [
        _p(1, "h2h", "home", 0.03),
        _p(1, "spreads", "home", 0.05),
    ]
    kept, dropped = filter_correlated_picks(picks)
    assert len(kept) == 1
    assert kept[0]["market"] == "spreads"
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "correlated_cross_family"


def test_opposite_sides_same_match_keep_higher_edge() -> None:
    # Bot apostando home+away en el mismo match = contradicción pura
    picks = [
        _p(5, "h2h", "home", 0.02),
        _p(5, "h2h", "away", 0.06),
    ]
    kept, dropped = filter_correlated_picks(picks)
    assert len(kept) == 1
    assert kept[0]["outcome"] == "away"
    assert dropped[0]["reason"] == "same_family_lower_edge"


def test_different_events_not_filtered() -> None:
    picks = [
        _p(1, "h2h", "home", 0.03),
        _p(2, "h2h", "away", 0.04),
        _p(3, "spreads", "home", 0.05),
    ]
    kept, dropped = filter_correlated_picks(picks)
    assert len(kept) == 3
    assert dropped == []


def test_totals_over_under_contradictory_kept_one() -> None:
    picks = [
        _p(7, "totals", "over", 0.03),
        _p(7, "totals", "under", 0.04),
    ]
    kept, dropped = filter_correlated_picks(picks)
    assert len(kept) == 1
    assert kept[0]["outcome"] == "under"


def test_props_markets_pass_through() -> None:
    # Props NO se correlacionan con h2h → deben pasar todos
    picks = [
        _p(9, "h2h", "home", 0.05),
        _p(9, "player_points", "over", 0.06),
        _p(9, "player_rebounds", "over", 0.04),
    ]
    kept, dropped = filter_correlated_picks(picks)
    assert len(kept) == 3
    assert dropped == []


def test_empty_list_returns_empty() -> None:
    kept, dropped = filter_correlated_picks([])
    assert kept == []
    assert dropped == []


def test_single_pick_untouched() -> None:
    picks = [_p(1, "h2h", "home", 0.03)]
    kept, dropped = filter_correlated_picks(picks)
    assert len(kept) == 1
    assert dropped == []


def test_real_bos_nyy_case_29_vs_37() -> None:
    # Regresión del bug real del dataset 22-23 abr:
    # #29 BOS-NYY h2h/away edge 0.03 + #37 BOS-NYY spreads/home edge 0.04.
    # Son lados OPUESTOS del mismo evento. Conservar el de mayor edge.
    picks = [
        _p(100, "h2h", "away", 0.03),
        _p(100, "spreads", "home", 0.04),
    ]
    kept, dropped = filter_correlated_picks(picks)
    assert len(kept) == 2  # diferentes sides → se conservan ambos (no contradictorios por side)


def test_h2h_home_vs_spread_away_not_same_side() -> None:
    # h2h/home y spread/away NO son correlacionados positivamente — son opuestos
    picks = [
        _p(200, "h2h", "home", 0.03),
        _p(200, "spreads", "away", 0.04),
    ]
    kept, _dropped = filter_correlated_picks(picks)
    assert len(kept) == 2  # lados distintos del mismo evento → pasan ambos
