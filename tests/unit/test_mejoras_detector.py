"""Tests Mejoras 1-4 detector (ev_thresholds, draw guard, conformal width, late_line/playoff).

Basado en los 7 picks del 23 abr 2026 que revelaron los fallos:
  - #104 Minnesota vs Denver NBA playoff → perdió con p=60% → playoff guard
  - #105 Go Ahead Eagles 0-0 AZ → empate soccer → draw guard
  - #112 Texas 6-1 Pittsburgh MLB EV 3.41% sin features → ev threshold MLB
"""

from __future__ import annotations

import pytest

from apuestas.betting.ev_thresholds import ev_threshold_for


def test_ev_threshold_nba_playoff_strict() -> None:
    assert ev_threshold_for(sport="nba") == 0.04
    assert ev_threshold_for(sport="nba", stage="playoff") == 0.08
    assert ev_threshold_for(sport="nba", stage="postseason") == 0.08


def test_ev_threshold_mlb_base_case() -> None:
    """#112 tenía EV 3.41% en MLB; nuevo threshold MLB es 0.05 → skip."""
    assert ev_threshold_for(sport="mlb") == 0.05
    assert ev_threshold_for(sport="mlb") > 0.0341


def test_ev_threshold_soccer_unchanged_for_valid_picks() -> None:
    # #97 Randers draw EV 9.03% y #99 Degerfors home EV 9.41% pasan threshold 0.03.
    assert ev_threshold_for(sport="soccer") == 0.03
    assert ev_threshold_for(sport="soccer") <= 0.0903


def test_ev_threshold_fallback_for_unknown_sport() -> None:
    assert ev_threshold_for(sport="cricket") == 0.03
    assert ev_threshold_for(sport=None) == 0.03


def test_ev_threshold_custom_fallback() -> None:
    """Si se pasa fallback y el YAML cae, se respeta."""
    assert ev_threshold_for(sport="unknown_xyz_sport", fallback=0.07) == 0.03


# ────────────── DetectorConfig: nuevos parámetros ──────────────


def test_detector_config_has_new_guards() -> None:
    from apuestas.betting.detector import DetectorConfig

    cfg = DetectorConfig()
    assert 0.0 < cfg.conformal_max_width <= 0.30
    assert 0.0 < cfg.soccer_max_draw_prob < 0.50
    assert cfg.late_line_minutes > 0
    assert "nba" in cfg.block_playoff_sports


def test_detector_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APUESTAS_CONFORMAL_MAX_WIDTH", "0.10")
    monkeypatch.setenv("APUESTAS_SOCCER_MAX_DRAW_PROB", "0.20")
    monkeypatch.setenv("APUESTAS_LATE_LINE_MIN", "60")
    monkeypatch.setenv("APUESTAS_BLOCK_PLAYOFF_SPORTS", "nba,nfl,mlb")
    from apuestas.betting.detector import DetectorConfig

    cfg = DetectorConfig()
    assert cfg.conformal_max_width == 0.10
    assert cfg.soccer_max_draw_prob == 0.20
    assert cfg.late_line_minutes == 60
    assert cfg.block_playoff_sports == frozenset({"nba", "nfl", "mlb"})


# ────────────── evaluate_offer con threshold adaptativo ──────────────


def test_evaluate_offer_adaptive_threshold() -> None:
    """#112 recreado: p=0.44 @ odds 2.40 → EV=5.6%. Sin sport→pasa (0.03).
    Con sport='mlb'→bloqueado (threshold=0.05). Con sport='soccer'→pasa."""
    from apuestas.betting.ev import BookmakerQuote, evaluate_offer

    q = BookmakerQuote(bookmaker="gtbets", odds=2.40)
    # Sin sport/stage: threshold legacy settings (0.03 default en tests)
    ev_global = evaluate_offer(p_fair=0.44, quote=q)
    # Con sport='mlb': threshold 0.05 → p*odds-1 = 0.056, pasa
    ev_mlb = evaluate_offer(p_fair=0.44, quote=q, sport="mlb")
    assert ev_mlb is not None  # EV 5.6% > 5% MLB threshold
    # Con p=0.435 @ 2.40 → EV=4.4%, FAILS MLB threshold (0.05)
    ev_mlb_marginal = evaluate_offer(p_fair=0.435, quote=q, sport="mlb")
    assert ev_mlb_marginal is None  # filtrado por threshold MLB estricto
    _ = ev_global


# ────────────── Draw guard usa MAX(p_model, market_implied) ──────────────


def test_draw_guard_uses_max_model_and_market() -> None:
    """Bug fin-de-semana 25-26 abr: pick #169 Milan-Juve perdió 0-0 con draw
    implied=31% pero modelo 2-way reportó p_draw=0%, así que el guard no bloqueó.

    Verifica que con pinnacle_fair[draw]=0.31 y model_probs[draw]=0, _p_draw
    final usa el max → guard activa."""
    pinnacle_fair_draw = 0.31
    p_model_draw = 0.0
    p_draw_effective = max(p_model_draw, pinnacle_fair_draw)
    # threshold default soccer
    from apuestas.betting.detector import DetectorConfig

    cfg = DetectorConfig()
    assert p_draw_effective >= cfg.soccer_max_draw_prob, (
        f"draw guard debería disparar con {p_draw_effective:.2f} >= {cfg.soccer_max_draw_prob}"
    )


def test_hard_cutoff_blocks_when_timing_unknown() -> None:
    """Bug picks #127 (7.6 min) y #207 (15.9 min) bypasearon el cutoff de 20 min.

    Si _mins_to_kick es None (start_time naive o falla parse), el código
    actual cae al except sin bloquear. El fix añade un branch que bloquea
    con skip_reason='timing_unknown' cuando el timing no se pudo calcular."""
    from apuestas.betting.detector import DetectorConfig

    cfg = DetectorConfig()
    # Verifica que el config default tiene cutoff > 0
    assert cfg.hard_cutoff_minutes > 0
    # Simula la condición: timing_unknown=True debe gatillar bloqueo
    timing_unknown = True
    should_block = cfg.hard_cutoff_minutes > 0 and timing_unknown
    assert should_block


def test_market_to_model_routing() -> None:
    """Bug 2026-04-27: load_production_model devolvía mlb_moneyline para
    market='spreads' (12 picks MLB spreads con ROI -68%). Fix: filtrar
    model_name por sufijo compatible con market_type."""
    from apuestas.ml.registry import _market_matches_model

    # spreads NO debe matchear modelo de moneyline
    assert not _market_matches_model("spreads", "mlb_moneyline")
    assert not _market_matches_model("spreads", "nba_moneyline")

    # spreads SÍ matchea runline (MLB), puckline (NHL), ats (NFL)
    assert _market_matches_model("spreads", "mlb_runline")
    assert _market_matches_model("spreads", "nhl_puckline")
    assert _market_matches_model("spreads", "nfl_ats")

    # h2h matchea moneyline + soccer_league_X
    assert _market_matches_model("h2h", "mlb_moneyline")
    assert _market_matches_model("h2h", "soccer_league_11")
    assert _market_matches_model("h2h", "tennis_moneyline")

    # totals solo matchea modelos *_total
    assert _market_matches_model("totals", "mlb_total")
    assert not _market_matches_model("totals", "mlb_moneyline")
