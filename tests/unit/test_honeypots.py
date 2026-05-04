"""Honeypots adversariales (Gap 10 / A18).

Verifica que el detector rechaza inputs claramente corruptos en vez de
emitir picks con EV absurdo. Plan §18 anti-patterns checklist.
"""

from __future__ import annotations

import numpy as np
import pytest

from apuestas.betting.devig import overround, power, shin


def test_odds_one_flat_rejected_by_overround() -> None:
    """Odds todas exactamente 1.0 → overround inválido (ValueError)."""
    with pytest.raises(ValueError):
        overround([1.0, 1.0])


def test_power_handles_very_extreme_favorite() -> None:
    """Heavy favorito (odds 1.01) no debe romper el solver."""
    p = power([1.01, 50.0])
    assert p.sum() == pytest.approx(1.0, abs=1e-5)
    assert 0.95 < p[0] < 1.0  # favorito MUY fuerte


def test_shin_on_negative_margin_falls_back_gracefully() -> None:
    """Sum(1/odds) < 1 (no hay margen) no debe crashear."""
    p = shin([3.0, 3.0])  # implied 0.667 < 1
    assert abs(p.sum() - 1.0) < 1e-5


def test_ev_threshold_rejects_tiny_edge() -> None:
    """En evaluate_offer, edge < ev_threshold default debe devolver None."""
    from apuestas.betting.ev import BookmakerQuote, evaluate_offer

    q = BookmakerQuote(bookmaker="caliente", odds=1.50)
    # p=0.68 sobre 1.50 → EV=0.02, threshold default 0.03 → None
    assert evaluate_offer(p_fair=0.68, quote=q) is None


def test_confidence_empty_signals_returns_baja() -> None:
    from apuestas.bot.confidence import classify_confidence

    r = classify_confidence(ev_raw=0.0, p_blended=0.50, p_lower=0.40, p_upper=0.60)
    assert r.label in ("Baja", "Marginal")


def test_classify_alert_never_raises_on_missing_fields() -> None:
    """Regla: classifier es pure fn, no debe crashear con edge cases."""
    from apuestas.flows.live_scores import _classify_alert

    # line=None en totals → None retorno, no excepción
    assert (
        _classify_alert(
            market="totals", outcome="over", line=None, home_score=3, away_score=2, sport="mlb"
        )
        is None
    )


def test_backtest_metrics_bounded() -> None:
    """Brier siempre ∈ [0,1], BSS puede ser negativo pero finito."""
    from apuestas.ml.metrics import brier_score, brier_skill_score

    y = np.array([1, 0, 1, 0])
    p = np.array([0.7, 0.3, 0.8, 0.2])
    bs = brier_score(y, p)
    bss = brier_skill_score(y, p, p_climatology=0.5)
    assert 0.0 <= bs <= 1.0
    assert np.isfinite(bss)
