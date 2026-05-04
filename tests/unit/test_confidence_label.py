"""Tests para la etiqueta de confianza Sprint 2.

La fórmula tiene 5 componentes con pesos que suman 1.0:

  edge_c = min(EV, 0.15) / 0.15 * 0.40
  prob_c = min(p_blended - 0.50, 0.25) / 0.25 * 0.20
  cert_c = max(0, 0.20 - (p_up - p_low)) / 0.20 * 0.15
  cal_c  = max(0, 0.08 - ECE) / 0.08 * 0.15
  cons_c = max(0, 0.08 - consensus_delta) / 0.08 * 0.10

El bug original (ev>=0.05 → "Muy alta") está cubierto por el caso
`test_low_probability_but_decent_ev_should_not_be_muy_alta`.
"""

from __future__ import annotations

import pytest

from apuestas.bot.confidence import classify_confidence


def test_score_is_bounded_between_zero_and_one() -> None:
    """Score siempre en [0, 1] aunque todas las señales sean extremas."""
    result = classify_confidence(
        ev_raw=1.0,
        p_blended=1.0,
        p_lower=0.99,
        p_upper=1.00,
        rolling_ece_30d=0.0,
        market_consensus_delta=0.0,
    )
    assert 0.0 <= result.score <= 1.0


def test_baseline_no_signal_returns_low() -> None:
    """EV=0 + p=0.50 (flip de moneda) + ECE default → Baja o Marginal."""
    result = classify_confidence(ev_raw=0.0, p_blended=0.50)
    assert result.label in ("Baja", "Marginal")
    assert result.score < 0.18


def test_low_probability_but_decent_ev_should_not_be_muy_alta() -> None:
    """Bug corregido: ev=0.05 + p=0.56 NO debe ser 'Muy alta'.

    Antes (ev_raw >= 0.05 → "Muy alta"), ahora con la fórmula
    multi-componente cae en "Alta" como máximo (depende del ECE).
    """
    result = classify_confidence(
        ev_raw=0.05,
        p_blended=0.56,
        p_lower=0.49,
        p_upper=0.63,
        rolling_ece_30d=0.05,
    )
    assert result.label != "Muy alta"


def test_all_signals_strong_yields_muy_alta() -> None:
    """EV alto + p alta + intervalo estrecho + ECE bajo + consenso → Muy alta.

    Usamos señales cercanas al máximo para garantizar score ≥ 0.75 incluso
    con algo de margen: EV=0.12 (80% del cap), p=0.72 (88% del cap en la
    parte > 0.5), intervalo muy estrecho (4pp), ECE excelente (0.01),
    consensus casi perfecto (0.01).
    """
    result = classify_confidence(
        ev_raw=0.14,
        p_blended=0.72,
        p_lower=0.70,
        p_upper=0.74,
        rolling_ece_30d=0.01,
        market_consensus_delta=0.01,
    )
    assert result.label == "Muy alta"
    assert result.score >= 0.75


def test_monotonic_in_ev() -> None:
    """Con todo lo demás igual, más EV nunca reduce el score."""
    base = classify_confidence(
        ev_raw=0.02,
        p_blended=0.60,
        p_lower=0.54,
        p_upper=0.66,
        rolling_ece_30d=0.04,
    ).score
    higher = classify_confidence(
        ev_raw=0.08,
        p_blended=0.60,
        p_lower=0.54,
        p_upper=0.66,
        rolling_ece_30d=0.04,
    ).score
    assert higher >= base


def test_monotonic_in_probability() -> None:
    """Más p_blended → más score (con resto igual)."""
    low = classify_confidence(ev_raw=0.04, p_blended=0.52).score
    high = classify_confidence(ev_raw=0.04, p_blended=0.68).score
    assert high >= low


def test_wider_interval_reduces_certainty_component() -> None:
    """Intervalo más ancho → menor score (penaliza incertidumbre)."""
    narrow = classify_confidence(
        ev_raw=0.05,
        p_blended=0.60,
        p_lower=0.58,
        p_upper=0.62,
    ).score
    wide = classify_confidence(
        ev_raw=0.05,
        p_blended=0.60,
        p_lower=0.48,
        p_upper=0.72,
    ).score
    assert narrow > wide


def test_higher_ece_reduces_score() -> None:
    """Peor calibración (ECE alto) → score menor."""
    good = classify_confidence(
        ev_raw=0.05,
        p_blended=0.60,
        rolling_ece_30d=0.02,
    ).score
    bad = classify_confidence(
        ev_raw=0.05,
        p_blended=0.60,
        rolling_ece_30d=0.08,
    ).score
    assert good > bad


def test_soft_tag_penalty_drops_tier() -> None:
    """soft_tag 'pricing_error' multiplica score por 0.80."""
    ref = classify_confidence(
        ev_raw=0.08,
        p_blended=0.62,
        p_lower=0.56,
        p_upper=0.68,
        rolling_ece_30d=0.03,
    )
    penalized = classify_confidence(
        ev_raw=0.08,
        p_blended=0.62,
        p_lower=0.56,
        p_upper=0.68,
        rolling_ece_30d=0.03,
        soft_tags=frozenset({"pricing_error"}),
    )
    assert penalized.score == pytest.approx(ref.score * 0.80, rel=1e-6)


def test_stars_consistent_with_label() -> None:
    """El número de estrellas escala con la tier."""
    muy_alta = classify_confidence(ev_raw=0.12, p_blended=0.75, rolling_ece_30d=0.01)
    baja = classify_confidence(ev_raw=0.0, p_blended=0.45)
    assert muy_alta.stars.count("⭐") >= baja.stars.count("⭐")


def test_consensus_disagreement_reduces_score() -> None:
    """Mayor market_consensus_delta → menor score (peor acuerdo sharp)."""
    agree = classify_confidence(ev_raw=0.05, p_blended=0.60, market_consensus_delta=0.0).score
    disagree = classify_confidence(
        ev_raw=0.05,
        p_blended=0.60,
        market_consensus_delta=0.10,
    ).score
    assert agree > disagree
