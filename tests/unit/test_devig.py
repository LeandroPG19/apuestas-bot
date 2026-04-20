"""Tests de-vigging: multiplicative, power, shin."""

from __future__ import annotations

import numpy as np
import pytest

from apuestas.betting.devig import (
    consensus_fair_probs,
    devig,
    multiplicative,
    overround,
    power,
    shin,
)


def test_overround_typical_margin() -> None:
    """Liga MX 2.10/3.40/3.60 → overround ~4.8%."""
    o = overround([2.10, 3.40, 3.60])
    assert 0.04 < o < 0.06


def test_overround_raises_on_bad_odds() -> None:
    with pytest.raises(ValueError):
        overround([0.5, 2.0])


def test_multiplicative_sums_to_one() -> None:
    p = multiplicative([2.10, 3.40, 3.60])
    assert p.sum() == pytest.approx(1.0, abs=1e-10)
    assert all(0 < x < 1 for x in p)


def test_power_sums_to_one() -> None:
    p = power([2.10, 3.40, 3.60])
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_shin_sums_to_one() -> None:
    p = shin([2.10, 3.40, 3.60])
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_shin_vs_multiplicative_differ() -> None:
    """Shin produce distinto resultado que multiplicative en línea con margen."""
    odds = [2.10, 3.40, 3.60]
    mul = multiplicative(odds)
    sh = shin(odds)
    diff = np.max(np.abs(mul - sh))
    # Shin y multiplicative divergen más en mercados con mayor margen.
    # Para este mercado moderado, aceptamos diferencia >= 0.003 (ajuste Shin).
    assert diff > 0.003


def test_devig_dispatcher() -> None:
    odds = [1.91, 1.91]
    assert devig(odds, method="multiplicative").sum() == pytest.approx(1.0)
    assert devig(odds, method="power").sum() == pytest.approx(1.0, abs=1e-6)
    assert devig(odds, method="shin").sum() == pytest.approx(1.0, abs=1e-6)


def test_devig_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        devig([2.0, 2.0], method="unknown_method")  # type: ignore[arg-type]


def test_consensus_fair_probs_uses_only_sharps() -> None:
    odds_by_bm = {
        "pinnacle": [2.05, 1.85],
        "caliente": [2.00, 1.80],  # soft, debe ignorarse
        "strendus": [2.10, 1.75],
    }
    fair = consensus_fair_probs(odds_by_bm, method="shin")
    assert fair is not None
    assert fair.sum() == pytest.approx(1.0, abs=1e-6)
    # Debe ser cercano al shin del pinnacle puro
    from apuestas.betting.devig import shin as shin_direct

    pinn = shin_direct([2.05, 1.85])
    assert np.max(np.abs(fair - pinn)) < 0.01


def test_consensus_returns_none_without_sharp_books() -> None:
    odds = {"caliente": [2.00, 1.80]}
    assert consensus_fair_probs(odds) is None


def test_shin_with_two_way_market_balanced() -> None:
    """Mercado 2-way perfectamente justo (sin margen) debe dar p=0.5 cada uno."""
    # Suma < 1 → no hay margen; cae a normalización
    p = shin([2.05, 2.05])
    assert p[0] == pytest.approx(p[1], abs=1e-6)


def test_power_converges_for_heavy_favorite() -> None:
    p = power([1.10, 8.00])
    assert p[0] > 0.85  # favorito gana
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
