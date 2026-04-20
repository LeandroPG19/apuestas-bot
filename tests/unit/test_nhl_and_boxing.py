"""Tests NHL Poisson bivariado + boxeo Elo + inactivity curve."""

from __future__ import annotations

from datetime import date

import pytest

from apuestas.features.boxing import (
    age_curve_adjustment,
    compute_fighter_age,
    inactivity_penalty,
)
from apuestas.features.boxing import (
    elo_update as boxing_elo_update,
)
from apuestas.features.nhl import (
    derive_market_probabilities,
    hockey_poisson_bivariate,
)

# ═══════════════════════ NHL ════════════════════════════════════════════


def test_hockey_poisson_matrix_sums_to_one() -> None:
    matrix = hockey_poisson_bivariate(lambda_home=2.8, lambda_away=2.5)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-4)


def test_hockey_poisson_higher_lambda_home_more_home_wins() -> None:
    m1 = hockey_poisson_bivariate(lambda_home=4.0, lambda_away=2.0)
    m2 = hockey_poisson_bivariate(lambda_home=2.0, lambda_away=4.0)
    p1 = derive_market_probabilities(m1)
    p2 = derive_market_probabilities(m2)
    assert p1["home_win"] > p2["home_win"]


def test_derive_market_probabilities_structure() -> None:
    matrix = hockey_poisson_bivariate(lambda_home=3.0, lambda_away=2.8)
    probs = derive_market_probabilities(matrix)
    for key in ("home_win", "away_win", "tie_regulation", "over_5.5", "over_6.5", "btts_yes"):
        assert key in probs
    assert 0 <= probs["home_win"] <= 1
    # home + away + tie = 1
    assert probs["home_win"] + probs["away_win"] + probs["tie_regulation"] == pytest.approx(
        1.0, abs=1e-3
    )


def test_over_5_5_larger_than_over_6_5() -> None:
    matrix = hockey_poisson_bivariate(lambda_home=3.2, lambda_away=2.9)
    p = derive_market_probabilities(matrix)
    assert p["over_5.5"] >= p["over_6.5"]


# ═══════════════════════ Boxeo ══════════════════════════════════════════


def test_compute_fighter_age() -> None:
    # Fighter nació 1985-06-15, pelea 2026-04-19 → ~40.8 años
    age = compute_fighter_age(date(1985, 6, 15), date(2026, 4, 19))
    assert age == pytest.approx(40.85, abs=0.1)


def test_compute_fighter_age_none_returns_default() -> None:
    assert compute_fighter_age(None, date(2026, 1, 1)) == 30.0


def test_age_curve_peak_27_30() -> None:
    assert age_curve_adjustment(27) == 1.0
    assert age_curve_adjustment(29) == 1.0
    # <25 penalty
    assert age_curve_adjustment(22) < 1.0
    # >32 penalty
    assert age_curve_adjustment(35) < 1.0


def test_age_curve_declive_post_32() -> None:
    a32 = age_curve_adjustment(32)
    a36 = age_curve_adjustment(36)
    assert a32 > a36


def test_inactivity_penalty_active() -> None:
    assert inactivity_penalty(90) == 1.0  # 3 meses = activo


def test_inactivity_penalty_long_layoff() -> None:
    assert inactivity_penalty(730) < 0.9  # 2 años → <0.9
    # 4 años → aún peor
    assert inactivity_penalty(1460) < inactivity_penalty(730)


def test_boxing_elo_update_symmetric() -> None:
    """Sum of changes = 0 (zero-sum)."""
    new_a, new_b = boxing_elo_update(rating_a=1500, rating_b=1500, outcome_a=1, k=32)
    assert new_a + new_b == pytest.approx(3000, abs=0.1)


def test_boxing_elo_draw_middle() -> None:
    """Draw → ratings casi iguales se mantienen."""
    new_a, new_b = boxing_elo_update(rating_a=1600, rating_b=1600, outcome_a=0.5, k=32)
    assert new_a == pytest.approx(1600, abs=0.1)
    assert new_b == pytest.approx(1600, abs=0.1)
