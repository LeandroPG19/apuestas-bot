"""Tests de distribuciones estadísticas para player props."""

from __future__ import annotations

import numpy as np
import pytest

from apuestas.ml.props_distributions import (
    BernoulliDist,
    EmpiricalDist,
    GammaDist,
    NegBinomialDist,
    PoissonDist,
    TruncNormalDist,
    WeibullDist,
    bootstrap_conformal_interval,
    fit_distribution,
    fit_gamma,
    fit_neg_binomial,
    fit_poisson,
    monte_carlo_plate_appearances,
)

# ═══════════════════════ Poisson ════════════════════════════════════════


def test_poisson_mean_std() -> None:
    d = PoissonDist(lam=5.0)
    assert d.mean == pytest.approx(5.0)
    assert d.std == pytest.approx(np.sqrt(5.0))


def test_poisson_p_over_under_complement() -> None:
    d = PoissonDist(lam=3.0)
    assert d.p_over(2.5) + d.p_under(2.5) == pytest.approx(1.0, abs=1e-6)


def test_poisson_fit_recovers_mean() -> None:
    rng = np.random.default_rng(42)
    samples = rng.poisson(4.0, size=5000)
    d = fit_poisson(samples)
    assert d.lam == pytest.approx(4.0, abs=0.1)


# ═══════════════════════ NegBinomial ════════════════════════════════════


def test_neg_binomial_overdispersion() -> None:
    """NegBin debe tener var > mean cuando dispersion > 1."""
    d = NegBinomialDist(mean_param=5.0, dispersion=2.0)
    assert d.std > np.sqrt(5.0)  # mayor que Poisson con mismo mean


def test_neg_binomial_p_over_valid() -> None:
    d = NegBinomialDist(mean_param=10.0, dispersion=1.5)
    p = d.p_over(8.5)
    assert 0 <= p <= 1


def test_fit_neg_binomial_handles_underdispersion() -> None:
    # Samples con var < mean → debe clampear dispersion a 1.01
    d = fit_neg_binomial(np.array([3, 3, 3, 4, 4, 3, 4]))
    assert d.dispersion >= 1.0


# ═══════════════════════ Gamma ══════════════════════════════════════════


def test_gamma_fit_continuous() -> None:
    rng = np.random.default_rng(7)
    samples = rng.gamma(shape=2.0, scale=50.0, size=2000)
    d = fit_gamma(samples)
    assert d.mean == pytest.approx(100.0, rel=0.15)


def test_gamma_p_over_zero() -> None:
    d = GammaDist(shape=2.0, scale=50.0)
    assert d.p_over(-10) == 1.0  # menos que 0 = todo por encima


# ═══════════════════════ Bernoulli ══════════════════════════════════════


def test_bernoulli_p_yes() -> None:
    d = BernoulliDist(p=0.3)
    assert d.p_yes() == 0.3
    assert d.mean == 0.3


def test_bernoulli_p_over_line() -> None:
    d = BernoulliDist(p=0.4)
    # line < 1 → P(X=1) = 0.4
    assert d.p_over(0.5) == pytest.approx(0.4)
    # line >= 1 → 0
    assert d.p_over(1.5) == 0.0


# ═══════════════════════ TruncNormal ════════════════════════════════════


def test_trunc_normal_positive_domain() -> None:
    d = TruncNormalDist(mu=30.0, sigma=5.0, lower=0.0)
    samples = d.sample(1000, rng=np.random.default_rng(1))
    assert (samples >= 0).all()


# ═══════════════════════ Weibull ════════════════════════════════════════


def test_weibull_survival_decreasing() -> None:
    d = WeibullDist(shape=2.0, scale=8.0)
    assert d.p_over(0) > d.p_over(5) > d.p_over(10)


# ═══════════════════════ Empirical ══════════════════════════════════════


def test_empirical_p_over_matches_data() -> None:
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    d = EmpiricalDist(samples=samples.astype(np.float64))
    assert d.p_over(5.5) == pytest.approx(0.5)


# ═══════════════════════ Dispatcher ═════════════════════════════════════


def test_fit_distribution_dispatcher() -> None:
    samples = np.random.default_rng(0).poisson(3, 500)
    d = fit_distribution(samples, "poisson")
    assert isinstance(d, PoissonDist)


def test_fit_distribution_unknown_raises() -> None:
    with pytest.raises(ValueError):
        fit_distribution(np.array([1, 2, 3]), "unknown_dist")


# ═══════════════════════ Conformal bootstrap ════════════════════════════


def test_bootstrap_conformal_interval_brackets_point() -> None:
    d = PoissonDist(lam=5.0)
    p, lo, hi = bootstrap_conformal_interval(d, line=4.5, n_bootstrap=100)
    assert lo <= p <= hi
    assert 0 <= lo <= 1 and 0 <= hi <= 1


# ═══════════════════════ Monte Carlo PA MLB ═════════════════════════════


def test_monte_carlo_plate_appearances_produces_distributions() -> None:
    dists = monte_carlo_plate_appearances(
        n_pa=4,
        xwoba_batter=0.350,
        pitcher_k_rate=0.22,
        pitcher_bb_rate=0.08,
        park_factor_hr=1.0,
        n_simulations=1000,
    )
    assert {"hits", "total_bases", "home_runs", "strikeouts", "walks"} <= dists.keys()
    # home_runs debe tener mean > 0 pero razonable (<1 en 4 PAs normal)
    assert 0.0 < dists["home_runs"].mean < 1.5
    # total_bases mean ~1-3 con xwoba .350
    assert 0.5 < dists["total_bases"].mean < 5.0


def test_monte_carlo_park_factor_increases_hr() -> None:
    """Park factor >1 → más HRs."""
    dists_low = monte_carlo_plate_appearances(
        n_pa=4,
        xwoba_batter=0.320,
        pitcher_k_rate=0.22,
        pitcher_bb_rate=0.08,
        park_factor_hr=0.8,
        n_simulations=500,
    )
    dists_high = monte_carlo_plate_appearances(
        n_pa=4,
        xwoba_batter=0.320,
        pitcher_k_rate=0.22,
        pitcher_bb_rate=0.08,
        park_factor_hr=1.3,
        n_simulations=500,
    )
    assert dists_high["home_runs"].mean > dists_low["home_runs"].mean
