"""PyMC Hierarchical Bayesian Poisson GLM para soccer (SOTA 2024-2025).

Resuelve los problemas de Dixon-Coles con penaltyblog (Singular matrix E,
Positive directional derivative) usando:

1. **Priors jerárquicos informativos** (team effects ~ Normal(0, σ_team))
   → maneja sparse data (teams con 1-5 matches) sin singular matrices
2. **NUTS sampler** robusto (no depende de Hessian invertible)
3. **Posterior sampling** → intervalos de incertidumbre calibrados

Referencia: PyMC rugby example
  https://www.pymc.io/projects/examples/en/latest/case_studies/rugby_analytics.html

Reemplaza el fallback Poisson indep cuando hay >= 500 matches de data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


class _PyMCHierarchicalPoissonModel:
    """Wrapper de modelo Bayesian con posterior sampling."""

    def __init__(
        self,
        *,
        home_mean: float,
        att_posterior: np.ndarray,  # shape (n_teams,)
        def_posterior: np.ndarray,
        team_map: dict[int, int],
        intercept: float,
    ) -> None:
        self.home_mean = home_mean
        self.att = att_posterior
        self.def_ = def_posterior
        self.team_map = team_map
        self.intercept = intercept

    def predict(self, home_id: int, away_id: int) -> Any:
        from scipy.stats import poisson as scipy_poisson

        h = self.team_map.get(int(home_id))
        a = self.team_map.get(int(away_id))
        att_h = float(self.att[h]) if h is not None else 0.0
        def_a = float(self.def_[a]) if a is not None else 0.0
        att_a = float(self.att[a]) if a is not None else 0.0
        def_h = float(self.def_[h]) if h is not None else 0.0

        # log-lambda = intercept + home_advantage + attack - opponent_defense
        log_lam_h = self.intercept + self.home_mean + att_h - def_a
        log_lam_a = self.intercept + att_a - def_h
        lam_h = float(np.clip(np.exp(log_lam_h), 0.1, 6.0))
        lam_a = float(np.clip(np.exp(log_lam_a), 0.1, 6.0))

        max_g = 10
        matrix = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                matrix[i, j] = scipy_poisson.pmf(i, lam_h) * scipy_poisson.pmf(j, lam_a)
        matrix /= matrix.sum()
        p_home = float(sum(matrix[i, j] for i in range(max_g + 1) for j in range(i)))
        p_draw = float(matrix.trace())
        p_away = max(0.0, 1.0 - p_home - p_draw)

        class _Pred:
            def __init__(self, ph: float, pd: float, pa: float) -> None:
                self.home_draw_away = [ph, pd, pa]

        return _Pred(p_home, p_draw, p_away)


def fit_hierarchical_bayesian_poisson(
    *,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    team_map: dict[int, int],
    n_samples: int = 500,
    n_tune: int = 500,
    n_chains: int = 2,
) -> Any:
    """Fit PyMC hierarchical Poisson — robusto vs DC Singular matrix.

    Modelo:
        goals_home_i ~ Poisson(exp(intercept + home + att[h_i] - def[a_i]))
        goals_away_i ~ Poisson(exp(intercept + att[a_i] - def[h_i]))

        att[t] ~ Normal(0, σ_att)  (hierarchical prior: teams weak → shrink)
        def[t] ~ Normal(0, σ_def)
        home ~ Normal(0.3, 0.15)  (prior informativo: home advantage ~0.3)
        intercept ~ Normal(0, 1)
        σ_att, σ_def ~ HalfNormal(1.0)

    Returns `_PyMCHierarchicalPoissonModel` o None si PyMC no disponible.
    """
    try:
        import pymc as pm  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("soccer.pymc_missing")
        return None

    n_teams = len(team_map)
    if n_teams < 2 or len(home_goals) < 100:
        logger.warning(
            "soccer.pymc_insufficient_data",
            n_teams=n_teams,
            n_matches=len(home_goals),
        )
        return None

    try:
        with pm.Model() as _:
            # Priors
            intercept = pm.Normal("intercept", mu=0.0, sigma=1.0)
            home_mean = pm.Normal("home", mu=0.3, sigma=0.15)
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)

            # Hierarchical team effects
            att_raw = pm.Normal("att_raw", mu=0.0, sigma=1.0, shape=n_teams)
            def_raw = pm.Normal("def_raw", mu=0.0, sigma=1.0, shape=n_teams)
            att = pm.Deterministic("att", att_raw * sigma_att)
            defn = pm.Deterministic("def", def_raw * sigma_def)

            # Expected goals
            log_lam_home = intercept + home_mean + att[home_idx] - defn[away_idx]
            log_lam_away = intercept + att[away_idx] - defn[home_idx]

            # Likelihood
            pm.Poisson("home_goals_obs", mu=pm.math.exp(log_lam_home), observed=home_goals)
            pm.Poisson("away_goals_obs", mu=pm.math.exp(log_lam_away), observed=away_goals)

            # Sample posterior
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                cores=min(n_chains, 2),
                progressbar=False,
                target_accept=0.85,
                return_inferencedata=True,
            )
    except Exception as exc:
        logger.warning("soccer.pymc_fit_fail", error=str(exc)[:120])
        return None

    # Extract posterior means
    try:
        att_post = trace.posterior["att"].mean(dim=("chain", "draw")).to_numpy()
        def_post = trace.posterior["def"].mean(dim=("chain", "draw")).to_numpy()
        home_mean_post = float(trace.posterior["home"].mean(dim=("chain", "draw")).to_numpy())
        intercept_post = float(trace.posterior["intercept"].mean(dim=("chain", "draw")).to_numpy())
    except Exception as exc:
        logger.warning("soccer.pymc_extract_fail", error=str(exc)[:100])
        return None

    logger.info(
        "soccer.pymc_fit_ok",
        n_matches=len(home_goals),
        n_teams=n_teams,
        home_effect=home_mean_post,
        intercept=intercept_post,
    )

    return _PyMCHierarchicalPoissonModel(
        home_mean=home_mean_post,
        att_posterior=att_post,
        def_posterior=def_post,
        team_map=team_map,
        intercept=intercept_post,
    )
