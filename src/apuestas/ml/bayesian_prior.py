"""Bayesian Beta-Binomial prior para deportes con pocas muestras (plan §2.9).

Problema: al inicio de la temporada (MLB abril, NBA octubre, NFL septiembre)
los equipos tienen <20 partidos y las features rolling no alcanzan umbral
(`features_insufficient`). El detector se queda en `skip_no_model`.

Solución: Beta prior basado en la temporada anterior, actualizado con la
evidencia de la temporada actual. Cierra el gap con incertidumbre cuantificada
(retorna p_mean + CI 90%).

Uso:
    p_mean, p_low, p_up = beta_binomial_win_prob(
        wins_this_season=5, games_this_season=12,
        wins_prior_season=48, games_prior_season=82,
        shrinkage=0.25,
    )

El llamador (detector.py) decide si usar el prior como señal y debe:
  - Cap el tier de confianza a "📊 Media" (el prior añade incertidumbre).
  - Anexar soft_tag='bayesian_prior_used' para que el usuario lo sepa.

Referencia: Gelman et al., *Bayesian Data Analysis* 3rd ed. cap. 3.2.
"""

from __future__ import annotations

from scipy.stats import beta


def beta_binomial_win_prob(
    wins_this_season: int,
    games_this_season: int,
    *,
    wins_prior_season: int,
    games_prior_season: int,
    shrinkage: float = 0.25,
    ci_alpha: float = 0.10,
) -> tuple[float, float, float]:
    """Posterior Beta combinando prior (temporada anterior) + evidencia actual.

    Args:
        wins_this_season / games_this_season: registro actual (pocos games OK).
        wins_prior_season / games_prior_season: record temporada previa.
        shrinkage: peso del prior (0..1). Decae a 0 con más samples actuales.
        ci_alpha: nivel del intervalo (0.10 = CI 90%).

    Returns:
        (p_mean, p_lower, p_upper).
    """
    if games_prior_season < 0 or games_this_season < 0:
        msg = "games_* deben ser ≥ 0"
        raise ValueError(msg)
    alpha_0 = shrinkage * wins_prior_season + 0.5
    beta_0 = shrinkage * max(games_prior_season - wins_prior_season, 0) + 0.5
    alpha_post = alpha_0 + wins_this_season
    beta_post = beta_0 + max(games_this_season - wins_this_season, 0)
    p_mean = alpha_post / (alpha_post + beta_post)
    p_lower = float(beta.ppf(ci_alpha / 2, alpha_post, beta_post))
    p_upper = float(beta.ppf(1 - ci_alpha / 2, alpha_post, beta_post))
    return float(p_mean), p_lower, p_upper


def adaptive_shrinkage(games_this_season: int) -> float:
    """Decae shrinkage de 0.5 (ningún partido actual) a 0 (≥82 partidos).

    Fórmula: shrinkage = 0.5 * max(0, 1 − games/82). Con 41 games cae a 0.25,
    con 82+ cae a 0 (solo evidencia actual).
    """
    return 0.5 * max(0.0, 1.0 - games_this_season / 82)


__all__ = ["adaptive_shrinkage", "beta_binomial_win_prob"]
