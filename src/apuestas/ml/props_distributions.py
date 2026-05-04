"""Distribuciones estadísticas para player props.

Cada distribución implementa:
- `fit(samples)` → ajusta parámetros por MLE.
- `p_over(line)` → P(X > line).
- `p_under(line)` → P(X ≤ line).
- `p_exact(k)` → P(X = k) solo para discretas.
- `sample(n)` → genera muestras (para Monte Carlo composite).
- `quantile(q)` → percentil.

Todas las distribuciones soportan conformal intervals vía bootstrap.
Elección por tipo de prop según `schemas.props.PropDistribution`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import stats as scipy_stats


class PropDistributionProtocol(Protocol):
    def p_over(self, line: float) -> float: ...
    def p_under(self, line: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray: ...
    @property
    def mean(self) -> float: ...
    @property
    def std(self) -> float: ...


# ═══════════════════════ Poisson ════════════════════════════════════════


@dataclass(slots=True, frozen=True)
class PoissonDist:
    """X ~ Poisson(lam). Enteros ≥ 0 con var = mean."""

    lam: float

    @property
    def mean(self) -> float:
        return float(self.lam)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.lam))

    def p_exact(self, k: int) -> float:
        return float(scipy_stats.poisson.pmf(k, mu=self.lam))

    def p_over(self, line: float) -> float:
        k = int(np.floor(line))
        return float(scipy_stats.poisson.sf(k, mu=self.lam))

    def p_under(self, line: float) -> float:
        return 1.0 - self.p_over(line)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.poisson(self.lam, size=n)

    def quantile(self, q: float) -> float:
        return float(scipy_stats.poisson.ppf(q, mu=self.lam))


def fit_poisson(samples: np.ndarray) -> PoissonDist:
    lam = float(np.clip(np.mean(samples), 1e-6, None))
    return PoissonDist(lam=lam)


# ═══════════════════════ Negative Binomial ══════════════════════════════


@dataclass(slots=True, frozen=True)
class NegBinomialDist:
    """X ~ NB(n, p). Maneja overdispersion (var > mean) típica en NBA points/Ks.

    Parametrizado por (mean, dispersion) donde dispersion = var / mean.
    """

    mean_param: float
    dispersion: float  # ≥ 1; 1 ≈ Poisson

    @property
    def mean(self) -> float:
        return float(self.mean_param)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.mean_param * self.dispersion))

    @property
    def _n_p(self) -> tuple[float, float]:
        # scipy nbinom: mean = n(1-p)/p, var = n(1-p)/p^2
        var = self.mean_param * self.dispersion
        if var <= self.mean_param:
            # Cae a Poisson si dispersión inválida
            return float("inf"), 1.0
        p = self.mean_param / var
        n = self.mean_param * p / (1 - p)
        return max(n, 1e-6), max(min(p, 1 - 1e-6), 1e-6)

    def p_exact(self, k: int) -> float:
        n, p = self._n_p
        return float(scipy_stats.nbinom.pmf(k, n=n, p=p))

    def p_over(self, line: float) -> float:
        n, p = self._n_p
        k = int(np.floor(line))
        return float(scipy_stats.nbinom.sf(k, n=n, p=p))

    def p_under(self, line: float) -> float:
        return 1.0 - self.p_over(line)

    def sample(self, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        n, p = self._n_p
        return rng.negative_binomial(n, p, size=n_samples)

    def quantile(self, q: float) -> float:
        n, p = self._n_p
        return float(scipy_stats.nbinom.ppf(q, n=n, p=p))


def fit_neg_binomial(samples: np.ndarray) -> NegBinomialDist:
    mean = float(np.mean(samples))
    var = float(np.var(samples))
    if mean <= 0:
        return NegBinomialDist(mean_param=1e-6, dispersion=1.0)
    # Si var ≤ mean (underdispersion), clamp dispersion a 1.01 (casi Poisson)
    dispersion = max(var / mean, 1.01) if mean > 0 else 1.01
    return NegBinomialDist(mean_param=mean, dispersion=dispersion)


# ═══════════════════════ Gamma ══════════════════════════════════════════


@dataclass(slots=True, frozen=True)
class GammaDist:
    """X ~ Gamma(shape, scale). Para continuas positivas (NFL yardage)."""

    shape: float
    scale: float

    @property
    def mean(self) -> float:
        return float(self.shape * self.scale)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.shape) * self.scale)

    def p_over(self, line: float) -> float:
        if line <= 0:
            return 1.0
        return float(scipy_stats.gamma.sf(line, a=self.shape, scale=self.scale))

    def p_under(self, line: float) -> float:
        return 1.0 - self.p_over(line)

    def p_exact(self, k: int) -> float:
        # No aplica para continuas; retorna densidad aproximada para conveniencia
        return float(scipy_stats.gamma.pdf(k, a=self.shape, scale=self.scale))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.gamma(shape=self.shape, scale=self.scale, size=n)

    def quantile(self, q: float) -> float:
        return float(scipy_stats.gamma.ppf(q, a=self.shape, scale=self.scale))


def fit_gamma(samples: np.ndarray) -> GammaDist:
    """MLE cerrada de Gamma shape/scale."""
    positive = samples[samples > 0]
    if len(positive) < 3:
        return GammaDist(shape=1.0, scale=max(float(np.mean(samples)), 1e-3))
    try:
        # scipy floc=0 para mean-preserving
        shape, _, scale = scipy_stats.gamma.fit(positive, floc=0)
    except Exception:
        mean = float(np.mean(positive))
        var = float(np.var(positive))
        shape = mean**2 / var if var > 0 else 1.0
        scale = var / mean if mean > 0 else 1.0
    return GammaDist(shape=max(shape, 1e-3), scale=max(scale, 1e-3))


# ═══════════════════════ Normal truncada ════════════════════════════════


@dataclass(slots=True, frozen=True)
class TruncNormalDist:
    """X ~ TruncNormal(mu, sigma, lower=0). Para continuas positivas suaves."""

    mu: float
    sigma: float
    lower: float = 0.0
    upper: float = float("inf")

    @property
    def mean(self) -> float:
        a = (self.lower - self.mu) / self.sigma if self.sigma > 0 else 0.0
        b = (self.upper - self.mu) / self.sigma if self.sigma > 0 else 1.0
        return float(scipy_stats.truncnorm.mean(a, b, loc=self.mu, scale=self.sigma))

    @property
    def std(self) -> float:
        a = (self.lower - self.mu) / self.sigma if self.sigma > 0 else 0.0
        b = (self.upper - self.mu) / self.sigma if self.sigma > 0 else 1.0
        return float(scipy_stats.truncnorm.std(a, b, loc=self.mu, scale=self.sigma))

    def p_over(self, line: float) -> float:
        a = (self.lower - self.mu) / self.sigma
        b = (self.upper - self.mu) / self.sigma
        return float(scipy_stats.truncnorm.sf(line, a, b, loc=self.mu, scale=self.sigma))

    def p_under(self, line: float) -> float:
        return 1.0 - self.p_over(line)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        a = (self.lower - self.mu) / self.sigma
        b = (self.upper - self.mu) / self.sigma
        return scipy_stats.truncnorm.rvs(
            a, b, loc=self.mu, scale=self.sigma, size=n, random_state=rng
        )

    def quantile(self, q: float) -> float:
        a = (self.lower - self.mu) / self.sigma
        b = (self.upper - self.mu) / self.sigma
        return float(scipy_stats.truncnorm.ppf(q, a, b, loc=self.mu, scale=self.sigma))


def fit_trunc_normal(samples: np.ndarray, lower: float = 0.0) -> TruncNormalDist:
    mu = float(np.mean(samples))
    sigma = float(np.std(samples))
    return TruncNormalDist(mu=mu, sigma=max(sigma, 1e-3), lower=lower)


# ═══════════════════════ Bernoulli ══════════════════════════════════════


@dataclass(slots=True, frozen=True)
class BernoulliDist:
    """X ~ Bernoulli(p). Para anytime_goalscorer, double-double, KO, etc."""

    p: float

    @property
    def mean(self) -> float:
        return float(self.p)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.p * (1 - self.p)))

    def p_over(self, line: float) -> float:
        """En Bernoulli, `line` se interpreta como threshold. p_over(0.5)=P(X=1)."""
        if line < 1:
            return float(self.p)
        return 0.0

    def p_under(self, line: float) -> float:
        return 1.0 - self.p_over(line)

    def p_yes(self) -> float:
        return float(self.p)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.binomial(1, self.p, size=n)


def fit_bernoulli(samples: np.ndarray) -> BernoulliDist:
    p = float(np.clip(np.mean(samples), 1e-6, 1 - 1e-6))
    return BernoulliDist(p=p)


# ═══════════════════════ Weibull ════════════════════════════════════════


@dataclass(slots=True, frozen=True)
class WeibullDist:
    """X ~ Weibull(shape, scale). Round survival analysis en boxing/MMA."""

    shape: float
    scale: float

    @property
    def mean(self) -> float:
        import math

        return float(self.scale * math.gamma(1 + 1 / self.shape))

    @property
    def std(self) -> float:
        import math

        m2 = self.scale**2 * math.gamma(1 + 2 / self.shape)
        return float(np.sqrt(m2 - self.mean**2))

    def p_over(self, line: float) -> float:
        """Survival function S(line) = P(X > line)."""
        if line <= 0:
            return 1.0
        return float(scipy_stats.weibull_min.sf(line, c=self.shape, scale=self.scale))

    def p_under(self, line: float) -> float:
        return 1.0 - self.p_over(line)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.weibull(a=self.shape, size=n) * self.scale

    def quantile(self, q: float) -> float:
        return float(scipy_stats.weibull_min.ppf(q, c=self.shape, scale=self.scale))


def fit_weibull(samples: np.ndarray) -> WeibullDist:
    positive = samples[samples > 0]
    if len(positive) < 5:
        return WeibullDist(shape=1.0, scale=float(np.mean(samples)) or 1.0)
    shape, _, scale = scipy_stats.weibull_min.fit(positive, floc=0)
    return WeibullDist(shape=max(shape, 1e-3), scale=max(scale, 1e-3))


# ═══════════════════════ Empirical (Monte Carlo result) ════════════════


@dataclass(slots=True, frozen=True)
class EmpiricalDist:
    """Distribución empírica sobre muestras simuladas. Para MC composites."""

    samples: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples))

    def p_over(self, line: float) -> float:
        return float(np.mean(self.samples > line))

    def p_under(self, line: float) -> float:
        return float(np.mean(self.samples <= line))

    def p_exact(self, k: int) -> float:
        return float(np.mean(self.samples == k))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.choice(self.samples, size=n, replace=True)

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))


# ═══════════════════════ Conformal via bootstrap ═══════════════════════


def bootstrap_conformal_interval(
    dist: PropDistributionProtocol,
    line: float,
    n_bootstrap: int = 500,
    alpha: float = 0.1,
    sample_size: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Retorna (p_over_point, p_over_lower, p_over_upper) via bootstrap.

    Simula sample_size muestras de `dist` n_bootstrap veces; estima p_over
    por muestra; usa quantiles alpha/2 y 1-alpha/2 como intervalo.
    Alternativa ligera a MAPIE cuando sólo necesitamos intervalo para 1 line.
    """
    rng = rng or np.random.default_rng()
    p_overs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        s = dist.sample(sample_size, rng=rng)
        p_overs[i] = float(np.mean(s > line))

    p_point = dist.p_over(line)
    lower = float(np.quantile(p_overs, alpha / 2))
    upper = float(np.quantile(p_overs, 1 - alpha / 2))
    return p_point, lower, upper


# ═══════════════════════ Monte Carlo composite ═════════════════════════


def monte_carlo_plate_appearances(
    *,
    n_pa: int,
    xwoba_batter: float,
    pitcher_k_rate: float,
    pitcher_bb_rate: float,
    park_factor_hr: float = 1.0,
    wind_to_of_pct: float = 0.0,
    n_simulations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, EmpiricalDist]:
    """Simulador MLB plate-by-plate (§19.4 del plan).

    Por cada PA:
    - Primero decide outcome categórico (strikeout / walk / BIP) según tasas.
    - Si BIP, asigna probabilidad hit/HR usando xwOBA del bateador escalado
      por park_factor_hr × viento.
    - Contabiliza bases y eventos.

    Returns distribuciones empíricas para: hits, total_bases, home_runs,
    strikeouts, walks.
    """
    rng = rng or np.random.default_rng()

    # Probabilidades base por PA
    p_k = np.clip(pitcher_k_rate, 0.0, 0.95)
    p_bb = np.clip(pitcher_bb_rate, 0.0, 0.3)
    p_bip = max(1 - p_k - p_bb, 0.0)

    # xwOBA → distribución aproximada de resultados sobre BIP
    # Simplificación: xwOBA escalado define P(hit|BIP) y P(HR|BIP)
    # xwOBA típico .320 → hits ~.28, HR ~.04
    p_hit_bip = float(np.clip(xwoba_batter * 0.85, 0.05, 0.55))
    p_hr_bip_base = float(np.clip(xwoba_batter * 0.12, 0.005, 0.15))
    p_hr_bip = float(np.clip(p_hr_bip_base * park_factor_hr * (1 + wind_to_of_pct), 0.001, 0.30))

    # Ajuste: si HR > hit, recalibrar
    if p_hr_bip > p_hit_bip:
        p_hit_bip = p_hr_bip + 0.01

    # P(single|BIP), P(double|BIP), P(triple|BIP) distribución típica MLB
    # de los hits, ~65% singles, 20% doubles, 2% triples, 13% HR
    p_single_bip = (p_hit_bip - p_hr_bip) * 0.75
    p_double_bip = (p_hit_bip - p_hr_bip) * 0.22
    p_triple_bip = (p_hit_bip - p_hr_bip) * 0.03

    # Simulación vectorizada
    hits = np.zeros(n_simulations, dtype=np.int32)
    total_bases = np.zeros(n_simulations, dtype=np.int32)
    home_runs = np.zeros(n_simulations, dtype=np.int32)
    strikeouts = np.zeros(n_simulations, dtype=np.int32)
    walks = np.zeros(n_simulations, dtype=np.int32)

    for sim in range(n_simulations):
        for _ in range(n_pa):
            u1 = rng.random()
            if u1 < p_k:
                strikeouts[sim] += 1
            elif u1 < p_k + p_bb:
                walks[sim] += 1
            else:
                # BIP — segunda lotería
                u2 = rng.random()
                if u2 < p_hr_bip:
                    home_runs[sim] += 1
                    hits[sim] += 1
                    total_bases[sim] += 4
                elif u2 < p_hr_bip + p_triple_bip:
                    hits[sim] += 1
                    total_bases[sim] += 3
                elif u2 < p_hr_bip + p_triple_bip + p_double_bip:
                    hits[sim] += 1
                    total_bases[sim] += 2
                elif u2 < p_hr_bip + p_triple_bip + p_double_bip + p_single_bip:
                    hits[sim] += 1
                    total_bases[sim] += 1
                # Resto: out en juego

    return {
        "hits": EmpiricalDist(samples=hits.astype(np.float64)),
        "total_bases": EmpiricalDist(samples=total_bases.astype(np.float64)),
        "home_runs": EmpiricalDist(samples=home_runs.astype(np.float64)),
        "strikeouts": EmpiricalDist(samples=strikeouts.astype(np.float64)),
        "walks": EmpiricalDist(samples=walks.astype(np.float64)),
    }


# ═══════════════════════ Dispatcher ═════════════════════════════════════


DIST_FITTERS = {
    "poisson": fit_poisson,
    "neg_binomial": fit_neg_binomial,
    "gamma": fit_gamma,
    "normal_trunc": fit_trunc_normal,
    "bernoulli": fit_bernoulli,
    "weibull": fit_weibull,
}


def fit_distribution(samples: np.ndarray, distribution: str) -> PropDistributionProtocol:
    """Factory pattern: llama al fitter correcto según tipo."""
    fitter = DIST_FITTERS.get(distribution)
    if fitter is None:
        msg = f"Distribución no soportada: {distribution}. Usa Monte Carlo directo."
        raise ValueError(msg)
    return fitter(samples)


# ═══════════════════════ Fase 3.1 — Dixon-Coles full matrix ═════════════


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Factor de corrección Dixon-Coles para resultados bajos (0-0, 1-1, 0-1, 1-0).

    El modelo Poisson simple sobreestima empates (especialmente 0-0, 1-1) y
    subestima diferenciales bajos (0-1, 1-0). `tau` corrige.
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_simulate(
    home_strength: float,
    away_strength: float,
    *,
    max_goals: int = 10,
    rho: float = -0.1,
) -> np.ndarray:
    """Genera matriz (max_goals x max_goals) de P(home_goals=i, away_goals=j).

    `home_strength`, `away_strength` = λ, μ (expected goals para home y away).
    `rho` = parámetro DC (correlación entre marcadores bajos, típico -0.1).

    Retorna matriz normalizada (suma=1) lista para derivar cualquier prop:
      - BTTS: sum(M[i][j] for i≥1, j≥1)
      - Over 2.5: sum(M[i][j] for i+j≥3)
      - Correct score X-Y: M[X][Y]
      - Home win: sum(M[i][j] for i>j)
      - Draw: sum(M[i][i])
    """
    from scipy.stats import poisson  # type: ignore[import-untyped]

    matrix = np.zeros((max_goals, max_goals), dtype=np.float64)
    for i in range(max_goals):
        for j in range(max_goals):
            p_home = poisson.pmf(i, home_strength)
            p_away = poisson.pmf(j, away_strength)
            tau = dixon_coles_tau(i, j, home_strength, away_strength, rho)
            matrix[i, j] = p_home * p_away * tau

    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def dc_prob_btts(matrix: np.ndarray) -> float:
    """P(ambos equipos anotan) = P(home≥1 AND away≥1)."""
    return float(matrix[1:, 1:].sum())


def dc_prob_over(matrix: np.ndarray, threshold: float) -> float:
    """P(total goals > threshold). Útil para Over X.5."""
    total = 0.0
    max_goals = matrix.shape[0]
    for i in range(max_goals):
        for j in range(max_goals):
            if i + j > threshold:
                total += matrix[i, j]
    return float(total)


def dc_prob_correct_score(matrix: np.ndarray, home_goals: int, away_goals: int) -> float:
    """P(score exacto)."""
    if 0 <= home_goals < matrix.shape[0] and 0 <= away_goals < matrix.shape[1]:
        return float(matrix[home_goals, away_goals])
    return 0.0


def dc_prob_home_win(matrix: np.ndarray) -> float:
    max_goals = matrix.shape[0]
    return float(sum(matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i > j))


def dc_prob_draw(matrix: np.ndarray) -> float:
    return float(np.trace(matrix))


def dc_prob_away_win(matrix: np.ndarray) -> float:
    max_goals = matrix.shape[0]
    return float(sum(matrix[i, j] for i in range(max_goals) for j in range(max_goals) if j > i))


def dc_prob_anytime_scorer(
    matrix: np.ndarray,
    player_team: str,  # "home" | "away"
    player_share_of_team_goals: float,  # 0-1
) -> float:
    """P(player anota al menos 1 gol).

    Aproximación: P(team marca al menos 1 gol) × share del jugador.
    Más preciso sería integrar player-Poisson individual.
    """
    if player_team == "home":
        p_team_scores = 1.0 - float(matrix[0, :].sum())
    else:
        p_team_scores = 1.0 - float(matrix[:, 0].sum())
    return p_team_scores * player_share_of_team_goals


def dc_prob_double_chance(matrix: np.ndarray, variant: str) -> float:
    """variant: '1X' (home or draw) | 'X2' (draw or away) | '12' (not draw)."""
    if variant == "1X":
        return dc_prob_home_win(matrix) + dc_prob_draw(matrix)
    if variant == "X2":
        return dc_prob_draw(matrix) + dc_prob_away_win(matrix)
    if variant == "12":
        return dc_prob_home_win(matrix) + dc_prob_away_win(matrix)
    msg = f"variant debe ser '1X', 'X2' o '12', got {variant!r}"
    raise ValueError(msg)
