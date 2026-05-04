"""Dixon-Coles bivariate Poisson para soccer — Sprint 10 (Mejora #1).

Paper: Dixon & Coles 1997, "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market". Applied Statistics 46(2).

Implementa un modelo independent Poisson con corrección `ρ` para los
marcadores bajos (0-0, 1-0, 0-1, 1-1) donde la asunción de independencia
es más débil. Decay temporal ξ opcional para ponderar partidos recientes.

**Parámetros** (por liga, entrenados por MLE):
- `μ` log-base (media de goles overall)
- `α_i` ataque del equipo i (centrado en 0)
- `β_i` defensa del equipo i (centrado en 0)
- `γ` home advantage (log scale)
- `ρ` corrección low-score (típicamente negativo pequeño, ~-0.1)

**Predicción**: para (home, away), λ_home = exp(μ + α_home − β_away + γ),
λ_away = exp(μ + α_away − β_home). Probabilidad de (h_goals, a_goals):

    P(h, a) = τ(h, a, λ_H, λ_A, ρ) × Pois(h|λ_H) × Pois(a|λ_A)

donde τ corrige (0,0), (1,0), (0,1), (1,1) y = 1 en el resto.

**1X2/totals/BTTS** se integran sumando sobre la grilla de marcadores.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def _tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Corrección DC para marcadores bajos."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


@dataclass(slots=True)
class DixonColesModel:
    """Parámetros entrenados + predicción.

    Uso:
        model = DixonColesModel.fit(matches, xi=0.0018)
        probs = model.predict_1x2(home_id=10, away_id=20)
        # → {"home": 0.45, "draw": 0.28, "away": 0.27}
    """

    mu: float
    alpha: dict[int, float] = field(default_factory=dict)  # ataque por team_id
    beta: dict[int, float] = field(default_factory=dict)  # defensa por team_id
    gamma: float = 0.25  # home advantage log-scale
    rho: float = -0.1  # low-score correction
    max_goals: int = 10  # truncamiento de la grilla

    def _lambdas(self, home_id: int, away_id: int) -> tuple[float, float]:
        a_home = self.alpha.get(home_id, 0.0)
        b_home = self.beta.get(home_id, 0.0)
        a_away = self.alpha.get(away_id, 0.0)
        b_away = self.beta.get(away_id, 0.0)
        lam_h = math.exp(self.mu + a_home - b_away + self.gamma)
        lam_a = math.exp(self.mu + a_away - b_home)
        return lam_h, lam_a

    def score_matrix(self, home_id: int, away_id: int) -> np.ndarray:
        """Matriz (max_goals+1 x max_goals+1) de probabilidades P(h, a)."""
        lam_h, lam_a = self._lambdas(home_id, away_id)
        n = self.max_goals + 1
        M = np.zeros((n, n), dtype=float)
        for h in range(n):
            p_h = _poisson_pmf(h, lam_h)
            for a in range(n):
                p_a = _poisson_pmf(a, lam_a)
                M[h, a] = _tau(h, a, lam_h, lam_a, self.rho) * p_h * p_a
        total = M.sum()
        if total > 0:
            M /= total
        return M

    def predict_1x2(self, home_id: int, away_id: int) -> dict[str, float]:
        M = self.score_matrix(home_id, away_id)
        p_home = float(np.tril(M, k=-1).sum())
        p_draw = float(np.trace(M))
        p_away = float(np.triu(M, k=1).sum())
        return {"home": p_home, "draw": p_draw, "away": p_away}

    def predict_total(self, home_id: int, away_id: int, line: float) -> dict[str, float]:
        """P(total goals > line) — totals over/under."""
        M = self.score_matrix(home_id, away_id)
        n = M.shape[0]
        p_over = 0.0
        p_under = 0.0
        for h in range(n):
            for a in range(n):
                t = h + a
                if t > line:
                    p_over += M[h, a]
                elif t < line:
                    p_under += M[h, a]
                # t == line (push, line entera): ignorado
        total = p_over + p_under
        if total > 0:
            p_over /= total
            p_under /= total
        return {"over": p_over, "under": p_under}

    def predict_btts(self, home_id: int, away_id: int) -> dict[str, float]:
        """P(both teams to score)."""
        M = self.score_matrix(home_id, away_id)
        n = M.shape[0]
        p_yes = float(M[1:, 1:].sum())  # ambos ≥ 1
        p_no = float(1.0 - p_yes)
        return {"yes": p_yes, "no": p_no}

    @classmethod
    def fit(
        cls,
        matches: Iterable[dict],
        *,
        xi: float = 0.0018,
        max_goals: int = 10,
        n_iter: int = 100,
        learning_rate: float = 0.01,
    ) -> DixonColesModel:
        """MLE via gradient ascent simplificado.

        matches: iterable de dicts con {home_id, away_id, home_goals,
                 away_goals, date (opcional para decay)}.
        xi: decay per day (Dixon-Coles 1997 recomienda 0.0018).
        """
        data = [m for m in matches if _valid_match(m)]
        if not data:
            msg = "DixonColesModel.fit: sin matches válidos"
            raise ValueError(msg)

        teams = sorted({m["home_id"] for m in data} | {m["away_id"] for m in data})
        alpha = dict.fromkeys(teams, 0.0)
        beta = dict.fromkeys(teams, 0.0)
        mu = math.log(sum(m["home_goals"] + m["away_goals"] for m in data) / (2 * len(data)))
        gamma = 0.25
        rho = -0.08

        # Pesos temporales con decay
        import datetime as _dt

        now = max((m.get("date") for m in data if m.get("date")), default=None)
        if now is None:
            weights = [1.0] * len(data)
        else:
            weights = []
            for m in data:
                d = m.get("date")
                if d is None:
                    weights.append(1.0)
                    continue
                days = (now - d).days if isinstance(d, _dt.datetime) else 0
                weights.append(math.exp(-xi * max(0, days)))

        # Gradient ascent rudimentario
        for _ in range(n_iter):
            grad_alpha = dict.fromkeys(teams, 0.0)
            grad_beta = dict.fromkeys(teams, 0.0)
            grad_mu = 0.0
            grad_gamma = 0.0
            for m, w in zip(data, weights, strict=False):
                h_id = m["home_id"]
                a_id = m["away_id"]
                h_g = m["home_goals"]
                a_g = m["away_goals"]
                lam_h = math.exp(mu + alpha[h_id] - beta[a_id] + gamma)
                lam_a = math.exp(mu + alpha[a_id] - beta[h_id])
                # Gradientes Poisson clásicos (τ tratado como pseudo-constante por iter)
                g_h = w * (h_g - lam_h)
                g_a = w * (a_g - lam_a)
                grad_alpha[h_id] += g_h
                grad_beta[a_id] -= g_h
                grad_alpha[a_id] += g_a
                grad_beta[h_id] -= g_a
                grad_mu += g_h + g_a
                grad_gamma += g_h

            for t in teams:
                alpha[t] += learning_rate * grad_alpha[t] / max(len(data), 1)
                beta[t] += learning_rate * grad_beta[t] / max(len(data), 1)
            mu += learning_rate * grad_mu / (2 * max(len(data), 1))
            gamma += learning_rate * grad_gamma / max(len(data), 1)

            # Centrar α, β (identificabilidad)
            mean_a = sum(alpha.values()) / len(teams)
            mean_b = sum(beta.values()) / len(teams)
            for t in teams:
                alpha[t] -= mean_a
                beta[t] -= mean_b

        logger.info(
            "dixon_coles.fit",
            n_matches=len(data),
            n_teams=len(teams),
            mu=round(mu, 3),
            gamma=round(gamma, 3),
        )
        return cls(mu=mu, alpha=alpha, beta=beta, gamma=gamma, rho=rho, max_goals=max_goals)


def _valid_match(m: dict) -> bool:
    return (
        m.get("home_id") is not None
        and m.get("away_id") is not None
        and m.get("home_goals") is not None
        and m.get("away_goals") is not None
    )


__all__ = ["DixonColesModel"]
