"""Poisson GLM para runs MLB con park factors — Sprint 10 (Mejora #8).

Paper: Koopman & Lit 2015, "A dynamic bivariate Poisson model for analysing
and forecasting match results in the English Premier League". JRSS-A 178(1).
Extendido a MLB con park factors (Silver 2010, Fangraphs).

Diferencia clave vs LGBM genérico: runs MLB siguen distribución de conteo
(Poisson). LGBM puede aproximarla pero Poisson natural con log-link +
offset park factor captura mejor la variación por estadio (Coors Field
+15% runs, Petco Park −10%, etc.).

**Modelo**:
    log(λ_team) = β_offensive + δ_opponent_defensive + park_factor + HFA

donde:
- β_team: run-scoring rate por equipo (como atacante)
- δ_team: run-prevention rate por equipo (como defensa)
- park_factor: multiplicador log del estadio (pre-calculado)
- HFA: home advantage (~1.04x en MLB)

**Predicción moneyline**: simular marcadores (h,a) con λ_H, λ_A Poisson
independientes; P(home_wins) = Σ_{h>a} P(h,a).

**Predicción totals**: similar pero sobre total = h+a.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Park factors de referencia (Fangraphs 2023-2025 promedio).
# Multiplicador sobre runs: >1.0 pro-ofensivo, <1.0 pro-pitcher.
# Se usan como LOG factor (ln(pf)) sumado al término constante.
DEFAULT_PARK_FACTORS: dict[str, float] = {
    # Hitter-friendly
    "coors field": 1.15,
    "great american ball park": 1.08,
    "fenway park": 1.06,
    "globe life field": 1.04,
    "citizens bank park": 1.04,
    "chase field": 1.03,
    "minute maid park": 1.03,
    "yankee stadium": 1.03,
    # Neutral
    "wrigley field": 1.01,
    "guaranteed rate field": 1.00,
    "busch stadium": 1.00,
    "pnc park": 0.99,
    # Pitcher-friendly
    "tropicana field": 0.97,
    "oracle park": 0.95,
    "petco park": 0.93,
    "marlins park": 0.92,
}


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


@dataclass(slots=True)
class MLBPoissonModel:
    """Parámetros entrenados + predicción runs MLB."""

    mu: float  # log-base rate global (~log(4.5 runs/game))
    offense: dict[int, float] = field(default_factory=dict)  # team_id → β
    defense: dict[int, float] = field(default_factory=dict)  # team_id → δ
    hfa: float = 0.04  # home-field advantage log-scale (~4%)
    park_factors_log: dict[str, float] = field(default_factory=dict)  # venue_name → log(pf)
    max_runs: int = 25

    def _lambdas(
        self, home_id: int, away_id: int, venue_name: str | None = None
    ) -> tuple[float, float]:
        pf_log = 0.0
        if venue_name:
            pf_log = self.park_factors_log.get(venue_name.lower(), 0.0)
        off_h = self.offense.get(home_id, 0.0)
        def_h = self.defense.get(home_id, 0.0)
        off_a = self.offense.get(away_id, 0.0)
        def_a = self.defense.get(away_id, 0.0)
        lam_h = math.exp(self.mu + off_h + def_a + pf_log + self.hfa)
        lam_a = math.exp(self.mu + off_a + def_h + pf_log)
        return lam_h, lam_a

    def score_matrix(self, home_id: int, away_id: int, venue_name: str | None = None) -> np.ndarray:
        lam_h, lam_a = self._lambdas(home_id, away_id, venue_name)
        n = self.max_runs + 1
        M = np.zeros((n, n), dtype=float)
        for h in range(n):
            p_h = _poisson_pmf(h, lam_h)
            for a in range(n):
                p_a = _poisson_pmf(a, lam_a)
                M[h, a] = p_h * p_a
        total = M.sum()
        if total > 0:
            M /= total
        return M

    def predict_moneyline(
        self, home_id: int, away_id: int, venue_name: str | None = None
    ) -> dict[str, float]:
        """P(home wins) / P(away wins); MLB no permite empate en regla."""
        M = self.score_matrix(home_id, away_id, venue_name)
        p_home = float(np.tril(M, k=-1).sum())
        p_away = float(np.triu(M, k=1).sum())
        # Empate imposible (MLB va a extra innings) → renormalizar
        total = p_home + p_away
        if total > 0:
            p_home /= total
            p_away /= total
        return {"home": p_home, "away": p_away}

    def predict_total(
        self, home_id: int, away_id: int, line: float, venue_name: str | None = None
    ) -> dict[str, float]:
        M = self.score_matrix(home_id, away_id, venue_name)
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
        total = p_over + p_under
        if total > 0:
            p_over /= total
            p_under /= total
        return {"over": p_over, "under": p_under}

    def predict_runline(
        self,
        home_id: int,
        away_id: int,
        line: float,
        venue_name: str | None = None,
    ) -> dict[str, float]:
        """P(home cubre runline). Línea típica MLB es ±1.5."""
        M = self.score_matrix(home_id, away_id, venue_name)
        n = M.shape[0]
        p_home_cover = 0.0
        p_away_cover = 0.0
        for h in range(n):
            for a in range(n):
                margin = h - a
                if margin + line > 0:
                    p_home_cover += M[h, a]
                elif margin + line < 0:
                    p_away_cover += M[h, a]
        total = p_home_cover + p_away_cover
        if total > 0:
            p_home_cover /= total
            p_away_cover /= total
        return {"home": p_home_cover, "away": p_away_cover}

    @classmethod
    def fit(
        cls,
        matches: Iterable[dict],
        *,
        park_factors: dict[str, float] | None = None,
        n_iter: int = 100,
        learning_rate: float = 0.01,
    ) -> MLBPoissonModel:
        """MLE via gradient ascent.

        matches: iterable de dicts con {home_id, away_id, home_runs,
                 away_runs, venue_name (opcional)}.
        park_factors: mapping venue_name → multiplier; default uses MLB reference.
        """
        data = [m for m in matches if _valid(m)]
        if not data:
            msg = "MLBPoissonModel.fit: sin matches válidos"
            raise ValueError(msg)

        teams = sorted({m["home_id"] for m in data} | {m["away_id"] for m in data})
        offense = dict.fromkeys(teams, 0.0)
        defense = dict.fromkeys(teams, 0.0)
        mean_runs = sum(m["home_runs"] + m["away_runs"] for m in data) / (2 * len(data))
        mu = math.log(max(mean_runs, 0.5))
        hfa = 0.04

        pf = park_factors if park_factors is not None else DEFAULT_PARK_FACTORS
        pf_log = {k.lower(): math.log(max(v, 0.01)) for k, v in pf.items()}

        for _ in range(n_iter):
            grad_off = dict.fromkeys(teams, 0.0)
            grad_def = dict.fromkeys(teams, 0.0)
            grad_mu = 0.0
            grad_hfa = 0.0
            for m in data:
                h_id = m["home_id"]
                a_id = m["away_id"]
                venue = (m.get("venue_name") or "").lower()
                pfl = pf_log.get(venue, 0.0)
                lam_h = math.exp(mu + offense[h_id] + defense[a_id] + pfl + hfa)
                lam_a = math.exp(mu + offense[a_id] + defense[h_id] + pfl)
                g_h = m["home_runs"] - lam_h
                g_a = m["away_runs"] - lam_a
                grad_off[h_id] += g_h
                grad_def[a_id] += g_h
                grad_off[a_id] += g_a
                grad_def[h_id] += g_a
                grad_mu += g_h + g_a
                grad_hfa += g_h

            n = max(len(data), 1)
            for t in teams:
                offense[t] += learning_rate * grad_off[t] / n
                defense[t] += learning_rate * grad_def[t] / n
            mu += learning_rate * grad_mu / (2 * n)
            hfa += learning_rate * grad_hfa / n

            # Centrado (identificabilidad)
            mean_off = sum(offense.values()) / len(teams)
            mean_def = sum(defense.values()) / len(teams)
            for t in teams:
                offense[t] -= mean_off
                defense[t] -= mean_def

        logger.info(
            "mlb_poisson.fit",
            n_matches=len(data),
            n_teams=len(teams),
            mu=round(mu, 3),
            hfa=round(hfa, 3),
            park_factors=len(pf_log),
        )
        return cls(
            mu=mu,
            offense=offense,
            defense=defense,
            hfa=hfa,
            park_factors_log=pf_log,
        )


class MLBPoissonSklearnWrapper:
    """Wrapper sklearn-compatible para usar Poisson como L0 en stacker.

    Espera `X` como ndarray con columnas específicas (home_id, away_id,
    venue_name_hash opcional). El wrapper extrae IDs, llama al MLE y
    devuelve `predict_proba` binario para moneyline (home_win).

    Uso en train_mlb con stacker:
        estimator = MLBPoissonSklearnWrapper(target='moneyline')
        estimator.fit(X, y, matches=matches_raw)  # matches_raw para fit MLE
        probs = estimator.predict_proba(X_test)[:, 1]  # p_home
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        target: str = "moneyline",
        n_iter: int = 100,
        learning_rate: float = 0.01,
    ) -> None:
        self.target = target
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.model_: MLBPoissonModel | None = None
        self.classes_ = np.array([0, 1])

    def __sklearn_tags__(self) -> object:
        from sklearn.utils._tags import ClassifierTags, Tags

        return Tags(
            estimator_type="classifier",
            classifier_tags=ClassifierTags(),
            target_tags=None,
            transformer_tags=None,
            regressor_tags=None,
        )

    def fit(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        matches: list[dict] | None = None,
    ) -> MLBPoissonSklearnWrapper:
        """Ajusta el modelo MLE.

        Si `matches` se provee directamente (list[dict]) lo usa; si no,
        espera que X tenga formato (home_id, away_id, home_runs, away_runs,
        venue_name) — último caso útil para pipelines automáticos.
        """
        if matches is None and X is not None:
            matches = [
                {
                    "home_id": int(row[0]),
                    "away_id": int(row[1]),
                    "home_runs": int(row[2]),
                    "away_runs": int(row[3]),
                    "venue_name": None,
                }
                for row in X
            ]
        if not matches:
            msg = "MLBPoissonSklearnWrapper.fit: matches vacío"
            raise ValueError(msg)
        self.model_ = MLBPoissonModel.fit(
            matches, n_iter=self.n_iter, learning_rate=self.learning_rate
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLBPoissonSklearnWrapper.fit debe invocarse antes")
        out = np.zeros((len(X), 2), dtype=float)
        for i, row in enumerate(X):
            home_id = int(row[0])
            away_id = int(row[1])
            venue = None
            if len(row) >= 5 and isinstance(row[4], str):
                venue = row[4]
            probs = self.model_.predict_moneyline(home_id, away_id, venue_name=venue)
            out[i, 1] = probs["home"]
            out[i, 0] = probs["away"]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "target": self.target,
            "n_iter": self.n_iter,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, **params: object) -> MLBPoissonSklearnWrapper:
        for k, v in params.items():
            setattr(self, k, v)
        return self


def _valid(m: dict) -> bool:
    return (
        m.get("home_id") is not None
        and m.get("away_id") is not None
        and m.get("home_runs") is not None
        and m.get("away_runs") is not None
    )


__all__ = ["DEFAULT_PARK_FACTORS", "MLBPoissonModel", "MLBPoissonSklearnWrapper"]
