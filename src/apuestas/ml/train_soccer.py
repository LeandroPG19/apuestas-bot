"""Pipeline entrenamiento fútbol con Dixon-Coles (§6).

Estrategia dual:
1. Dixon-Coles (penaltyblog) para P(home_goals, away_goals) → 1X2/totals/BTTS/AH.
2. LightGBM stacker sobre residuos DC + features xG/form/Elo.

Liga MX prioritario (§22), Big-5 europea secundaria.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cloudpickle
import mlflow
import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SoccerTrainConfig:
    league_id: int
    seasons: list[str]
    n_trials: int = 20
    random_state: int = 42
    stage: str = "shadow"
    experiment_name: str = "soccer_liga_mx"
    xi_decay: float = 0.0018  # Dixon-Coles decay per day (blueprint §6)


async def load_soccer_data(league_id: int, seasons: list[str]) -> list[dict[str, Any]]:
    """Carga matches finalizados de la liga."""
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT m.home_team_id AS home_id, m.away_team_id AS away_id,
                       m.home_score AS home_goals, m.away_score AS away_goals,
                       m.start_time AS date
                FROM matches m
                WHERE m.sport_code = 'soccer'
                  AND m.league_id = :lid
                  AND m.season = ANY(:seasons)
                  AND m.status = 'finished'
                ORDER BY m.start_time
                """
            ),
            {"lid": league_id, "seasons": seasons},
        )
        return [dict(r._mapping) for r in result.all()]


def fit_dixon_coles(
    matches: list[dict[str, Any]],
    xi: float = 0.0018,
    bayesian_priors: dict[int, tuple[float, float]] | None = None,
) -> Any:
    """Entrena Dixon-Coles con penaltyblog.

    Si DC optimizer falla (common con multi-season sparse), fallback a
    Poisson independiente usando Bayesian priors si están disponibles.
    """
    try:
        from penaltyblog.models import DixonColesGoalModel
    except ImportError:
        logger.error("soccer.penaltyblog_missing")
        return None

    # Preparar DataFrame
    df = pl.DataFrame(matches)
    if df.height == 0:
        return None

    # penaltyblog Cython requires WRITABLE numpy arrays + team IDs contiguos
    # 0..N-1 (no PKs globales de la DB). Remapeamos para que optimizer converja.
    raw_home = df["home_id"].to_numpy().astype(np.int64, copy=True)
    raw_away = df["away_id"].to_numpy().astype(np.int64, copy=True)
    unique_teams = sorted(set(raw_home.tolist()) | set(raw_away.tolist()))
    team_map = {t: i for i, t in enumerate(unique_teams)}
    home = np.ascontiguousarray(np.array([team_map[t] for t in raw_home], dtype=np.int64))
    away = np.ascontiguousarray(np.array([team_map[t] for t in raw_away], dtype=np.int64))
    home_goals = np.ascontiguousarray(df["home_goals"].to_numpy().astype(np.int64, copy=True))
    away_goals = np.ascontiguousarray(df["away_goals"].to_numpy().astype(np.int64, copy=True))
    # Weights exponenciales por decay. datetimes de postgres vienen con tz pero
    # al pasar por polars/numpy se pueden convertir a naive — normalizamos.
    today = datetime.now(tz=UTC)
    dates = df["date"].to_numpy()

    def _to_aware_utc(d: Any) -> datetime:
        if isinstance(d, datetime):
            return d if d.tzinfo is not None else d.replace(tzinfo=UTC)
        # numpy datetime64 → python datetime UTC
        return datetime.fromisoformat(str(d)[:19]).replace(tzinfo=UTC)

    days_ago = np.array([(today - _to_aware_utc(d)).total_seconds() / 86400 for d in dates])
    weights = np.ascontiguousarray(np.exp(-xi * days_ago).astype(np.float64, copy=True))

    # Multi-season con xi exponencial degenera el Hessiano (optimizer falla
    # "Positive directional derivative"). Estrategia tiered: numerical gradient
    # primero; si falla, uniform weights; finalmente sample más reciente.
    fit_strategies = [
        ("numerical", weights, False),
        ("uniform_weights", np.ones_like(weights), False),
        ("analytical_uniform", np.ones_like(weights), True),
    ]
    inner = None
    for label, w, use_grad in fit_strategies:
        try:
            inner = DixonColesGoalModel(home_goals, away_goals, home, away, w)
            inner.fit(
                use_gradient=use_grad,
                minimizer_options={"maxiter": 500, "ftol": 1e-6},
            )
            logger.info("soccer.dc_fit_ok", n_matches=len(matches), strategy=label)
            break
        except Exception as exc:
            logger.info("soccer.dc_strategy_fail", strategy=label, error=str(exc)[:80])
            inner = None

    if inner is None:
        logger.warning(
            "soccer.dc_all_strategies_failed_trying_pymc_bayesian",
            n_matches=len(matches),
        )
        # Fallback SOTA v3: PyMC hierarchical Bayesian Poisson (Kull 2024).
        # Resuelve Singular Matrix del DC usando priors jerárquicos + NUTS sampler.
        try:
            from apuestas.ml.soccer_bayesian_pymc import fit_hierarchical_bayesian_poisson

            bayesian_model = fit_hierarchical_bayesian_poisson(
                home_goals=home_goals,
                away_goals=away_goals,
                home_idx=home,
                away_idx=away,
                team_map=team_map,
                n_samples=500,
                n_tune=500,
                n_chains=2,
            )
            if bayesian_model is not None:
                logger.info("soccer.using_pymc_bayesian")
                return bayesian_model
        except Exception as exc:
            logger.warning("soccer.pymc_bayesian_fail", error=str(exc)[:120])

        logger.warning(
            "soccer.pymc_failed_falling_back_to_poisson_v2",
            n_matches=len(matches),
        )
        # Fallback v2: Poisson con home_advantage + time-decay + tau grid-search.
        return _fit_independent_poisson(
            home_goals=home_goals,
            away_goals=away_goals,
            home=home,
            away=away,
            team_map=team_map,
            bayesian_priors=bayesian_priors,
            days_ago=days_ago,
            xi_decay=xi,
        )

    try:
        # Wrap con team_map para que predict() acepte IDs globales de DB
        class _DCModelWithMap:
            def __init__(self, inner_model: Any, tmap: dict[int, int]) -> None:
                self.inner = inner_model
                self.team_map = tmap

            def predict(self, home_id: int, away_id: int) -> Any:
                h = self.team_map.get(int(home_id))
                a = self.team_map.get(int(away_id))
                if h is None or a is None:
                    raise ValueError("Both teams must have been in the training data.")
                return self.inner.predict(h, a)

        return _DCModelWithMap(inner, team_map)
    except Exception as exc:
        logger.exception("soccer.dc_fit_failed", error=str(exc))
        return None


class _IndependentPoissonModel:
    """Fallback robusto cuando Dixon-Coles no converge.

    Mejoras v3:
      - Home advantage factor explícito (λ_home × HAF)
      - Low-score correction tau (Dixon-Coles style) para empates 0-0/1-1
      - Recent-form decay (time-weighted recent games count 3× vs old)
      - Form adjustment: últimos 5 partidos goals_for/against por team
    """

    def __init__(
        self,
        *,
        attack: np.ndarray,
        defense: np.ndarray,
        lg_avg_home: float,
        lg_avg_away: float,
        team_map: dict[int, int],
        home_advantage: float = 1.0,
        tau: float = 0.0,
        form_attack: np.ndarray | None = None,
        form_defense: np.ndarray | None = None,
        form_weight: float = 0.3,
    ) -> None:
        self.attack = attack
        self.defense = defense
        self.lg_avg_home = lg_avg_home
        self.lg_avg_away = lg_avg_away
        self.team_map = team_map
        self.home_advantage = home_advantage
        self.tau = tau  # Dixon-Coles low-score correction parameter
        self.form_attack = form_attack
        self.form_defense = form_defense
        self.form_weight = form_weight

    def _dc_tau(self, i: int, j: int, lam_h: float, lam_a: float) -> float:
        """Dixon-Coles tau adjustment para scores 0-0, 0-1, 1-0, 1-1.

        `tau` se añadió en v2 — pickled models v1 no lo tienen.
        """
        tau = float(getattr(self, "tau", 0.0))
        if tau == 0.0:
            return 1.0
        if i == 0 and j == 0:
            return 1.0 - lam_h * lam_a * tau
        if i == 0 and j == 1:
            return 1.0 + lam_h * tau
        if i == 1 and j == 0:
            return 1.0 + lam_a * tau
        if i == 1 and j == 1:
            return 1.0 - tau
        return 1.0

    def predict(self, home_id: int, away_id: int) -> Any:
        from scipy.stats import poisson as scipy_poisson

        h = self.team_map.get(int(home_id))
        a = self.team_map.get(int(away_id))
        attack_h = float(self.attack[h]) if h is not None else 1.0
        defense_a = float(self.defense[a]) if a is not None else 1.0
        attack_a = float(self.attack[a]) if a is not None else 1.0
        defense_h = float(self.defense[h]) if h is not None else 1.0

        # Form adjustment: últimos 5 partidos weighted blend.
        # `form_attack`/`form_defense`/`form_weight` se añadieron en v3 — modelos
        # persistidos en MLflow antes de v3 NO tienen estos atributos en su pickle.
        # Usamos getattr con default None para mantener compat retro.
        form_attack = getattr(self, "form_attack", None)
        form_defense = getattr(self, "form_defense", None)
        if form_attack is not None and form_defense is not None:
            fw = float(getattr(self, "form_weight", 0.3))
            if h is not None:
                attack_h = (1 - fw) * attack_h + fw * float(form_attack[h])
                defense_h = (1 - fw) * defense_h + fw * float(form_defense[h])
            if a is not None:
                attack_a = (1 - fw) * attack_a + fw * float(form_attack[a])
                defense_a = (1 - fw) * defense_a + fw * float(form_defense[a])

        # Apply home advantage multiplicativo (default 1.0 si no presente)
        home_advantage = float(getattr(self, "home_advantage", 1.0))
        lam_home = attack_h * defense_a * self.lg_avg_home * home_advantage
        lam_away = attack_a * defense_h * self.lg_avg_away
        lam_home = max(0.1, min(lam_home, 6.0))
        lam_away = max(0.1, min(lam_away, 6.0))

        max_g = 10
        matrix = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                p = scipy_poisson.pmf(i, lam_home) * scipy_poisson.pmf(j, lam_away)
                matrix[i, j] = p * self._dc_tau(i, j, lam_home, lam_away)
        matrix = np.maximum(matrix, 0.0)  # tau puede dar negativo con lam grande
        matrix /= matrix.sum()
        p_home = float(sum(matrix[i, j] for i in range(max_g + 1) for j in range(i)))
        p_draw = float(matrix.trace())
        p_away = 1.0 - p_home - p_draw

        class _Pred:
            def __init__(self, p_h: float, p_d: float, p_a: float) -> None:
                self.home_draw_away = [p_h, p_d, p_a]

        return _Pred(p_home, p_draw, p_away)


def _fit_independent_poisson(
    *,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    team_map: dict[int, int],
    bayesian_priors: dict[int, tuple[float, float]] | None = None,
    days_ago: np.ndarray | None = None,
    xi_decay: float = 0.003,
) -> Any:
    """Poisson by-team rates con shrinkage + home advantage + time-weighted decay.

    Mejoras v2 vs v1:
      - Home advantage: cociente lg_avg_home / (lg_avg_home + lg_avg_away) × 2
      - Time-weighted (xi exponencial decay): matches recientes pesan más
      - Tau Dixon-Coles: corrige bias low-score empates
      - n_games dinámico en shrinkage (teams con menos games → más shrinkage)
    """
    n_teams = len(team_map)
    lg_avg_home = float(home_goals.mean())
    lg_avg_away = float(away_goals.mean())
    home_advantage = (lg_avg_home * 2) / max(lg_avg_home + lg_avg_away, 1e-3)

    # Pesos temporales (decay exponencial). Sin days_ago → uniform.
    if days_ago is not None and len(days_ago) == len(home_goals):
        weights = np.exp(-xi_decay * days_ago).astype(np.float64)
    else:
        weights = np.ones(len(home_goals), dtype=np.float64)

    goals_for = np.zeros(n_teams)
    goals_against = np.zeros(n_teams)
    weighted_games = np.zeros(n_teams)

    for i in range(len(home_goals)):
        h = home[i]
        a = away[i]
        w = weights[i]
        goals_for[h] += home_goals[i] * w
        goals_against[h] += away_goals[i] * w
        goals_for[a] += away_goals[i] * w
        goals_against[a] += home_goals[i] * w
        weighted_games[h] += w
        weighted_games[a] += w

    # Shrinkage Bayesian dinámico (teams con pocos games → más shrinkage)
    k_shrink = 15.0  # más agresivo: antes era 10
    baseline = (lg_avg_home + lg_avg_away) / 2
    avg_for = (goals_for + k_shrink * baseline) / (weighted_games + k_shrink)
    avg_against = (goals_against + k_shrink * baseline) / (weighted_games + k_shrink)
    attack = avg_for / max(baseline, 1e-3)
    defense = avg_against / max(baseline, 1e-3)

    # Blend con Bayesian priors (50/50 peso si disponibles)
    if bayesian_priors:
        for team_id, idx in team_map.items():
            prior = bayesian_priors.get(team_id)
            if prior:
                prior_attack, prior_defense = prior
                attack[idx] = 0.5 * attack[idx] + 0.5 * prior_attack
                defense[idx] = 0.5 * defense[idx] + 0.5 * prior_defense

    # Fit tau via 3-fold CV (no training leakage): split training en 3 folds
    # y escoge tau que maximiza log-likelihood en fold-held-out.
    tau = _fit_tau_via_cv(
        home_goals=home_goals,
        away_goals=away_goals,
        home=home,
        away=away,
        team_map=team_map,
        lg_avg_home=lg_avg_home,
        lg_avg_away=lg_avg_away,
        home_advantage=home_advantage,
        k_shrink=k_shrink,
    )

    # Form features: attack/defense de últimos 5 partidos por team
    form_attack, form_defense = _compute_team_form(
        home_goals=home_goals,
        away_goals=away_goals,
        home=home,
        away=away,
        team_map=team_map,
        baseline=baseline,
        last_n=5,
    )

    return _IndependentPoissonModel(
        attack=attack,
        defense=defense,
        lg_avg_home=lg_avg_home,
        lg_avg_away=lg_avg_away,
        team_map=team_map,
        home_advantage=home_advantage,
        tau=tau,
        form_attack=form_attack,
        form_defense=form_defense,
        form_weight=0.15,  # 15% peso forma (tweak suave, evita sobre-reacción a racha)
    )


def _compute_team_form(
    *,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    team_map: dict[int, int],
    baseline: float,
    last_n: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute form (attack/defense) para últimos `last_n` partidos por team."""
    n_teams = len(team_map)
    # last_n matches por team — usa stack deque-like
    recent_goals_for: dict[int, list[float]] = {t: [] for t in range(n_teams)}
    recent_goals_against: dict[int, list[float]] = {t: [] for t in range(n_teams)}

    for i in range(len(home_goals)):
        h = int(home[i])
        a = int(away[i])
        hg = float(home_goals[i])
        ag = float(away_goals[i])
        recent_goals_for[h].append(hg)
        recent_goals_against[h].append(ag)
        recent_goals_for[a].append(ag)
        recent_goals_against[a].append(hg)
        # Mantener solo últimos last_n
        if len(recent_goals_for[h]) > last_n:
            recent_goals_for[h].pop(0)
            recent_goals_against[h].pop(0)
        if len(recent_goals_for[a]) > last_n:
            recent_goals_for[a].pop(0)
            recent_goals_against[a].pop(0)

    form_attack = np.ones(n_teams)
    form_defense = np.ones(n_teams)
    for t in range(n_teams):
        if recent_goals_for[t]:
            form_attack[t] = float(np.mean(recent_goals_for[t])) / max(baseline, 1e-3)
            form_defense[t] = float(np.mean(recent_goals_against[t])) / max(baseline, 1e-3)
    return form_attack, form_defense


def _fit_tau_via_cv(
    *,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    team_map: dict[int, int],
    lg_avg_home: float,
    lg_avg_away: float,
    home_advantage: float,
    k_shrink: float,
) -> float:
    """Time-series CV para tau: split train en 3 bloques cronológicos, fit
    attack/defense solo en bloque i, evalúa tau en bloque i+1. Evita leakage
    donde tau se overfitea a training data.
    """
    from scipy.stats import poisson as scipy_poisson

    n = len(home_goals)
    if n < 300:
        return 0.0

    best_tau = 0.0
    best_avg_ll = -np.inf
    tau_grid = np.linspace(-0.12, 0.12, 9)

    fold_size = n // 3
    for tau in tau_grid:
        fold_lls: list[float] = []
        for fold in range(2):  # usa fold 0→eval 1, fold 1→eval 2
            train_end = (fold + 1) * fold_size
            eval_start = train_end
            eval_end = min(train_end + fold_size, n)

            # Fit attack/defense SOLO en train portion
            n_teams = len(team_map)
            gf = np.zeros(n_teams)
            ga = np.zeros(n_teams)
            wg = np.zeros(n_teams)
            for i in range(train_end):
                h = int(home[i])
                a = int(away[i])
                gf[h] += home_goals[i]
                ga[h] += away_goals[i]
                gf[a] += away_goals[i]
                ga[a] += home_goals[i]
                wg[h] += 1
                wg[a] += 1

            baseline = (lg_avg_home + lg_avg_away) / 2
            attack_f = (gf + k_shrink * baseline) / (wg + k_shrink) / max(baseline, 1e-3)
            defense_f = (ga + k_shrink * baseline) / (wg + k_shrink) / max(baseline, 1e-3)

            # Eval sobre fold siguiente
            total_ll = 0.0
            n_eval = 0
            for i in range(eval_start, eval_end):
                h = int(home[i])
                a = int(away[i])
                lam_h = max(
                    0.1, min(attack_f[h] * defense_f[a] * lg_avg_home * home_advantage, 6.0)
                )
                lam_a = max(0.1, min(attack_f[a] * defense_f[h] * lg_avg_away, 6.0))
                hg = int(home_goals[i])
                ag = int(away_goals[i])

                p = scipy_poisson.pmf(hg, lam_h) * scipy_poisson.pmf(ag, lam_a)
                if hg == 0 and ag == 0:
                    p *= max(1.0 - lam_h * lam_a * tau, 1e-6)
                elif hg == 0 and ag == 1:
                    p *= max(1.0 + lam_h * tau, 1e-6)
                elif hg == 1 and ag == 0:
                    p *= max(1.0 + lam_a * tau, 1e-6)
                elif hg == 1 and ag == 1:
                    p *= max(1.0 - tau, 1e-6)

                total_ll += np.log(max(p, 1e-10))
                n_eval += 1

            if n_eval > 0:
                fold_lls.append(total_ll / n_eval)

        if fold_lls:
            avg_ll = float(np.mean(fold_lls))
            if avg_ll > best_avg_ll:
                best_avg_ll = avg_ll
                best_tau = float(tau)

    logger.info("soccer.poisson_tau_cv", best_tau=best_tau, avg_ll=best_avg_ll)
    return best_tau


def _fit_tau_grid(
    *,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    attack: np.ndarray,
    defense: np.ndarray,
    lg_avg_home: float,
    lg_avg_away: float,
    home_advantage: float,
) -> float:
    """Grid search over tau [-0.15, 0.15] maximizando log-likelihood en training.

    Tau Dixon-Coles corrige sesgo 0-0/1-1 cuando hay sparse scoring en la liga.
    """
    from scipy.stats import poisson as scipy_poisson

    best_tau = 0.0
    best_ll = -np.inf
    tau_grid = np.linspace(-0.15, 0.15, 11)

    for tau in tau_grid:
        total_ll = 0.0
        n = 0
        for i in range(len(home_goals)):
            h = int(home[i])
            a = int(away[i])
            lam_h = max(0.1, min(attack[h] * defense[a] * lg_avg_home * home_advantage, 6.0))
            lam_a = max(0.1, min(attack[a] * defense[h] * lg_avg_away, 6.0))
            hg = int(home_goals[i])
            ag = int(away_goals[i])

            p = scipy_poisson.pmf(hg, lam_h) * scipy_poisson.pmf(ag, lam_a)
            # Apply tau correction
            if hg == 0 and ag == 0:
                p *= max(1.0 - lam_h * lam_a * tau, 1e-6)
            elif hg == 0 and ag == 1:
                p *= max(1.0 + lam_h * tau, 1e-6)
            elif hg == 1 and ag == 0:
                p *= max(1.0 + lam_a * tau, 1e-6)
            elif hg == 1 and ag == 1:
                p *= max(1.0 - tau, 1e-6)

            total_ll += np.log(max(p, 1e-10))
            n += 1

        if total_ll > best_ll:
            best_ll = total_ll
            best_tau = float(tau)

    logger.info("soccer.poisson_tau_fit", best_tau=best_tau, best_ll=best_ll / max(n, 1))
    return best_tau


async def _load_bayesian_priors(team_ids: list[int]) -> dict[int, tuple[float, float]]:
    """Lee team_strength_bayesian para todos los teams del training set."""
    if not team_ids:
        return {}
    async with session_scope() as session:
        r = await session.execute(
            text(
                """
                SELECT team_id, attack_rating, defense_rating, n_matches
                FROM team_strength_bayesian
                WHERE team_id = ANY(:ids) AND n_matches >= 5
                """
            ),
            {"ids": team_ids},
        )
        return {
            int(row.team_id): (float(row.attack_rating), float(row.defense_rating))
            for row in r.all()
        }


def evaluate_dc_model(model: Any, holdout: list[dict[str, Any]]) -> dict[str, float]:
    """Computa log-loss + Brier 1X2 sobre holdout."""
    if model is None or not holdout:
        return {}

    losses: list[float] = []
    briers: list[float] = []
    for m in holdout:
        try:
            prediction = model.predict(m["home_id"], m["away_id"])
            # penaltyblog devuelve dict con probabilities {home_win, draw, away_win}
            probs = prediction.home_draw_away
            hg = int(m["home_goals"])
            ag = int(m["away_goals"])
            if hg > ag:
                actual_idx = 0
                actual = np.array([1, 0, 0])
            elif hg == ag:
                actual_idx = 1
                actual = np.array([0, 1, 0])
            else:
                actual_idx = 2
                actual = np.array([0, 0, 1])
            p_actual = max(probs[actual_idx], 1e-7)
            losses.append(-np.log(p_actual))
            briers.append(float(np.sum((np.asarray(probs) - actual) ** 2)) / 3.0)
        except Exception as exc:
            logger.debug("soccer.dc_predict_fail", error=str(exc))
            continue

    if not losses:
        return {}
    return {
        "log_loss": float(np.mean(losses)),
        "brier": float(np.mean(briers)),
        "n_holdout": len(losses),
    }


async def train_soccer(cfg: SoccerTrainConfig | None = None) -> dict[str, Any]:
    """Pipeline Dixon-Coles con MLflow logging."""
    cfg = cfg or SoccerTrainConfig(league_id=262, seasons=["2024", "2025", "2026"])

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    matches = await load_soccer_data(cfg.league_id, cfg.seasons)
    if len(matches) < 100:
        msg = f"Muestra insuficiente ({len(matches)} matches) para Dixon-Coles"
        raise RuntimeError(msg)

    # Load Bayesian priors si están disponibles (actualizados online por
    # settle_bets.update_bayesian_team_strengths)
    team_ids = list({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
    bayesian_priors = await _load_bayesian_priors(team_ids)
    logger.info(
        "soccer.bayesian_priors_loaded",
        n_teams=len(team_ids),
        n_with_prior=len(bayesian_priors),
    )

    # Walk-forward 80/20
    split_idx = int(len(matches) * 0.8)
    train_matches = matches[:split_idx]
    holdout_matches = matches[split_idx:]

    with mlflow.start_run(
        run_name=f"soccer_{cfg.league_id}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M')}"
    ):
        mlflow.log_params(
            {
                "league_id": cfg.league_id,
                "seasons": ",".join(cfg.seasons),
                "model": "dixon_coles",
                "xi_decay": cfg.xi_decay,
                "n_train": len(train_matches),
                "n_holdout": len(holdout_matches),
            }
        )

        model = fit_dixon_coles(
            train_matches, xi=cfg.xi_decay, bayesian_priors=bayesian_priors or None
        )
        if model is None:
            return {"ok": False, "reason": "dc_fit_failed"}

        metrics = evaluate_dc_model(model, holdout_matches)
        for k, v in metrics.items():
            if isinstance(v, int | float):
                mlflow.log_metric(k, float(v))

        # Log modelo
        model_path = Path("/tmp") / "soccer_dc.pkl"
        with model_path.open("wb") as f:
            cloudpickle.dump(
                {
                    "estimator": model,
                    "feature_names": [],
                    "target": "1x2",  # SoccerTrainConfig usa Dixon-Coles, target fijo
                    "sport": "soccer",
                    "model_type": "dixon_coles",
                    "config": cfg,
                },
                f,
            )
        mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        from apuestas.ml.registry_helper import register_model_in_db

        await register_model_in_db(
            mlflow_run_id=run_id,
            model_name=cfg.experiment_name,
            sport_code="soccer",
            stage=cfg.stage,
            metrics=metrics,
        )

        logger.info(
            "soccer.train.done",
            league=cfg.league_id,
            log_loss=metrics.get("log_loss"),
            brier=metrics.get("brier"),
        )

    return {"ok": True, "metrics": metrics}
