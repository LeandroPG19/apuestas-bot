"""Bayesian xG runtime predictor — Fase 1 wire #141.

Persistir traces de Bayesian xG a disk y re-usar en inference. Evita re-fit
cada vez. Carga posteriors cached desde `artifacts/bayesian_xg/{league_id}/`.

predict_match_probs(league_id, home_team_id, away_team_id) →
    {"home_win": p, "draw": p, "away_win": p}

Uso en detector: si sport=soccer + league en hierarchy Bayesian → usar en
vez de Dixon-Coles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts" / "bayesian_xg"
_CACHE: dict[int, dict] = {}


def save_posteriors(
    league_id: int,
    *,
    mu: float,
    home_adv: float,
    att_by_team: dict[int, float],
    def_by_team: dict[int, float],
    n_games: int,
) -> Path:
    """Guarda posteriors point-estimates a artifacts/bayesian_xg/{league}/posteriors.json."""
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACTS / f"league_{league_id}" / "posteriors.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "league_id": league_id,
        "mu": mu,
        "home_adv": home_adv,
        "att": {str(k): float(v) for k, v in att_by_team.items()},
        "def_": {str(k): float(v) for k, v in def_by_team.items()},
        "n_games": n_games,
    }
    path.write_text(json.dumps(payload, indent=2))
    _CACHE.pop(league_id, None)
    return path


def _load_posteriors(league_id: int) -> dict[str, Any] | None:
    if league_id in _CACHE:
        return _CACHE[league_id]
    path = _ARTIFACTS / f"league_{league_id}" / "posteriors.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        data["att"] = {int(k): float(v) for k, v in data["att"].items()}
        data["def_"] = {int(k): float(v) for k, v in data["def_"].items()}
        _CACHE[league_id] = data
        return data
    except Exception as exc:
        logger.debug("bayes_xg_runtime.load_fail", league=league_id, error=str(exc)[:80])
        return None


def predict_match_probs(
    league_id: int, home_team_id: int, away_team_id: int
) -> dict[str, float] | None:
    """Predice p(home_win|draw|away_win) via Poisson sampling."""
    p = _load_posteriors(league_id)
    if p is None:
        return None
    try:
        lam_home = float(
            np.exp(
                p["mu"]
                + p["home_adv"]
                + p["att"].get(home_team_id, 0.0)
                - p["def_"].get(away_team_id, 0.0)
            )
        )
        lam_away = float(
            np.exp(p["mu"] + p["att"].get(away_team_id, 0.0) - p["def_"].get(home_team_id, 0.0))
        )
        # Monte Carlo 10k
        rng = np.random.default_rng(42)
        n = 10_000
        hs = rng.poisson(lam_home, n)
        as_ = rng.poisson(lam_away, n)
        return {
            "home_win": float((hs > as_).mean()),
            "draw": float((hs == as_).mean()),
            "away_win": float((hs < as_).mean()),
            "lambda_home": lam_home,
            "lambda_away": lam_away,
        }
    except Exception as exc:
        logger.debug("bayes_xg_runtime.predict_fail", league=league_id, error=str(exc)[:80])
        return None


class BayesianXGModel:
    """sklearn-compat wrapper para Bayesian xG posteriors."""

    def __init__(self, league_id: int):
        self.league_id = league_id
        self.classes_ = ["away", "draw", "home"]
        self.feature_names_in_ = ["home_team_id", "away_team_id"]

    def predict_proba(self, X):
        probs = []
        for row in X:
            try:
                if hasattr(row, "__len__") and len(row) >= 2:
                    h, a = int(row[0]), int(row[1])
                else:
                    h, a = 0, 0
            except Exception:
                h, a = 0, 0
            p = predict_match_probs(self.league_id, h, a)
            if p is None:
                probs.append([0.33, 0.33, 0.34])
            else:
                probs.append([p["away_win"], p["draw"], p["home_win"]])
        return np.array(probs)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array([self.classes_[i] for i in proba.argmax(axis=1)])


def posteriors_available() -> list[int]:
    """Lista de league_ids con posteriors persistidos en disk."""
    if not _ARTIFACTS.exists():
        return []
    out = []
    for p in _ARTIFACTS.iterdir():
        if p.is_dir() and p.name.startswith("league_"):
            post = p / "posteriors.json"
            if post.exists():
                try:
                    out.append(int(p.name.replace("league_", "")))
                except ValueError:
                    pass
    return sorted(out)


__all__ = [
    "BayesianXGModel",
    "posteriors_available",
    "predict_match_probs",
    "save_posteriors",
]
