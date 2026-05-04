"""Dixon-Coles cross-liga runtime — sklearn wrapper.

Reusa `apuestas.features.soccer.dixon_coles_predict` que predice basado en
`team_strength_bayesian` (495 teams cubiertos: top ligas europeas + Brasil +
Argentina). Se usa como modelo de fallback en `model_hierarchy` para ligas
sin trainer dedicado (UCL/UEL/Sudamerica/Cup competitions).

Si los teams no tienen strength registrado, `dixon_coles_predict` retorna
None y este wrapper devuelve probs uniformes [0.33, 0.33, 0.34] que el
detector trata como "modelo no útil" via shrinkage anchor.
"""

from __future__ import annotations

import numpy as np

from apuestas.features.soccer import (
    dixon_coles_predict,
    dixon_coles_predict_btts,
    dixon_coles_predict_total,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


class DixonColesCrossLeagueModel:
    """sklearn-compat wrapper para DC cross-liga.

    Espera X con primeras 2 columnas = [home_team_id, away_team_id]. Si las
    features tienen más columnas se ignoran (DC solo usa team_strength).

    classes_ está en orden alfabético sklearn: ['away', 'draw', 'home'] para
    3-way. Para totals/btts (2-way), classes_ se setea en `predict_proba_market`.
    """

    _estimator_type = "classifier"

    def __init__(self, line: float | None = None) -> None:
        self.classes_ = np.array(["away", "draw", "home"])
        self.feature_names_in_ = ["home_team_id", "away_team_id"]
        self.line = line  # para totals

    def _extract_ids(self, X: np.ndarray) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for row in np.asarray(X):
            try:
                if hasattr(row, "__len__") and len(row) >= 2:
                    out.append((int(row[0]), int(row[1])))
                else:
                    out.append((0, 0))
            except ValueError, TypeError:
                out.append((0, 0))
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """3-way h2h prediction (away, draw, home)."""
        probs = []
        for h, a in self._extract_ids(X):
            p = dixon_coles_predict(h, a)
            if p is None:
                probs.append([0.33, 0.33, 0.34])  # uniforme, anchor lo neutraliza
            else:
                probs.append([p["p_away"], p["p_draw"], p["p_home"]])
        return np.array(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def predict_proba_total(self, X: np.ndarray, line: float) -> np.ndarray:
        """Devuelve proba [under, over] para totals goles."""
        probs = []
        for h, a in self._extract_ids(X):
            p = dixon_coles_predict_total(h, a, line)
            if p is None:
                probs.append([0.5, 0.5])
            else:
                probs.append([p["under"], p["over"]])
        return np.array(probs)

    def predict_proba_btts(self, X: np.ndarray) -> np.ndarray:
        """Devuelve proba [no, yes] para BTTS."""
        probs = []
        for h, a in self._extract_ids(X):
            p = dixon_coles_predict_btts(h, a)
            if p is None:
                probs.append([0.5, 0.5])
            else:
                probs.append([p["no"], p["yes"]])
        return np.array(probs)


__all__ = ["DixonColesCrossLeagueModel"]
