"""Elo rating bidireccional + rest days — Sprint 10 Fase 2 (Mejora #7).

Implementa Elo dinámico (Hvattum & Arntzen 2010) como feature de entrada para
trainers. Calibrado para multi-deporte con parámetros K, HFA (home-field
advantage) y score-margin scaling específicos.

Uso:
    >>> builder = EloBuilder(sport="nba")
    >>> builder.update_match(home="LAL", away="GSW", home_score=110, away_score=108)
    >>> elo_home = builder.rating("LAL")  # 1514.3
    >>> feats = builder.features_for_upcoming("LAL", "GSW", home_rest_days=2, away_rest_days=0)
    >>> feats["elo_diff"]  # rating home - rating away (con HFA aplicado)

Parámetros por deporte (referencia academic literature):
- NBA: K=20, HFA=+100, margin scale 7 (Silver 538 metodología)
- NFL: K=20, HFA=+65, margin scale 10 (Elo Pro-Football-Reference)
- MLB: K=4, HFA=+24, margin scale 3 (Silver 2010)
- NHL: K=6, HFA=+50, margin scale 2 (Silver 2010)
- Soccer: K=20, HFA=+100, margin scale 1 + goal-diff log (clubelo.com)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_DEFAULT_RATING = 1500.0


@dataclass(slots=True)
class EloParams:
    k_factor: float
    hfa: float
    margin_scale: float
    use_log_margin: bool = False  # soccer: log(|goal_diff|+1)


_PARAMS_BY_SPORT: dict[str, EloParams] = {
    "nba": EloParams(k_factor=20.0, hfa=100.0, margin_scale=7.0),
    "nfl": EloParams(k_factor=20.0, hfa=65.0, margin_scale=10.0),
    "mlb": EloParams(k_factor=4.0, hfa=24.0, margin_scale=3.0),
    "nhl": EloParams(k_factor=6.0, hfa=50.0, margin_scale=2.0),
    "soccer": EloParams(k_factor=20.0, hfa=100.0, margin_scale=1.0, use_log_margin=True),
    "tennis": EloParams(k_factor=32.0, hfa=0.0, margin_scale=1.0),
    "boxing": EloParams(k_factor=24.0, hfa=0.0, margin_scale=1.0),
    "mma": EloParams(k_factor=24.0, hfa=0.0, margin_scale=1.0),
}


def _params_for(sport: str) -> EloParams:
    return _PARAMS_BY_SPORT.get(sport.lower(), EloParams(k_factor=20.0, hfa=50.0, margin_scale=5.0))


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilidad implícita de que A gane contra B (Elo classic)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(home_score: int, away_score: int, params: EloParams) -> float:
    """Ajuste por margen de victoria (Silver 2010).

    Premia blowouts sin inflar: para NBA +20 pts multiplica por ~1.8x;
    para soccer aplica log para mitigar goleadas.
    """
    diff = abs(home_score - away_score)
    if diff == 0:
        return 1.0
    if params.use_log_margin:
        return math.log(diff + 1.0) / math.log(2.0) + 1.0
    # Polinomial: 1 + diff/scale
    return 1.0 + diff / max(params.margin_scale, 0.1)


@dataclass(slots=True)
class EloBuilder:
    """Mantiene ratings por equipo con updates online.

    Diseñado para ser alimentado en orden cronológico desde train_base.
    No persiste estado — el caller decide cómo almacenar (DB/parquet).
    """

    sport: str
    ratings: dict[str, float] = field(default_factory=dict)
    last_played: dict[str, float] = field(default_factory=dict)  # timestamp unix
    n_matches: dict[str, int] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, _DEFAULT_RATING)

    def update_match(
        self,
        *,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        match_ts: float | None = None,
    ) -> tuple[float, float]:
        """Actualiza ratings tras match. Devuelve (new_home_rating, new_away_rating)."""
        params = _params_for(self.sport)
        r_home = self.rating(home)
        r_away = self.rating(away)

        # Aplicar HFA al cálculo de expected (home recibe boost)
        exp_home = expected_score(r_home + params.hfa, r_away)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        mm = margin_multiplier(home_score, away_score, params)
        delta = params.k_factor * mm * (actual_home - exp_home)

        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        if match_ts is not None:
            self.last_played[home] = match_ts
            self.last_played[away] = match_ts
        self.n_matches[home] = self.n_matches.get(home, 0) + 1
        self.n_matches[away] = self.n_matches.get(away, 0) + 1
        return self.ratings[home], self.ratings[away]

    def features_for_upcoming(
        self,
        home: str,
        away: str,
        *,
        home_rest_days: int | None = None,
        away_rest_days: int | None = None,
        match_ts: float | None = None,
    ) -> dict[str, float]:
        """Features para un partido futuro (a usar en train/predict)."""
        params = _params_for(self.sport)
        r_home = self.rating(home)
        r_away = self.rating(away)
        elo_diff = (r_home + params.hfa) - r_away
        p_home = expected_score(r_home + params.hfa, r_away)

        feats: dict[str, float] = {
            "elo_home": r_home,
            "elo_away": r_away,
            "elo_diff": elo_diff,
            "elo_p_home": p_home,
            "elo_n_matches_home": float(self.n_matches.get(home, 0)),
            "elo_n_matches_away": float(self.n_matches.get(away, 0)),
        }
        # Rest days: si se proveen explícitos, usarlos; si no, calcular desde last_played.
        if home_rest_days is None and match_ts is not None and home in self.last_played:
            home_rest_days = max(0, int((match_ts - self.last_played[home]) / 86400))
        if away_rest_days is None and match_ts is not None and away in self.last_played:
            away_rest_days = max(0, int((match_ts - self.last_played[away]) / 86400))
        if home_rest_days is not None:
            feats["rest_days_home"] = float(home_rest_days)
        if away_rest_days is not None:
            feats["rest_days_away"] = float(away_rest_days)
        if home_rest_days is not None and away_rest_days is not None:
            feats["rest_days_diff"] = float(home_rest_days - away_rest_days)
            # Back-to-back flag: 0 rest = cansancio NBA/NHL
            feats["b2b_home"] = 1.0 if home_rest_days == 0 else 0.0
            feats["b2b_away"] = 1.0 if away_rest_days == 0 else 0.0
        return feats


__all__ = ["EloBuilder", "EloParams", "expected_score", "margin_multiplier"]
