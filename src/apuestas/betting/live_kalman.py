"""Kalman filter para live betting — Sprint 11 Fase I.

Paper: Ötting 2024 "Live Betting Demand Analysis" (Applied Stochastic Models
in Business and Industry). DraftKings dataset.

Idea: durante un match, la probabilidad P(home_wins) se actualiza con cada
evento (gol, canasta, turnover, etc.). Un Kalman filter mantiene el estado
(probabilidad + incertidumbre) y lo actualiza en O(1) por evento.

Modelo simplificado:
- Estado: x = logit(P(home_wins))
- Proceso: x_{t+1} = x_t + w, w ~ N(0, Q) (drift lento)
- Observación: z = x + v, v ~ N(0, R) donde R depende del shock
    - Gol marcador = observation fuerte (R pequeño)
    - Minuto transcurrido sin cambio = observation débil (R grande)

Por deporte define:
- Q (drift variance): NBA 0.001 / min, NFL 0.0005, soccer 0.0002
- shock_by_event: {goal: N(0, 0.05), touchdown: N(0, 0.1), ...}

Uso:
    kf = LiveKalmanFilter(sport="soccer", initial_p_home=0.55)
    kf.observe_goal(team="home", minute=23)  # Update posterior
    p = kf.p_home_win()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def _logit(p: float) -> float:
    p_clip = max(min(p, 1 - 1e-6), 1e-6)
    return math.log(p_clip / (1 - p_clip))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(slots=True)
class KalmanParams:
    """Parámetros de process + observation por deporte."""

    drift_variance_per_min: float
    goal_shock_std: float
    minor_event_std: float  # yellow card, timeout, etc.


_PARAMS_BY_SPORT: dict[str, KalmanParams] = {
    "soccer": KalmanParams(
        drift_variance_per_min=0.0002, goal_shock_std=0.35, minor_event_std=0.05
    ),
    "nba": KalmanParams(drift_variance_per_min=0.001, goal_shock_std=0.08, minor_event_std=0.02),
    "nfl": KalmanParams(drift_variance_per_min=0.0005, goal_shock_std=0.25, minor_event_std=0.05),
    "nhl": KalmanParams(drift_variance_per_min=0.0003, goal_shock_std=0.30, minor_event_std=0.04),
    "mlb": KalmanParams(drift_variance_per_min=0.0002, goal_shock_std=0.25, minor_event_std=0.03),
}


def params_for(sport: str) -> KalmanParams:
    return _PARAMS_BY_SPORT.get(sport.lower(), KalmanParams(0.0005, 0.20, 0.05))


@dataclass(slots=True)
class LiveKalmanFilter:
    """1D Kalman filter para P(home_wins) durante el match."""

    sport: str
    initial_p_home: float
    state_logit: float = 0.0
    state_variance: float = 0.5
    last_update_minute: float = 0.0

    def __post_init__(self) -> None:
        self.state_logit = _logit(self.initial_p_home)

    def _predict(self, current_minute: float) -> None:
        """Avanza el estado (drift) sin observación."""
        params = params_for(self.sport)
        dt = max(0.0, current_minute - self.last_update_minute)
        # x estado se mantiene (no drift direccional en logit)
        # P aumenta con drift variance
        self.state_variance = self.state_variance + dt * params.drift_variance_per_min
        self.last_update_minute = current_minute

    def _update(self, observation_logit: float, obs_variance: float) -> None:
        """Fusión gaussiana: posterior = prior * observation."""
        denom = self.state_variance + obs_variance
        if denom <= 0:
            return
        kalman_gain = self.state_variance / denom
        self.state_logit = self.state_logit + kalman_gain * (observation_logit - self.state_logit)
        self.state_variance = (1 - kalman_gain) * self.state_variance

    def observe_goal(self, team: str, minute: float) -> None:
        """Actualiza posterior tras gol (u otro shock fuerte)."""
        self._predict(minute)
        params = params_for(self.sport)
        # Gol a favor de home → shift logit positivo; contra → negativo.
        direction = 1.0 if team.lower() == "home" else -1.0
        obs_logit = self.state_logit + direction * params.goal_shock_std * 2.0
        obs_var = params.goal_shock_std**2
        self._update(obs_logit, obs_var)
        logger.info(
            "live_kalman.goal",
            team=team,
            minute=minute,
            p_home=round(self.p_home_win(), 3),
        )

    def observe_minor(self, team: str, minute: float) -> None:
        """Evento menor (amarilla, falta, foul): update débil."""
        self._predict(minute)
        params = params_for(self.sport)
        direction = -1.0 if team.lower() == "home" else 1.0
        obs_logit = self.state_logit + direction * params.minor_event_std
        obs_var = params.minor_event_std**2 * 4.0  # baja confianza
        self._update(obs_logit, obs_var)

    def observe_score_delta(self, home_score: int, away_score: int, minute: float) -> None:
        """Actualiza con el marcador actual (override directo si disponible).

        Usa modelo de time-score correlation: score_diff > 0 con poco tiempo
        restante → P(home) alta.
        """
        self._predict(minute)
        params = params_for(self.sport)
        # Aproximación: prob home win dado marker = sigmoide(score_diff * weight)
        # weight aumenta al reducirse el tiempo restante
        total_minutes = {"soccer": 90, "nba": 48, "nfl": 60, "nhl": 60}.get(self.sport.lower(), 90)
        time_left_pct = max(0.05, 1.0 - minute / total_minutes)
        score_weight = 1.0 / time_left_pct  # ↑ con menos tiempo
        score_diff = home_score - away_score
        obs_logit = score_diff * score_weight * 0.3
        obs_var = max(params.goal_shock_std**2 * time_left_pct, 0.01)
        self._update(obs_logit, obs_var)

    def p_home_win(self) -> float:
        return _sigmoid(self.state_logit)

    def p_away_win(self) -> float:
        return 1.0 - self.p_home_win()

    def uncertainty_bps(self) -> float:
        """Incertidumbre convertida a bps para usar en Kelly fractional."""
        return float(self.state_variance * 100.0)


__all__ = ["KalmanParams", "LiveKalmanFilter", "params_for"]
