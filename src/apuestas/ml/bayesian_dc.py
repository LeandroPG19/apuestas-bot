"""Fase 3.5 — Bayesian team-strength online update (Dixon-Coles).

El modelo DC offline entrenado en `train_soccer.py` se congela al final de cada
run. Durante la temporada en curso, cada match aporta información nueva que el
modelo offline ignora hasta el próximo retrain. Esto degrada precisión en:
  - Equipos promovidos (poca data histórica en la liga).
  - Mid-season signings (Luka→Lakers).
  - Coaching changes que modifican estilo.

Solución: Bayesian online update con prior de liga:
  - Prior: media liga (attack=1.0, defense=1.0).
  - Posterior tras cada match: weighted average `new_rating = α·old + (1-α)·match_obs`.
  - α = 1 / (1 + n_matches · decay).

K-factor recomendado para soccer: 0.15 (Benham/Smartodds).

Tabla: `team_strength_bayesian` (migración 0012).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Valores default prior: equipo promedio en la liga
PRIOR_ATTACK = 1.0
PRIOR_DEFENSE = 1.0
PRIOR_VARIANCE = 0.25  # incertidumbre inicial alta
DECAY_FACTOR = 0.05  # cuánto peso pierde cada partido histórico
LEARNING_RATE = 0.15  # K-factor para update


@dataclass(slots=True)
class TeamStrengthBayesian:
    team_id: int
    attack_rating: float
    defense_rating: float
    variance: float
    n_matches: int
    updated_at: datetime


async def get_team_strength(team_id: int) -> TeamStrengthBayesian:
    """Fetch team strength o retorna prior si no existe."""
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT team_id, attack_rating, defense_rating,
                           variance, n_matches, updated_at
                    FROM team_strength_bayesian
                    WHERE team_id = :tid
                    """
                ),
                {"tid": team_id},
            )
        ).first()
    if row is None:
        return TeamStrengthBayesian(
            team_id=team_id,
            attack_rating=PRIOR_ATTACK,
            defense_rating=PRIOR_DEFENSE,
            variance=PRIOR_VARIANCE,
            n_matches=0,
            updated_at=datetime.now(tz=UTC),
        )
    return TeamStrengthBayesian(
        team_id=int(row.team_id),
        attack_rating=float(row.attack_rating),
        defense_rating=float(row.defense_rating),
        variance=float(row.variance),
        n_matches=int(row.n_matches),
        updated_at=row.updated_at,
    )


def _observed_strength_from_match(
    goals_scored: int,
    goals_conceded: int,
    opponent_strength: TeamStrengthBayesian,
) -> tuple[float, float]:
    """Extrae attack/defense observados de un solo partido.

    Usa el ratio goles vs expected-goles-vs-opponent.
    Simplificación: assume home advantage neutralizado por la liga.
    """
    # Expected goals conditionando en opponent defense
    # attack_observed ≈ goals_scored / opponent_defense_rating
    opp_def = max(opponent_strength.defense_rating, 0.3)
    opp_att = max(opponent_strength.attack_rating, 0.3)
    attack_obs = goals_scored / opp_def
    # defense_observed: inverso del daño recibido
    defense_obs = opp_att / max(goals_conceded, 0.3)
    return attack_obs, defense_obs


async def update_team_strength(
    team_id: int,
    *,
    goals_scored: int,
    goals_conceded: int,
    opponent_id: int,
    learning_rate: float = LEARNING_RATE,
) -> TeamStrengthBayesian:
    """Aplica un update bayesian tras un match.

    Ejecutado desde `flows/settle_bets.py` tras cada match soccer finalizado.
    """
    current = await get_team_strength(team_id)
    opponent = await get_team_strength(opponent_id)

    attack_obs, defense_obs = _observed_strength_from_match(goals_scored, goals_conceded, opponent)

    # Weighted update (α = 1 - learning_rate, (1-α) = learning_rate)
    alpha = 1.0 - learning_rate
    new_attack = alpha * current.attack_rating + learning_rate * attack_obs
    new_defense = alpha * current.defense_rating + learning_rate * defense_obs
    new_variance = current.variance * alpha + learning_rate * (
        (attack_obs - current.attack_rating) ** 2
    )
    new_n = current.n_matches + 1
    now = datetime.now(tz=UTC)

    async with session_scope() as session:
        await session.execute(
            text(
                """
                INSERT INTO team_strength_bayesian
                  (team_id, attack_rating, defense_rating, variance, n_matches, updated_at)
                VALUES (:tid, :att, :def_, :var, :n, :ts)
                ON CONFLICT (team_id) DO UPDATE
                SET attack_rating = EXCLUDED.attack_rating,
                    defense_rating = EXCLUDED.defense_rating,
                    variance = EXCLUDED.variance,
                    n_matches = EXCLUDED.n_matches,
                    updated_at = EXCLUDED.updated_at
                """
            ),
            {
                "tid": team_id,
                "att": new_attack,
                "def_": new_defense,
                "var": new_variance,
                "n": new_n,
                "ts": now,
            },
        )

    logger.debug(
        "bayesian_dc.updated",
        team_id=team_id,
        attack=new_attack,
        defense=new_defense,
        n_matches=new_n,
    )
    return TeamStrengthBayesian(
        team_id=team_id,
        attack_rating=new_attack,
        defense_rating=new_defense,
        variance=new_variance,
        n_matches=new_n,
        updated_at=now,
    )


async def process_match_settlement(
    match_id: int,
) -> dict[str, Any]:
    """Hook llamado desde settle_bets tras finalizar un match soccer.

    Update ambos equipos con las nuevas observaciones.
    """
    async with session_scope() as session:
        match = (
            await session.execute(
                text(
                    """
                    SELECT home_team_id, away_team_id, home_score, away_score, sport_code
                    FROM matches
                    WHERE id = :mid AND status = 'finished'
                    """
                ),
                {"mid": match_id},
            )
        ).first()

    if match is None or match.home_score is None:
        return {"skipped": True, "reason": "not_finished_or_missing"}
    if match.sport_code != "soccer":
        return {"skipped": True, "reason": "not_soccer"}

    home_new = await update_team_strength(
        team_id=match.home_team_id,
        goals_scored=int(match.home_score),
        goals_conceded=int(match.away_score),
        opponent_id=match.away_team_id,
    )
    away_new = await update_team_strength(
        team_id=match.away_team_id,
        goals_scored=int(match.away_score),
        goals_conceded=int(match.home_score),
        opponent_id=match.home_team_id,
    )
    return {
        "home": {
            "team_id": home_new.team_id,
            "attack": home_new.attack_rating,
            "defense": home_new.defense_rating,
        },
        "away": {
            "team_id": away_new.team_id,
            "attack": away_new.attack_rating,
            "defense": away_new.defense_rating,
        },
    }
