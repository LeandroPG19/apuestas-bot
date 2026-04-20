"""Coaching clutch tendencies — el edge #2 de Voulgaris.

Voulgaris descubrió que coaches NBA tenían hábitos predecibles en los
últimos 3 minutos de partidos cerrados. Modelamos esos hábitos para
ajustar totals/spreads en situaciones clutch.

Features por coach:
- timeout_usage_pre_clutch: % de timeouts consumidos antes del clutch
- clutch_close_out_offense_rate: fracción de posesiones donde juega star
- lineup_pattern_mins_3_4: patrones de substitución en mins 3-4 del Q
- hack_a_player_rate: cuándo deciden fouleo intencional (NBA)
- go_for_it_4th_down_rate: agresividad NFL
- bullpen_high_leverage_usage: MLB manager patterns
- pitch_count_early_hook_avg: MLB pull rapidez

Deriva de `play_by_play` agregado por coach_id.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def fetch_match_coaches(match_id: int) -> list[dict[str, Any]]:
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT mc.coach_id, mc.team_id, c.name, c.hired_at
                FROM match_coaches mc
                JOIN coaches c ON c.id = mc.coach_id
                WHERE mc.match_id = :m
                """
            ),
            {"m": match_id},
        )
        return [dict(row._mapping) for row in r.all()]


async def fetch_coach_tendencies(coach_id: int, sport_code: str) -> dict[str, Any] | None:
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT timeout_usage_pre_clutch, clutch_close_out_offense_rate,
                       hack_a_player_rate, go_for_it_4th_down_rate,
                       bullpen_high_leverage_usage, pitch_count_early_hook_avg,
                       n_games_sample
                FROM coaching_tendencies
                WHERE coach_id = :c AND sport_code = :s
                """
            ),
            {"c": coach_id, "s": sport_code},
        )
        row = r.first()
        return dict(row._mapping) if row else None


async def compute_coaching_features(match_id: int, sport_code: str) -> dict[str, float]:
    """Features coaching para inyectar al modelo + al LLM prompt."""
    coaches = await fetch_match_coaches(match_id)
    if not coaches:
        return {}

    features: dict[str, float] = {}
    for coach in coaches:
        role = "home" if coach.get("is_home") else "away"
        # Para simplicidad, el primero es home (o inferir via match join)
        ten = await fetch_coach_tendencies(int(coach["coach_id"]), sport_code)
        if not ten:
            continue
        prefix = f"coach_{role}"
        for k, v in ten.items():
            if v is not None:
                features[f"{prefix}_{k}"] = float(v)

    # Diferenciales entre coaches (si ambos están registrados)
    if (
        "coach_home_timeout_usage_pre_clutch" in features
        and "coach_away_timeout_usage_pre_clutch" in features
    ):
        features["coach_timeout_diff"] = (
            features["coach_home_timeout_usage_pre_clutch"]
            - features["coach_away_timeout_usage_pre_clutch"]
        )

    return features


async def recompute_nba_coaching_from_pbp(coach_id: int, team_id: int) -> dict[str, Any]:
    """Deriva tendencias NBA desde play_by_play últimas 82 games del coach.

    - timeout_usage_pre_clutch: timeouts consumidos antes de Q4≤3min / total
    - hack_a_player_rate: fouls intencionales en clutch / posesiones
    """
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                WITH coach_matches AS (
                    SELECT DISTINCT mc.match_id
                    FROM match_coaches mc
                    JOIN matches m ON m.id = mc.match_id
                    WHERE mc.coach_id = :c AND mc.team_id = :t
                      AND m.status = 'finished'
                      AND m.sport_code = 'nba'
                    ORDER BY mc.match_id DESC
                    LIMIT 82
                )
                SELECT
                    COUNT(DISTINCT pbp.match_id) AS n_games,
                    AVG(CASE
                        WHEN pbp.event_type = 'timeout'
                          AND (pbp.period < 4 OR pbp.clock_seconds_remaining > 180)
                        THEN 1.0 ELSE 0.0
                    END) AS timeout_pre_clutch,
                    AVG(CASE
                        WHEN pbp.event_type = 'foul'
                          AND pbp.period = 4
                          AND pbp.clock_seconds_remaining <= 180
                        THEN 1.0 ELSE 0.0
                    END) AS hack_a_rate
                FROM play_by_play pbp
                JOIN coach_matches cm ON cm.match_id = pbp.match_id
                WHERE pbp.team_id = :t
                """
            ),
            {"c": coach_id, "t": team_id},
        )
        row = r.first()
        if not row or not row.n_games:
            return {}
        stats = {
            "n_games_sample": int(row.n_games),
            "timeout_usage_pre_clutch": float(row.timeout_pre_clutch or 0),
            "hack_a_player_rate": float(row.hack_a_rate or 0),
        }

        await s.execute(
            text(
                """
                INSERT INTO coaching_tendencies
                    (coach_id, sport_code, timeout_usage_pre_clutch,
                     hack_a_player_rate, n_games_sample, last_computed)
                VALUES (:c, 'nba', :tu, :ha, :n, NOW())
                ON CONFLICT (coach_id, sport_code) DO UPDATE SET
                    timeout_usage_pre_clutch = EXCLUDED.timeout_usage_pre_clutch,
                    hack_a_player_rate = EXCLUDED.hack_a_player_rate,
                    n_games_sample = EXCLUDED.n_games_sample,
                    last_computed = NOW()
                """
            ),
            {
                "c": coach_id,
                "tu": stats["timeout_usage_pre_clutch"],
                "ha": stats["hack_a_player_rate"],
                "n": stats["n_games_sample"],
            },
        )
    logger.info("coaching_clutch.updated", coach_id=coach_id, **stats)
    return stats
