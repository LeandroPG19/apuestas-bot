"""Referee bias features — 25% del edge de Voulgaris según documentales.

Cada árbitro tiene tendencias estadísticamente significativas:
- NBA: FT/game, fouls/game, home team win rate, O/U rate
- NFL: penalties/game, home win rate, OT rate
- MLB umpire: strikezone size % vs automated
- Soccer: cards/game, var interventions, home win rate

Fuentes gratis: Umpire Scorecards (MLB), NBA Official (Refstats),
Pro Football Reference (NFL), FBref referee profiles (soccer).

Uso en pipeline:
    from apuestas.features.referee_bias import enrich_with_referee_bias
    features = await enrich_with_referee_bias(match_id, features)
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def fetch_match_referees(match_id: int) -> list[dict[str, Any]]:
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT r.id, r.name, r.role, mr.role AS match_role
                FROM match_referees mr
                JOIN referees r ON r.id = mr.referee_id
                WHERE mr.match_id = :m
                """
            ),
            {"m": match_id},
        )
        return [dict(row._mapping) for row in r.all()]


async def fetch_referee_profile(referee_id: int, sport_code: str) -> dict[str, Any] | None:
    async with session_scope() as s:
        r = await s.execute(
            text(
                """
                SELECT n_games, home_win_rate, home_ats_rate, over_rate,
                       avg_total, fouls_per_game, cards_per_game,
                       strikezone_size_pct, var_interventions_per_game
                FROM referee_bias_profile
                WHERE referee_id = :r AND sport_code = :s
                """
            ),
            {"r": referee_id, "s": sport_code},
        )
        row = r.first()
        return dict(row._mapping) if row else None


async def compute_referee_bias_features(match_id: int, sport_code: str) -> dict[str, float]:
    """Retorna features cuantitativos listos para inyectar al modelo.

    Si no hay árbitro asignado o no tiene perfil, retorna dict vacío
    (el modelo debe tolerar faltantes).
    """
    refs = await fetch_match_referees(match_id)
    if not refs:
        return {}

    # Para NBA/NFL: promedio del crew. Para soccer/MLB: árbitro principal.
    main_refs = [r for r in refs if r.get("match_role") in (None, "main", "crew_chief")]
    if not main_refs:
        main_refs = refs

    features: dict[str, float] = {}
    profiles: list[dict[str, Any]] = []
    for ref in main_refs:
        profile = await fetch_referee_profile(int(ref["id"]), sport_code)
        if profile:
            profiles.append(profile)

    if not profiles:
        return {}

    # Promedios (o principal si 1 solo)
    def _avg(field: str) -> float | None:
        vals = [float(p[field]) for p in profiles if p.get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    n_games = _avg("n_games") or 0
    home_wr = _avg("home_win_rate")
    home_ats = _avg("home_ats_rate")
    over_rate = _avg("over_rate")
    avg_total = _avg("avg_total")

    features["ref_n_games_sample"] = n_games
    if home_wr is not None:
        features["ref_home_win_rate"] = home_wr
        features["ref_home_win_bias_vs_league"] = home_wr - 0.55  # ~baseline NBA
    if home_ats is not None:
        features["ref_home_ats_rate"] = home_ats
    if over_rate is not None:
        features["ref_over_rate"] = over_rate
        features["ref_over_bias_vs_50"] = over_rate - 0.5
    if avg_total is not None:
        features["ref_avg_total"] = avg_total

    # Sport-specific
    sport_field_map = {
        "nba": "fouls_per_game",
        "nfl": "fouls_per_game",  # penalties como fouls
        "soccer": "cards_per_game",
        "mlb": "strikezone_size_pct",
    }
    sport_field = sport_field_map.get(sport_code)
    if sport_field:
        val = _avg(sport_field)
        if val is not None:
            features[f"ref_{sport_field}"] = val

    if sport_code == "soccer":
        var = _avg("var_interventions_per_game")
        if var is not None:
            features["ref_var_interventions_per_game"] = var

    # Confianza: muestras < 10 games → señal poco fiable, marcar
    features["ref_sample_reliable"] = 1.0 if n_games >= 20 else 0.0

    logger.debug(
        "referee_bias.computed",
        match_id=match_id,
        sport=sport_code,
        n_refs=len(profiles),
        n_features=len(features),
    )
    return features


async def recompute_all_referee_profiles(sport_code: str) -> int:
    """Job de mantenimiento: recalcula profiles agregando matches settled.

    Agrega sobre `matches` finished + `match_referees` + scores.
    """
    async with session_scope() as s:
        await s.execute(
            text(
                """
                INSERT INTO referee_bias_profile (
                    referee_id, sport_code, n_games, home_win_rate,
                    over_rate, avg_total, last_computed
                )
                SELECT
                    mr.referee_id,
                    m.sport_code,
                    COUNT(*)::int,
                    AVG(CASE WHEN m.home_score > m.away_score THEN 1.0 ELSE 0.0 END),
                    NULL,
                    AVG(m.home_score + m.away_score)::numeric(6,2),
                    NOW()
                FROM match_referees mr
                JOIN matches m ON m.id = mr.match_id
                WHERE m.sport_code = :sport
                  AND m.status = 'finished'
                  AND m.home_score IS NOT NULL
                GROUP BY mr.referee_id, m.sport_code
                HAVING COUNT(*) >= 5
                ON CONFLICT (referee_id, sport_code)
                DO UPDATE SET
                    n_games = EXCLUDED.n_games,
                    home_win_rate = EXCLUDED.home_win_rate,
                    avg_total = EXCLUDED.avg_total,
                    last_computed = NOW()
                """
            ),
            {"sport": sport_code},
        )
        r = await s.execute(
            text("SELECT COUNT(*) FROM referee_bias_profile WHERE sport_code = :s"),
            {"s": sport_code},
        )
        n = int(r.scalar_one())
    logger.info("referee_bias.recomputed", sport=sport_code, n_profiles=n)
    return n
