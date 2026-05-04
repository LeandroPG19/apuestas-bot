"""Feature store online: construye vector X para inferencia.

Lee `team_stats_rolling_{home,away}` y compone el vector con el mismo orden
que `model_obj["feature_names"]`. Retorna `None` si >20% features faltan
(fail-safe: sin features reales, `detector.py` skipea con `p_model=None`).

Principio: cero tolerancia a garbage input. Mejor skippear pick que
predecir con features inventadas.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# Alias de métricas: model feature_name token → JSON key en rolling table.
# Lista por sport_code. Si un token del feature_name no está aquí → missing.
_METRIC_ALIASES: dict[str, dict[str, str]] = {
    "nba": {
        "pts_for": "pts_for_avg",
        "pts_against": "pts_against_avg",
        "total_points": "pts_for_avg",  # proxy: usar pts_for_avg como total scored
        "win_margin": "pts_differential_avg",
        "win": "win_rate",
    },
    "mlb": {
        "runs_scored": "pts_for_avg",
        "runs_allowed": "pts_against_avg",
        "win_margin": "pts_differential_avg",
        "win": "win_rate",
    },
    "nfl": {
        "points_scored": "pts_for_avg",
        "points_allowed": "pts_against_avg",
        "win_margin": "pts_differential_avg",
        "win": "win_rate",
    },
    "nhl": {
        "goals_for": "pts_for_avg",
        "goals_against": "pts_against_avg",
        "win": "win_rate",
        "margin": "pts_differential_avg",
    },
}

# Features derivadas de `matches` (no-rolling): calendario / descanso.
# `side ∈ {home, away}`. `diff` = home - away.
_CALENDAR_FEATURES = {
    "rest_days",
    "back_to_back",
    "games_last_7d",
    "games_last_14d",
}

# Parser: captura `{metric}_roll_{window}_{venue_opt}_{side}` o `{metric}_{side}`.
# Ejemplos:
#   pts_for_avg_roll_5_home           → metric=pts_for, window=5, venue=None, side=home
#   ortg_home_roll_5_away             → metric=ortg, window=5, venue=home, side=away
#   rest_days_home                    → metric=rest_days, side=home
#   ortg_roll_5_diff                  → metric=ortg, window=5, side=diff
_ROLL_RE = re.compile(
    r"^(?P<metric>[a-z_]+?)(?:_(?P<venue>home|away))?_roll_(?P<window>\d+)_(?P<side>home|away|diff)$"
)
_SIMPLE_RE = re.compile(r"^(?P<metric>[a-z_0-9]+?)_(?P<side>home|away|diff)$")


def _parse_feature(name: str) -> dict[str, Any] | None:
    if m := _ROLL_RE.match(name):
        return {
            "kind": "roll",
            "metric": m.group("metric"),
            "window": int(m.group("window")),
            "venue": m.group("venue"),
            "side": m.group("side"),
        }
    if m := _SIMPLE_RE.match(name):
        return {
            "kind": "simple",
            "metric": m.group("metric"),
            "side": m.group("side"),
        }
    return None


_VALID_TABLES = {"team_stats_rolling_home", "team_stats_rolling_away"}


async def _fetch_rolling_by_team(
    team_id: int, sport_code: str, venue_table: str
) -> dict[int, dict[str, Any]]:
    """venue_table ∈ {'team_stats_rolling_home', 'team_stats_rolling_away'}."""
    if venue_table not in _VALID_TABLES:
        return {}
    async with session_scope() as session:
        result = await session.execute(
            text(f"""
                SELECT window_size, metrics
                FROM {venue_table}
                WHERE team_id = :tid AND sport_code = :sp
            """),
            {"tid": team_id, "sp": sport_code},
        )
        return {int(r.window_size): dict(r.metrics) for r in result.all()}


_TEAM_NAME_SYNONYMS = {
    "la ": "los angeles ",
    "ny ": "new york ",
    "nj ": "new jersey ",
    "okc ": "oklahoma city ",
    "sf ": "san francisco ",
}


def _normalize_team_name(name: str) -> str:
    """Normaliza nombre para match: lowercase + expand abbrev + strip."""
    n = name.lower().strip()
    for short, full in _TEAM_NAME_SYNONYMS.items():
        if n.startswith(short):
            n = full + n[len(short) :]
            break
    return n


async def _resolve_canonical_team_id(team_id: int, sport_code: str) -> int:
    """Si team_id tiene rolling → ya es canónico. Si no, busca alias verificado
    en team_external_id; si tampoco hay, intenta fuzzy name match contra teams
    con rolling en el mismo sport. Fail-safe: retorna original.

    El fuzzy match cubre el caso "LA Lakers" (Kambi team_id=3134) ↔
    "Los Angeles Lakers" (Pinnacle team_id=26): el resolver no encontraba alias
    en team_external_id (poblada solo con sofascore/clubelo/fbref), así que el
    detector skipeaba con `insufficient_history` en todo match con team Kambi.
    """
    async with session_scope() as session:
        r = await session.execute(
            text(
                "SELECT 1 FROM team_stats_rolling_home "
                "WHERE team_id = :tid AND sport_code = :sp LIMIT 1"
            ),
            {"tid": team_id, "sp": sport_code},
        )
        if r.first():
            return team_id
        r2 = await session.execute(
            text(
                """
                SELECT team_id FROM team_external_id
                WHERE source IN ('sofascore', 'pinnacle', 'odds_api')
                  AND external_id = :eid AND verified = true
                ORDER BY confidence DESC NULLS LAST
                LIMIT 1
                """
            ),
            {"eid": str(team_id)},
        )
        row = r2.first()
        if row:
            return int(row.team_id)

        # Fuzzy name match: el team original no tiene rolling. Buscar otro team
        # del mismo sport con rolling cuyo nombre normalizado matchee.
        name_row = (
            await session.execute(
                text("SELECT name FROM teams WHERE id = :tid AND sport_code = :sp"),
                {"tid": team_id, "sp": sport_code},
            )
        ).first()
        if name_row is None or not name_row.name:
            return team_id
        norm_target = _normalize_team_name(name_row.name)
        candidates = await session.execute(
            text(
                """
                SELECT t.id, t.name
                FROM teams t
                JOIN team_stats_rolling_home r ON r.team_id = t.id AND r.sport_code = t.sport_code
                WHERE t.sport_code = :sp AND t.id <> :tid
                """
            ),
            {"sp": sport_code, "tid": team_id},
        )
        for c in candidates:
            if _normalize_team_name(c.name) == norm_target:
                logger.info(
                    "feature_store.fuzzy_alias_resolved",
                    sport=sport_code,
                    derivative_id=team_id,
                    derivative_name=name_row.name,
                    canonical_id=int(c.id),
                    canonical_name=c.name,
                )
                return int(c.id)
    return team_id


async def _fetch_calendar_features(
    *, home_team_id: int, away_team_id: int, match_start: Any
) -> dict[str, float]:
    """Calcula rest_days, back_to_back, games_last_Nd para ambos equipos."""
    out: dict[str, float] = {}
    async with session_scope() as session:
        for side, tid in (("home", home_team_id), ("away", away_team_id)):
            result = await session.execute(
                text("""
                    SELECT
                        EXTRACT(EPOCH FROM (:ts - MAX(m.start_time))) / 86400.0 AS rest_days,
                        COUNT(*) FILTER (WHERE m.start_time >= :ts - INTERVAL '7 days') AS g7,
                        COUNT(*) FILTER (WHERE m.start_time >= :ts - INTERVAL '14 days') AS g14
                    FROM matches m
                    WHERE m.status = 'finished'
                      AND m.start_time < :ts
                      AND (m.home_team_id = :tid OR m.away_team_id = :tid)
                      AND m.start_time >= :ts - INTERVAL '30 days'
                """),
                {"ts": match_start, "tid": tid},
            )
            row = result.first()
            rest = float(row.rest_days) if row and row.rest_days is not None else 5.0
            g7 = int(row.g7) if row and row.g7 is not None else 0
            g14 = int(row.g14) if row and row.g14 is not None else 0
            out[f"rest_days_{side}"] = rest
            out[f"back_to_back_{side}"] = 1.0 if rest < 1.5 else 0.0
            out[f"games_last_7d_{side}"] = float(g7)
            out[f"games_last_14d_{side}"] = float(g14)
    return out


async def build_match_features(
    *,
    sport_code: str,
    home_team_id: int,
    away_team_id: int,
    match_start: Any,
    feature_names: list[str],
    min_coverage: float | None = None,
) -> np.ndarray | None:
    """Construye vector X (shape=(n_features,)) según `feature_names`.

    Retorna `None` si <min_coverage*100% features resolvibles → el caller
    debe skipear la inferencia (política fail-safe, nunca garbage input).

    `min_coverage` configurable via env `APUESTAS_FEATURE_MIN_COVERAGE`
    (default 0.80). Bajar a 0.30 cuando se mezclan modelos viejos con
    features nuevas post-migración.
    """
    import os as _os

    if min_coverage is None:
        min_coverage = float(_os.environ.get("APUESTAS_FEATURE_MIN_COVERAGE", "0.80"))
    if sport_code not in _METRIC_ALIASES:
        logger.debug("feature_store.unsupported_sport", sport=sport_code)
        return None

    aliases = _METRIC_ALIASES[sport_code]
    # Resuelve alias (sofascore/pinnacle/odds_api) → canonical team_id interno.
    canonical_home = await _resolve_canonical_team_id(home_team_id, sport_code)
    canonical_away = await _resolve_canonical_team_id(away_team_id, sport_code)
    home_roll = await _fetch_rolling_by_team(canonical_home, sport_code, "team_stats_rolling_home")
    away_roll = await _fetch_rolling_by_team(canonical_away, sport_code, "team_stats_rolling_away")
    if not home_roll or not away_roll:
        logger.debug(
            "feature_store.no_rolling",
            sport=sport_code,
            home=home_team_id,
            away=away_team_id,
            canonical_home=canonical_home,
            canonical_away=canonical_away,
        )
        return None

    # Calendar features (rest_days, back_to_back, ...) si el modelo las pide
    needs_calendar = any(
        any(f.startswith(f"{cf}_") for cf in _CALENDAR_FEATURES) for f in feature_names
    )
    calendar: dict[str, float] = {}
    if needs_calendar:
        try:
            calendar = await _fetch_calendar_features(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                match_start=match_start,
            )
        except Exception as exc:
            logger.debug("feature_store.calendar_fail", error=str(exc)[:120])
            calendar = {}

    values: list[float] = []
    missing: list[str] = []

    def _lookup(metric: str, window: int, side: str, venue: str | None) -> float | None:
        """side ∈ {home,away,diff}; venue ∈ {None,home,away} (para venue splits)."""
        # Venue splits (ortg_home_roll_5_...) no están en la rolling table base
        # → no soportados por este feature store; return None.
        if venue is not None:
            return None
        rolling_map = home_roll if side == "home" else away_roll if side == "away" else None
        if side == "diff":
            h = _lookup(metric, window, "home", None)
            a = _lookup(metric, window, "away", None)
            if h is None or a is None:
                return None
            return h - a
        if rolling_map is None:
            return None
        metrics = rolling_map.get(window)
        if metrics is None:
            return None
        json_key = aliases.get(metric)
        if json_key is None:
            return None
        val = metrics.get(json_key)
        return float(val) if val is not None else None

    for name in feature_names:
        # 1) Calendar / simple non-rolling
        if name in calendar:
            values.append(calendar[name])
            continue
        parsed = _parse_feature(name)
        if parsed is None:
            missing.append(name)
            values.append(0.0)
            continue
        if parsed["kind"] == "roll":
            v = _lookup(
                parsed["metric"],
                parsed["window"],
                parsed["side"],
                parsed.get("venue"),
            )
            if v is None:
                missing.append(name)
                values.append(0.0)
            else:
                values.append(v)
        elif parsed["kind"] == "simple" and parsed["metric"] in _CALENDAR_FEATURES:
            # Calendar ya hidratadas arriba; si llegamos aquí → missing.
            missing.append(name)
            values.append(0.0)
        else:
            missing.append(name)
            values.append(0.0)

    coverage = 1.0 - (len(missing) / max(1, len(feature_names)))
    if coverage < min_coverage:
        logger.info(
            "feature_store.insufficient_coverage",
            sport=sport_code,
            coverage=round(coverage, 3),
            min_coverage=min_coverage,
            missing_sample=missing[:10],
        )
        return None

    logger.debug(
        "feature_store.built",
        sport=sport_code,
        n_features=len(values),
        coverage=round(coverage, 3),
    )
    return np.array(values, dtype=np.float64)
