"""Runtime feature builder usando el MISMO pipeline que training.

Path B del fix `0 picks NBA/MLB`: en lugar de leer `team_stats_rolling_*` JSON
con 5 keys (que no cubre las 40-60+ features que el modelo MLflow espera),
reconstruye el vector de features ejecutando exactamente el pipeline del
training (`build_nba_feature_frame` / `build_mlb_feature_frame` / NFL local)
sobre matches+team_games traídos de la DB con ventana de lookback.

Resultado: cero skew train/inference + coverage muy superior al feature_store
legacy. Si la query devuelve filas, el vector es idéntico al de training.

Cache: memoiza por `(sport, home, away, match_date, feature_set_hash)` en
Valkey con TTL 30min para evitar recomputar el rolling cuando el mismo match
aparece en múltiples markets (h2h, spreads, totals).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
from sqlalchemy import text

from apuestas.cache import cache_get, cache_set
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS = 1800  # 30 min
# 730d cubre 2 temporadas — necesario para MLB/NBA inicio de temporada
# (la T-1 completa garantiza ≥30 matches/equipo). NFL en off-season seguirá
# devolviendo None vía _MIN_TEAM_HISTORY, que es el comportamiento correcto.
_DEFAULT_LOOKBACK_DAYS = 730
# Bajado 5→3: MLB inicio temporada (abril) tiene 4-5 matches/equipo en últimos
# 730d cuando aún no se cargó la temporada T-1 completa. Con 3 matches el
# rolling 5/10/20 sigue produciendo features útiles (NaN en w=10/20 pero w=5 sí).
_MIN_TEAM_HISTORY = 3  # min matches por equipo para que rolling sea útil


def _feature_set_hash(feature_names: list[str]) -> str:
    """Hash corto de la lista ordenada de feature_names — invalida cache si el
    modelo cambió su contrato (re-train con feature set distinto)."""
    joined = "|".join(feature_names)
    return hashlib.md5(joined.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


async def _resolve_canonical_team_id(team_id: int, sport_code: str) -> int:
    """Resuelve a team_id canónico con history vía 3 fallbacks:

    1. El team_id ya aparece en matches finished → no-op
    2. team_external_id verified mapping
    3. **Fuzzy match por nombre** — busca otro team del mismo sport con nombre
       similar (e.g. "Philadelphia Phillies" id=763 → "PHI" id=3426 que tiene 135
       matches). Crítico porque la DB tiene duplicados de identidad sin mapeo en
       team_external_id (la tabla está casi vacía, solo 4 rows).
    """
    async with session_scope() as session:
        # Caso normal: el team_id ya tiene matches finished propios
        r = await session.execute(
            text(
                """
                SELECT 1 FROM matches
                WHERE sport_code = :sp
                  AND status = 'finished'
                  AND (home_team_id = :tid OR away_team_id = :tid)
                LIMIT 1
                """
            ),
            {"tid": team_id, "sp": sport_code},
        )
        if r.first():
            return team_id

        # Fallback 1: alias verificado en team_external_id
        r2 = await session.execute(
            text(
                """
                SELECT team_id FROM team_external_id
                WHERE source IN ('sofascore', 'pinnacle', 'odds_api', 'mlb_stats')
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

        # Fallback 2: fuzzy match por nombre. Trae el nombre del team y busca
        # otros teams del mismo sport cuyo nombre sea sub-string o tenga overlap
        # significativo, ordenados por #matches finished DESC.
        name_row = (
            await session.execute(
                text("SELECT name FROM teams WHERE id = :tid"),
                {"tid": team_id},
            )
        ).first()
        if name_row is None or not name_row.name:
            return team_id
        target_name = str(name_row.name).strip().lower()

        # Genera tokens "fuertes" (>=3 chars, sin stopwords comunes)
        stop = {"the", "fc", "cf", "club", "de", "city", "united"}
        tokens = [t for t in target_name.replace(",", " ").split() if len(t) >= 3 and t not in stop]
        if not tokens:
            return team_id

        # Score por overlap de tokens + bonus si nombre del candidato es
        # sub-string del target (e.g. 'PHI' ⊂ 'philadelphia phillies' después
        # de la abreviación canónica). Limitamos a teams del MISMO sport con
        # ≥10 matches finished para evitar ruido.
        candidates = (
            await session.execute(
                text(
                    """
                    WITH counts AS (
                      SELECT team_id, COUNT(*) AS n FROM (
                        SELECT home_team_id AS team_id FROM matches
                          WHERE sport_code=:sp AND status='finished'
                            AND start_time > NOW() - INTERVAL '730 days'
                        UNION ALL
                        SELECT away_team_id FROM matches
                          WHERE sport_code=:sp AND status='finished'
                            AND start_time > NOW() - INTERVAL '730 days'
                      ) t GROUP BY 1 HAVING COUNT(*) >= 10
                    )
                    SELECT t.id, t.name, c.n
                    FROM counts c JOIN teams t ON t.id = c.team_id
                    WHERE t.id <> :tid
                    """
                ),
                {"sp": sport_code, "tid": team_id},
            )
        ).all()

        # Genera abreviación canónica del target (e.g. "Philadelphia Phillies" → "ph")
        # tomando primer carácter de cada token fuerte.
        target_abbrev = "".join(t[0] for t in tokens) if tokens else ""

        best_id: int | None = None
        best_score = 0.0
        for cand in candidates:
            cand_name = str(cand.name or "").strip().lower()
            if not cand_name:
                continue
            score = 0.0
            # 1) Match exacto de abreviación (3-4 chars de candidato == iniciales target)
            if len(cand_name) <= 4 and cand_name == target_abbrev:
                score = 2.0 + (cand.n / 10000.0)
            # 2) Match de candidato corto (≤4 chars) que sea PREFIJO de algún token
            elif len(cand_name) <= 4:
                if any(tk.startswith(cand_name) for tk in tokens):
                    score = 1.0 + (cand.n / 10000.0)
            # 3) Token overlap (≥50% de los tokens del target presentes en candidato)
            else:
                cand_tokens = set(cand_name.replace(",", " ").split())
                overlap = len(set(tokens) & cand_tokens)
                if overlap == 0:
                    continue
                ratio = overlap / max(len(tokens), 1)
                if ratio < 0.5:
                    continue
                score = ratio + (cand.n / 10000.0)
            if score > best_score:
                best_score = score
                best_id = int(cand.id)

        if best_id is not None and best_score >= 1.0:
            logger.info(
                "runtime_features.canonical_resolved_by_name",
                original=team_id,
                resolved=best_id,
                target_name=target_name,
                score=round(best_score, 3),
            )
            return best_id

    return team_id


async def _fetch_history(
    *,
    sport_code: str,
    home_team_id: int,
    away_team_id: int,
    match_start: datetime,
    lookback_days: int,
) -> list[dict[str, Any]]:
    """Trae matches finished de los 2 equipos en `[match_start - N días, match_start)`.

    Misma estructura que `load_*_training_data` (matches table) pero acotada
    por team_ids + ventana temporal as-of.
    """
    cutoff = match_start - timedelta(days=lookback_days)
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                SELECT id, external_id, home_team_id, away_team_id, start_time,
                       venue_id, home_score, away_score, status, season
                FROM matches
                WHERE sport_code = :sport
                  AND status = 'finished'
                  AND start_time >= :cutoff
                  AND start_time < :ts
                  AND (home_team_id = ANY(:tids) OR away_team_id = ANY(:tids))
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                ORDER BY start_time
                """
            ),
            {
                "sport": sport_code,
                "cutoff": cutoff,
                "ts": match_start,
                "tids": [int(home_team_id), int(away_team_id)],
            },
        )
        return [dict(r._mapping) for r in result.all()]


def _virtual_match_row(
    *, home_team_id: int, away_team_id: int, match_start: datetime
) -> dict[str, Any]:
    """Fila placeholder del match objetivo. Score=None, status='scheduled'.

    Necesaria para que el join `(home_team_id, start_time)` matchee la row
    en team_features (que también lleva start_time=match_start).
    """
    return {
        "id": -1,  # virtual id
        "external_id": "virtual",
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
        "start_time": match_start,
        "venue_id": None,
        "home_score": None,
        "away_score": None,
        "status": "scheduled",
        "season": None,
    }


def _build_team_games_nba(matches_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replica `load_nba_training_data` (train_nba.py:88-138) — 2 rows por match."""
    rows: list[dict[str, Any]] = []
    nan = float("nan")
    for r in matches_rows:
        if r["home_score"] is None or r["away_score"] is None:
            continue
        margin_home = float(r["home_score"]) - float(r["away_score"])
        total = float(r["home_score"]) + float(r["away_score"])
        for tid, is_home, pts, margin in (
            (r["home_team_id"], True, float(r["home_score"]), margin_home),
            (r["away_team_id"], False, float(r["away_score"]), -margin_home),
        ):
            rows.append(
                {
                    "team_id": int(tid),
                    "game_id": int(r["id"]),
                    "start_time": r["start_time"],
                    "is_home": is_home,
                    "pts": pts,
                    "win_margin": margin,
                    "total_points": total,
                    "fgm": nan,
                    "fga": nan,
                    "fg3m": nan,
                    "ftm": nan,
                    "fta": nan,
                    "oreb": nan,
                    "dreb": nan,
                    "tov": nan,
                    "ortg": nan,
                    "drtg": nan,
                }
            )
    return rows


def _virtual_team_games_nba(
    *, home_team_id: int, away_team_id: int, match_start: datetime
) -> list[dict[str, Any]]:
    """Virtual rows del match objetivo para que el rolling se cierre justo antes."""
    nan = float("nan")
    return [
        {
            "team_id": int(home_team_id),
            "game_id": -1,
            "start_time": match_start,
            "is_home": True,
            "pts": nan,
            "win_margin": nan,
            "total_points": nan,
            "fgm": nan,
            "fga": nan,
            "fg3m": nan,
            "ftm": nan,
            "fta": nan,
            "oreb": nan,
            "dreb": nan,
            "tov": nan,
            "ortg": nan,
            "drtg": nan,
        },
        {
            "team_id": int(away_team_id),
            "game_id": -1,
            "start_time": match_start,
            "is_home": False,
            "pts": nan,
            "win_margin": nan,
            "total_points": nan,
            "fgm": nan,
            "fga": nan,
            "fg3m": nan,
            "ftm": nan,
            "fta": nan,
            "oreb": nan,
            "dreb": nan,
            "tov": nan,
            "ortg": nan,
            "drtg": nan,
        },
    ]


def _build_team_games_mlb(matches_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replica `load_mlb_training_data` (train_mlb.py:67-85)."""
    rows: list[dict[str, Any]] = []
    for r in matches_rows:
        if r["home_score"] is None or r["away_score"] is None:
            continue
        total = float(r["home_score"]) + float(r["away_score"])
        for tid, runs_scored, runs_allowed in (
            (r["home_team_id"], float(r["home_score"]), float(r["away_score"])),
            (r["away_team_id"], float(r["away_score"]), float(r["home_score"])),
        ):
            rows.append(
                {
                    "team_id": int(tid),
                    "game_date": r["start_time"],
                    "runs_scored": runs_scored,
                    "runs_allowed": runs_allowed,
                    "total_runs": total,
                }
            )
    return rows


def _virtual_team_games_mlb(
    *, home_team_id: int, away_team_id: int, match_start: datetime
) -> list[dict[str, Any]]:
    nan = float("nan")
    return [
        {
            "team_id": int(home_team_id),
            "game_date": match_start,
            "runs_scored": nan,
            "runs_allowed": nan,
            "total_runs": nan,
        },
        {
            "team_id": int(away_team_id),
            "game_date": match_start,
            "runs_scored": nan,
            "runs_allowed": nan,
            "total_runs": nan,
        },
    ]


def _build_team_games_nfl(matches_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replica `load_nfl_training_data` (train_nfl.py:76-103)."""
    rows: list[dict[str, Any]] = []
    for r in matches_rows:
        if r["home_score"] is None or r["away_score"] is None:
            continue
        hs = float(r["home_score"])
        asc = float(r["away_score"])
        rows.append(
            {
                "team_id": int(r["home_team_id"]),
                "game_date": r["start_time"],
                "points_scored": hs,
                "points_allowed": asc,
                "win_margin": hs - asc,
                "win": 1.0 if hs > asc else 0.0,
            }
        )
        rows.append(
            {
                "team_id": int(r["away_team_id"]),
                "game_date": r["start_time"],
                "points_scored": asc,
                "points_allowed": hs,
                "win_margin": asc - hs,
                "win": 1.0 if asc > hs else 0.0,
            }
        )
    return rows


def _virtual_team_games_nfl(
    *, home_team_id: int, away_team_id: int, match_start: datetime
) -> list[dict[str, Any]]:
    nan = float("nan")
    return [
        {
            "team_id": int(home_team_id),
            "game_date": match_start,
            "points_scored": nan,
            "points_allowed": nan,
            "win_margin": nan,
            "win": nan,
        },
        {
            "team_id": int(away_team_id),
            "game_date": match_start,
            "points_scored": nan,
            "points_allowed": nan,
            "win_margin": nan,
            "win": nan,
        },
    ]


def _build_features_dispatch(
    *,
    sport_code: str,
    matches_df: pl.DataFrame,
    team_games_df: pl.DataFrame,
) -> pl.DataFrame:
    """Despacha al pipeline de features correcto según el sport.

    Reusa exactamente las funciones de training para garantizar cero skew.
    Importa lazy para evitar ciclos y módulos pesados (mlflow) en hot path.
    """
    from apuestas.features.common import add_elo_features

    if sport_code == "nba":
        from apuestas.features.nba import build_nba_feature_frame

        df = build_nba_feature_frame(matches_df, team_games_df)
        df = add_elo_features(df, sport="nba")
        # Sprint 14 #149 context features (rest/b2b/travel) — se importan de
        # train_nba para mantener idempotencia con el pipeline de entrenamiento.
        try:
            from apuestas.ml.train_nba import _add_nba_context_columns_sync

            df = _add_nba_context_columns_sync(df, matches_df)
        except Exception as exc:
            logger.debug("runtime_features.nba_context_skip", error=str(exc)[:120])
        return df

    if sport_code == "mlb":
        from apuestas.features.mlb import build_mlb_feature_frame

        df = build_mlb_feature_frame(matches_df, team_games_df)
        df = add_elo_features(df, sport="mlb")
        try:
            from apuestas.ml.train_mlb import _add_mlb_context_sync

            df = _add_mlb_context_sync(df, matches_df)
        except Exception as exc:
            logger.debug("runtime_features.mlb_context_skip", error=str(exc)[:120])
        return df

    if sport_code == "nfl":
        # NFL training usa una versión LOCAL de build_nfl_feature_frame en
        # train_nfl.py (FEATURE_SET_NAME='nfl_v2_basic'), no la de features/nfl.py.
        # Importamos la función local para reproducir exactamente el pipeline.
        from apuestas.ml.train_nfl import build_nfl_feature_frame as _nfl_local

        df = _nfl_local(matches_df, team_games_df)
        df = add_elo_features(df, sport="nfl")
        return df

    msg = f"Sport no soportado por runtime_features: {sport_code}"
    raise ValueError(msg)


async def build_match_features_from_raw(
    *,
    sport_code: str,
    home_team_id: int,
    away_team_id: int,
    match_start: datetime,
    feature_names: list[str],
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    use_cache: bool = True,
) -> np.ndarray | None:
    """Construye vector X reproduciendo el pipeline de training en runtime.

    Args:
        sport_code: 'nba' | 'mlb' | 'nfl'
        home_team_id: team id del equipo local
        away_team_id: team id del equipo visitante
        match_start: timestamp del kickoff (UTC)
        feature_names: lista en el orden EXACTO que espera el modelo MLflow
        lookback_days: ventana de historia (default 1 temporada)
        use_cache: leer/escribir Valkey (default True)

    Returns:
        np.ndarray shape `(len(feature_names),)` con missing rellenado en 0.0,
        o None si:
          - sport no soportado
          - <_MIN_TEAM_HISTORY matches por equipo (insuficiente para rolling)
          - el join no encuentra la fila virtual del match objetivo
          - >50% de feature_names faltan en el DataFrame final

    Política fail-safe (cero garbage input): preferimos retornar None y que
    el detector skipee con `skip_reason='insufficient_history'` antes que
    predecir con vector mayoritariamente lleno de ceros.
    """
    if sport_code not in {"nba", "mlb", "nfl"}:
        return None
    if not feature_names:
        return None

    # Cache key: incluye hash del feature_set para invalidar cuando el modelo cambia
    cache_key: str | None = None
    if use_cache:
        date_key = match_start.astimezone(UTC).strftime("%Y%m%d")
        fs_hash = _feature_set_hash(feature_names)
        cache_key = (
            f"runtime_features:{sport_code}:{home_team_id}:{away_team_id}:{date_key}:{fs_hash}"
        )
        cached = await cache_get(cache_key)
        if cached is not None:
            try:
                return np.asarray(cached, dtype=np.float64)
            except Exception:
                pass

    # Resolución canónica de team_ids (defensa contra variantes de identidad
    # MLB/NBA que tienen mismo equipo bajo distintos ids — ver memoria
    # identity_resolution_postmortem.md). Si el id ya tiene history → no-op.
    canonical_home = await _resolve_canonical_team_id(home_team_id, sport_code)
    canonical_away = await _resolve_canonical_team_id(away_team_id, sport_code)

    matches_rows = await _fetch_history(
        sport_code=sport_code,
        home_team_id=canonical_home,
        away_team_id=canonical_away,
        match_start=match_start,
        lookback_days=lookback_days,
    )

    # Coverage check temprano sobre IDs CANÓNICOS (defensa identidad MLB/NBA).
    home_count = sum(
        1 for r in matches_rows if canonical_home in (r["home_team_id"], r["away_team_id"])
    )
    away_count = sum(
        1 for r in matches_rows if canonical_away in (r["home_team_id"], r["away_team_id"])
    )
    if home_count < _MIN_TEAM_HISTORY or away_count < _MIN_TEAM_HISTORY:
        logger.info(
            "runtime_features.insufficient_team_history",
            sport=sport_code,
            home=home_team_id,
            away=away_team_id,
            canonical_home=canonical_home,
            canonical_away=canonical_away,
            home_matches=home_count,
            away_matches=away_count,
            min_required=_MIN_TEAM_HISTORY,
        )
        return None

    # Append fila virtual del match objetivo para que el join matchee.
    # Usar IDs canónicos para que coincida con team_games virtual rows.
    matches_rows.append(
        _virtual_match_row(
            home_team_id=canonical_home,
            away_team_id=canonical_away,
            match_start=match_start,
        )
    )

    # team_games dispatch por sport (IDs canónicos en virtual rows)
    if sport_code == "nba":
        team_rows = _build_team_games_nba(matches_rows[:-1])
        team_rows.extend(
            _virtual_team_games_nba(
                home_team_id=canonical_home,
                away_team_id=canonical_away,
                match_start=match_start,
            )
        )
    elif sport_code == "mlb":
        team_rows = _build_team_games_mlb(matches_rows[:-1])
        team_rows.extend(
            _virtual_team_games_mlb(
                home_team_id=canonical_home,
                away_team_id=canonical_away,
                match_start=match_start,
            )
        )
    else:  # nfl
        team_rows = _build_team_games_nfl(matches_rows[:-1])
        team_rows.extend(
            _virtual_team_games_nfl(
                home_team_id=canonical_home,
                away_team_id=canonical_away,
                match_start=match_start,
            )
        )

    if not team_rows:
        logger.info("runtime_features.empty_team_games", sport=sport_code)
        return None

    # infer_schema_length=None fuerza scan completo del rowset para evitar
    # "could not append value: X of type: i64 to the builder" cuando el primer
    # batch infiere f64 (por NaN del virtual row) y luego aparece un int.
    matches_df = pl.DataFrame(matches_rows, infer_schema_length=None)
    team_games_df = pl.DataFrame(team_rows, infer_schema_length=None)

    try:
        features_df = _build_features_dispatch(
            sport_code=sport_code,
            matches_df=matches_df,
            team_games_df=team_games_df,
        )
    except Exception as exc:
        logger.warning(
            "runtime_features.dispatch_fail",
            sport=sport_code,
            error=str(exc)[:160],
        )
        return None

    # Filtrar al row del match objetivo (start_time exacto)
    target_row = features_df.filter(pl.col("start_time") == match_start)
    if target_row.height == 0:
        logger.info(
            "runtime_features.virtual_row_missing",
            sport=sport_code,
            home=home_team_id,
            away=away_team_id,
        )
        return None
    if target_row.height > 1:
        target_row = target_row.head(1)

    # Extraer columnas en orden estricto de feature_names
    available_cols = set(target_row.columns)
    missing: list[str] = []
    values: list[float] = []
    row_dict = target_row.row(0, named=True)
    for name in feature_names:
        if name not in available_cols:
            missing.append(name)
            values.append(0.0)
            continue
        v = row_dict.get(name)
        if v is None:
            missing.append(name)
            values.append(0.0)
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            missing.append(name)
            values.append(0.0)
            continue
        if np.isnan(fv):
            missing.append(name)
            values.append(0.0)
        else:
            values.append(fv)

    coverage = 1.0 - (len(missing) / max(1, len(feature_names)))
    if coverage < 0.5:
        logger.info(
            "runtime_features.low_coverage",
            sport=sport_code,
            coverage=round(coverage, 3),
            n_missing=len(missing),
            sample_missing=missing[:8],
        )
        return None

    arr = np.array(values, dtype=np.float64)
    logger.debug(
        "runtime_features.built",
        sport=sport_code,
        n_features=len(values),
        coverage=round(coverage, 3),
        history_matches=len(matches_rows) - 1,
    )

    if use_cache and cache_key is not None:
        await cache_set(cache_key, arr.tolist(), ttl_seconds=_CACHE_TTL_SECONDS)

    return arr


__all__ = ["build_match_features_from_raw"]
