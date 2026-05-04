"""Features fútbol — xG (Understat/StatsBomb/FBref) + Elo clubelo + Dixon-Coles.

Blueprint §6: librería principal `penaltyblog` (Dixon-Coles + Bivariate
Poisson + Bayesian hierarchical + decaimiento ξ=0.0018). Aquí features
complementarias para stacker LightGBM sobre residuos DC.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import poisson as scipy_poisson

from apuestas.features.common import (
    days_since_last,
    rolling_mean_prev,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

FEATURE_SET_NAME = "soccer_v1"
WINDOWS = [5, 10, 20]

# Dixon-Coles parameters (defaults si no hay fitted model)
_DC_HOME_ADVANTAGE = 1.25
_DC_LG_AVG_HOME = 1.45  # goles promedio home por partido (premier/laliga)
_DC_LG_AVG_AWAY = 1.15  # goles promedio away
_DC_RHO = -0.1  # DC correction coefficient (negativo en empates bajos)
_DC_MAX_GOALS = 10


def _dc_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float = _DC_RHO) -> float:
    """Dixon-Coles tau correction para scores bajos (0-0, 0-1, 1-0, 1-1)."""
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _dixon_coles_score_matrix(
    home_team_id: int, away_team_id: int
) -> tuple[np.ndarray, float, float] | None:
    """Devuelve (score_matrix, lam_home, lam_away) para los dos teams, o None.

    Compartido entre h2h/totals/btts para evitar recomputar cuando el agente
    pide múltiples markets sobre el mismo partido.
    """
    import psycopg

    from apuestas.config import get_settings

    cfg = get_settings()
    dsn = (
        f"host={cfg.database.postgres_host} "
        f"port={cfg.database.postgres_host_port} "
        f"dbname={cfg.database.postgres_db} "
        f"user={cfg.database.postgres_user} "
        f"password={cfg.database.postgres_password.get_secret_value()}"
    )
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT team_id, attack_rating, defense_rating, n_matches "
                "FROM team_strength_bayesian WHERE team_id = ANY(%s)",
                ([home_team_id, away_team_id],),
            )
            rows = {int(r[0]): (float(r[1]), float(r[2]), int(r[3])) for r in cur.fetchall()}

            # Fallback: si el team_id directo NO tiene suficientes matches
            # (n<3) buscar entre teams duplicados por nombre canonical y usar
            # el de mayor n. Bug detectado 2026-05-02: Inter (1504, n=224),
            # Internazionale (901, n=1), Inter Milan (18191, n=1) son el
            # mismo equipo pero con FK distintos. El match #516 apunta a
            # 901 → DC retornaba None aunque las stats existían en 1504.
            for tid in (home_team_id, away_team_id):
                row = rows.get(tid)
                if row is not None and row[2] >= 3:
                    continue
                # Fallback ESTRICTO: el candidato debe pertenecer a la MISMA
                # liga que el team query (`teams.league_id`). Sin esto,
                # similarity 0.3 confundía Palmeiras → Las Palmas (España) y
                # Santos → Santander → predicciones completamente erróneas
                # (Santos 84% vs Palmeiras local, ridículo). Adicional: el
                # nombre query debe ser substring del candidato O viceversa
                # (no solo trigram), garantizando que "Inter" → "Internazionale"
                # match pero "Palmeiras" no calza a "Las Palmas".
                cur.execute(
                    """
                    SELECT tsb.team_id, tsb.attack_rating, tsb.defense_rating,
                           tsb.n_matches
                    FROM team_strength_bayesian tsb
                    JOIN teams t ON t.id = tsb.team_id
                    JOIN teams q ON q.id = %s
                    WHERE t.sport_code = 'soccer'
                      AND tsb.n_matches >= 10
                      AND t.id <> q.id
                      AND t.league_id IS NOT NULL
                      AND t.league_id = q.league_id
                      AND (
                        LOWER(t.name) LIKE '%%' || LOWER(q.name) || '%%'
                        OR LOWER(q.name) LIKE '%%' || LOWER(t.name) || '%%'
                      )
                    ORDER BY tsb.n_matches DESC
                    LIMIT 1
                    """,
                    (tid,),
                )
                fallback = cur.fetchone()
                if fallback is not None:
                    rows[tid] = (
                        float(fallback[1]),
                        float(fallback[2]),
                        int(fallback[3]),
                    )
    except Exception as exc:
        logger.debug("dixon_coles.db_fetch_fail", error=str(exc)[:120])
        return None

    if home_team_id not in rows or away_team_id not in rows:
        return None

    attack_h, defense_h, n_h = rows[home_team_id]
    attack_a, defense_a, n_a = rows[away_team_id]

    if n_h < 3 or n_a < 3:
        return None

    lam_home = attack_h * defense_a * _DC_LG_AVG_HOME * _DC_HOME_ADVANTAGE
    lam_away = attack_a * defense_h * _DC_LG_AVG_AWAY
    lam_home = max(0.1, min(lam_home, 6.0))
    lam_away = max(0.1, min(lam_away, 6.0))

    max_g = _DC_MAX_GOALS
    matrix = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = scipy_poisson.pmf(i, lam_home) * scipy_poisson.pmf(j, lam_away)
            matrix[i, j] = p * _dc_tau(i, j, lam_home, lam_away)
    matrix = np.maximum(matrix, 0.0)
    total = matrix.sum()
    if total <= 0:
        return None
    matrix /= total
    return matrix, lam_home, lam_away


def dixon_coles_predict(home_team_id: int, away_team_id: int) -> dict[str, float] | None:
    """Predice `[p_home, p_draw, p_away]` vía Dixon-Coles usando strengths bayesianos.

    Retorna `None` si alguno de los equipos no tiene strength registrado o
    tiene <3 partidos (fail-safe: sin priors, no forzamos predicción).
    """
    out = _dixon_coles_score_matrix(home_team_id, away_team_id)
    if out is None:
        return None
    matrix, _lam_h, _lam_a = out
    max_g = _DC_MAX_GOALS

    p_home = float(sum(matrix[i, j] for i in range(max_g + 1) for j in range(i)))
    p_draw = float(matrix.trace())
    p_away = max(0.0, 1.0 - p_home - p_draw)
    s = p_home + p_draw + p_away
    return {
        "p_home": p_home / s,
        "p_draw": p_draw / s,
        "p_away": p_away / s,
    }


def dixon_coles_predict_total(
    home_team_id: int, away_team_id: int, line: float
) -> dict[str, float] | None:
    """Predice `{"over": p, "under": p}` para totals goles via DC matrix."""
    out = _dixon_coles_score_matrix(home_team_id, away_team_id)
    if out is None:
        return None
    matrix, _lh, _la = out
    n = matrix.shape[0]
    p_over = 0.0
    p_under = 0.0
    for h in range(n):
        for a in range(n):
            t = h + a
            if t > line:
                p_over += float(matrix[h, a])
            elif t < line:
                p_under += float(matrix[h, a])
    s = p_over + p_under
    if s <= 0:
        return None
    return {"over": p_over / s, "under": p_under / s}


def dixon_coles_predict_btts(home_team_id: int, away_team_id: int) -> dict[str, float] | None:
    """Predice `{"yes": p, "no": p}` BTTS via DC matrix."""
    out = _dixon_coles_score_matrix(home_team_id, away_team_id)
    if out is None:
        return None
    matrix, _lh, _la = out
    p_yes = float(matrix[1:, 1:].sum())
    p_no = max(0.0, 1.0 - p_yes)
    return {"yes": p_yes, "no": p_no}


TEAM_METRICS = (
    "xg_for",
    "xg_against",
    "goals_for",
    "goals_against",
    "shots_per_game",
    "shots_on_target",
    "possession_pct",
    "pass_completion_pct",
    "ppda",  # passes per defensive action
    "deep_completions",
)


def team_rolling_features(team_games: pl.DataFrame) -> pl.DataFrame:
    result = team_games.sort(["team_id", "game_date"])
    for metric in TEAM_METRICS:
        if metric in result.columns:
            result = rolling_mean_prev(
                result,
                by="team_id",
                order="game_date",
                value=metric,
                windows=WINDOWS,
            )
    # Dixon-Coles decay weight (años desde partido)
    result = (
        result.with_columns(
            pl.col("game_date").max().over("team_id").alias("_latest_date"),
        )
        .with_columns(
            ((pl.col("_latest_date") - pl.col("game_date")).dt.total_days() / 365.0).alias(
                "years_ago"
            )
        )
        .with_columns(
            # ξ=0.0018 per day → convert años
            (-0.0018 * 365.0 * pl.col("years_ago")).exp().alias("dc_decay_weight")
        )
        .drop("_latest_date")
    )

    result = days_since_last(result, by="team_id", order="game_date", name="rest_days")
    return result


def add_elo_features(
    df: pl.DataFrame,
    elo_ratings: dict[int, float],
) -> pl.DataFrame:
    """Añade Elo overall desde dict team_id → Elo."""
    if not elo_ratings or "team_id" not in df.columns:
        return df
    elo_df = pl.DataFrame(
        [{"team_id": int(k), "elo_rating": float(v)} for k, v in elo_ratings.items()]
    )
    return df.join(elo_df, on="team_id", how="left")


def build_soccer_feature_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    elo_ratings: dict[int, float] | None = None,
) -> pl.DataFrame:
    """Pipeline completo fútbol."""
    feats = team_rolling_features(team_games)
    if elo_ratings:
        feats = add_elo_features(feats, elo_ratings)

    # Sprint 11 Fase E — xT rolling (opt-in). Si team_games tiene columnas
    # de shots/possession/progressive moves, enriquece con threat features.
    import os as _os

    if _os.environ.get("APUESTAS_ENABLE_XT", "true").lower() == "true":
        try:
            from apuestas.features.soccer_xt import add_xt_rolling

            feats = add_xt_rolling(feats)
        except Exception:
            pass  # fail-silent si falta columna

    base_cols = [c for c in feats.columns if any(c.endswith(f"_roll_{w}") for w in WINDOWS)]
    base_cols += ["rest_days", "dc_decay_weight", "elo_rating"]
    base_cols = [c for c in base_cols if c in feats.columns]

    home_renamed = feats.select(
        pl.col("team_id").alias("home_team_id"),
        pl.col("game_date").alias("start_time"),
        *[pl.col(c).alias(f"{c}_home") for c in base_cols],
    )
    away_renamed = feats.select(
        pl.col("team_id").alias("away_team_id"),
        pl.col("game_date").alias("start_time"),
        *[pl.col(c).alias(f"{c}_away") for c in base_cols],
    )

    merged = matches.join(home_renamed, on=["home_team_id", "start_time"], how="left")
    merged = merged.join(away_renamed, on=["away_team_id", "start_time"], how="left")

    # Diferenciales xG + Elo (los más predictivos)
    for m in ("xg_for_roll_10", "xg_against_roll_10", "goals_for_roll_10", "elo_rating"):
        h = f"{m}_home"
        a = f"{m}_away"
        if h in merged.columns and a in merged.columns:
            merged = merged.with_columns((pl.col(h) - pl.col(a)).alias(f"{m}_diff"))

    return merged
