"""Helpers genéricos de feature engineering.

Principio clave: TODAS las features rolling cierran en t-1 para evitar leakage.
Usar `closed='left'` o equivalente en windows temporales.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl


def rolling_mean_prev(
    df: pl.DataFrame,
    *,
    by: str,
    order: str,
    value: str,
    windows: list[int],
    name_prefix: str | None = None,
) -> pl.DataFrame:
    """Rolling mean SIN incluir el row actual (closed='left').

    Para cada grupo `by`, ordenado por `order`, calcula la media de `value`
    sobre los N-1 rows anteriores (shift 1 luego rolling).

    Args:
        df: DataFrame con al menos columnas [by, order, value].
        by: columna de grupo (ej. 'team_id').
        order: columna temporal (ej. 'start_time').
        value: métrica a promediar (ej. 'ortg').
        windows: lista de tamaños (ej. [5, 10, 20]).
        name_prefix: prefijo nombre output. Default `value`.
    """
    prefix = name_prefix or value
    result = df.sort([by, order])
    for w in windows:
        col_name = f"{prefix}_roll_{w}"
        result = result.with_columns(
            pl.col(value)
            .shift(1)
            .rolling_mean(window_size=w, min_samples=1)
            .over(by)
            .alias(col_name)
        )
    return result


def rolling_sum_prev(
    df: pl.DataFrame,
    *,
    by: str,
    order: str,
    value: str,
    windows: list[int],
    name_prefix: str | None = None,
) -> pl.DataFrame:
    """Igual que rolling_mean_prev pero suma."""
    prefix = name_prefix or value
    result = df.sort([by, order])
    for w in windows:
        col_name = f"{prefix}_rollsum_{w}"
        result = result.with_columns(
            pl.col(value)
            .shift(1)
            .rolling_sum(window_size=w, min_samples=1)
            .over(by)
            .alias(col_name)
        )
    return result


def exponential_decay_mean(
    df: pl.DataFrame,
    *,
    by: str,
    order: str,
    value: str,
    half_life_days: float = 30.0,
    name: str | None = None,
) -> pl.DataFrame:
    """Media ponderada exponencialmente por recencia.

    Pondera cada muestra pasada por `exp(-ln(2) * days_ago / half_life)`.
    Implementado vía agg por grupo; para datasets <10M filas es suficiente.
    """
    col_name = name or f"{value}_ewm_hl{int(half_life_days)}"
    ln2 = float(np.log(2))

    # Calcular dif. en días vs row actual dentro del grupo
    result = (
        df.sort([by, order])
        .with_columns(
            pl.col(order).cast(pl.Float64).alias("_order_f64"),
        )
        .with_columns(
            ((pl.col("_order_f64") - pl.col("_order_f64").shift(1).over(by)) / 86_400.0)
            .fill_null(0.0)
            .alias("_days_since_prev")
        )
    )

    # EWM recursivo con decay por días variables
    # Polars no tiene EWM time-weighted nativo; implementar vía running group agg
    groups = result.partition_by(by, as_dict=True)
    out_rows: list[pl.DataFrame] = []
    for key, g in groups.items():
        g_sorted = g.sort(order)
        vals = g_sorted[value].to_numpy()
        days = g_sorted["_days_since_prev"].to_numpy()
        if len(vals) == 0:
            g_sorted = g_sorted.with_columns(pl.lit(None, dtype=pl.Float64).alias(col_name))
            out_rows.append(g_sorted)
            continue
        ewm = np.empty(len(vals), dtype=np.float64)
        ewm[0] = np.nan  # No hay histórico previo
        acc = vals[0]
        for i in range(1, len(vals)):
            # Decay desde i-1 hasta i
            factor = np.exp(-ln2 * days[i] / half_life_days)
            # EWM es la media ponderada incluyendo histórico hasta i-1
            acc = factor * acc + (1 - factor) * vals[i - 1]
            ewm[i] = acc
        g_sorted = g_sorted.with_columns(pl.Series(name=col_name, values=ewm))
        out_rows.append(g_sorted)

    combined = pl.concat(out_rows) if out_rows else result
    return combined.drop(["_order_f64", "_days_since_prev"])


def days_since_last(
    df: pl.DataFrame,
    *,
    by: str,
    order: str,
    name: str = "rest_days",
) -> pl.DataFrame:
    """Días desde el último evento del mismo grupo."""
    return df.sort([by, order]).with_columns(
        (
            (pl.col(order).cast(pl.Float64) - pl.col(order).cast(pl.Float64).shift(1).over(by))
            / 86_400.0
        ).alias(name)
    )


def back_to_back_flag(
    df: pl.DataFrame,
    *,
    by: str,
    order: str,
    threshold_hours: float = 30.0,
    name: str = "back_to_back",
) -> pl.DataFrame:
    """B2B si el evento anterior fue hace < threshold horas."""
    return df.sort([by, order]).with_columns(
        (
            (pl.col(order).cast(pl.Float64) - pl.col(order).cast(pl.Float64).shift(1).over(by))
            / 3600.0
            < threshold_hours
        )
        .fill_null(False)
        .alias(name)
    )


def games_in_last_n_days(
    df: pl.DataFrame,
    *,
    by: str,
    order: str,
    n_days: int,
    name_prefix: str = "games",
) -> pl.DataFrame:
    """Cuenta eventos del grupo en los últimos N días (excluye el actual)."""
    col_name = f"{name_prefix}_last_{n_days}d"
    seconds = n_days * 86_400
    result_rows: list[pl.DataFrame] = []
    for _key, g in df.partition_by(by, as_dict=True).items():
        g = g.sort(order)
        ts = g[order].cast(pl.Int64).to_numpy()
        counts = np.zeros(len(ts), dtype=np.int32)
        j = 0
        for i in range(len(ts)):
            while j < i and ts[i] - ts[j] > seconds:
                j += 1
            counts[i] = i - j
        g = g.with_columns(pl.Series(name=col_name, values=counts))
        result_rows.append(g)
    return pl.concat(result_rows) if result_rows else df


def add_home_away_split(
    df: pl.DataFrame,
    *,
    team_col: str = "team_id",
    is_home_col: str = "is_home",
    value_col: str,
    order: str = "start_time",
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """Rolling separado cuando `is_home=true` y cuando `is_home=false`.

    Genera columnas `{value}_home_roll_{N}` y `{value}_away_roll_{N}`.
    """
    windows = windows or [5, 10]
    result = df.sort([team_col, order])
    for w in windows:
        # Home mask
        home_col = f"{value_col}_home_roll_{w}"
        away_col = f"{value_col}_away_roll_{w}"
        result = result.with_columns(
            pl.when(pl.col(is_home_col))
            .then(pl.col(value_col))
            .otherwise(None)
            .shift(1)
            .rolling_mean(window_size=w, min_samples=1)
            .over(team_col)
            .alias(home_col),
            pl.when(~pl.col(is_home_col))
            .then(pl.col(value_col))
            .otherwise(None)
            .shift(1)
            .rolling_mean(window_size=w, min_samples=1)
            .over(team_col)
            .alias(away_col),
        )
    return result


def diff_features(
    df: pl.DataFrame,
    *,
    home_cols: list[str],
    away_cols: list[str],
    suffix: str = "_diff",
) -> pl.DataFrame:
    """Genera columnas diferencial (home - away) para cada par alineado."""
    if len(home_cols) != len(away_cols):
        msg = "home_cols y away_cols deben tener mismo largo"
        raise ValueError(msg)
    for h, a in zip(home_cols, away_cols, strict=True):
        out_name = h.replace("_home", "") + suffix
        df = df.with_columns((pl.col(h) - pl.col(a)).alias(out_name))
    return df


def standardize(
    df: pl.DataFrame,
    *,
    columns: list[str],
    stats: dict[str, tuple[float, float]] | None = None,
    return_stats: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, tuple[float, float]]]:
    """Z-score por columna. Si stats dado, usa ese mean/std (evita leakage train→test)."""
    if stats is None:
        stats = {}
        for c in columns:
            mean = float(df[c].mean() or 0.0)
            std = float(df[c].std() or 1.0) or 1.0
            stats[c] = (mean, std)

    for c in columns:
        mean, std = stats[c]
        df = df.with_columns(((pl.col(c) - mean) / std).alias(c))

    if return_stats:
        return df, stats
    return df


def encode_categorical_target(
    df: pl.DataFrame,
    *,
    col: str,
    target: str,
    smoothing: float = 10.0,
    time_col: str | None = None,
) -> pl.DataFrame:
    """Target encoding con smoothing bayesiano, leave-one-out.

    Para cada fila, promedia `target` sobre todas las filas anteriores
    (si `time_col` dado) con mismo valor de `col`. Evita leakage.
    """
    out_col = f"{col}_target_enc"
    if time_col:
        df = df.sort(time_col)

    prior_mean = float(df[target].mean() or 0.0)
    # Agrupar: para cada categoría, calcular media acumulada hasta t-1
    # Polars cumsum/count per group
    df = df.with_columns(
        pl.col(target).shift(1).fill_null(prior_mean).alias("_t_prev"),
    )
    df = df.with_columns(
        pl.col("_t_prev").cum_sum().over(col).alias("_csum"),
        pl.col("_t_prev").cum_count().over(col).alias("_ccnt"),
    )
    df = df.with_columns(
        ((pl.col("_csum") + prior_mean * smoothing) / (pl.col("_ccnt") + smoothing)).alias(out_col)
    )
    return df.drop(["_t_prev", "_csum", "_ccnt"])


TargetKind = Literal["win", "ats", "total", "btts"]


def compute_target(
    df: pl.DataFrame,
    *,
    kind: TargetKind,
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    spread_col: str | None = None,
    total_col: str | None = None,
) -> pl.DataFrame:
    """Genera columna `y` (0/1) según el tipo de mercado."""
    if kind == "win":
        return df.with_columns(
            (pl.col(home_score_col) > pl.col(away_score_col)).cast(pl.Int8).alias("y")
        )
    if kind == "ats":
        if spread_col is None:
            msg = "ATS requiere spread_col"
            raise ValueError(msg)
        return df.with_columns(
            ((pl.col(home_score_col) + pl.col(spread_col)) > pl.col(away_score_col))
            .cast(pl.Int8)
            .alias("y")
        )
    if kind == "total":
        if total_col is None:
            msg = "TOTAL requiere total_col"
            raise ValueError(msg)
        return df.with_columns(
            ((pl.col(home_score_col) + pl.col(away_score_col)) > pl.col(total_col))
            .cast(pl.Int8)
            .alias("y")
        )
    if kind == "btts":
        return df.with_columns(
            ((pl.col(home_score_col) > 0) & (pl.col(away_score_col) > 0)).cast(pl.Int8).alias("y")
        )
    msg = f"Target kind desconocido: {kind}"
    raise ValueError(msg)


def add_elo_features(
    matches: pl.DataFrame,
    *,
    sport: str,
    home_team_col: str = "home_team_id",
    away_team_col: str = "away_team_id",
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    time_col: str = "start_time",
) -> pl.DataFrame:
    import os as _os

    # Sprint 10 — ablation support: si APUESTAS_ELO_FEATURES_DISABLED=true,
    # devolver DataFrame sin tocar (baseline para scripts/retrain_elo_ablation.py).
    if _os.environ.get("APUESTAS_ELO_FEATURES_DISABLED", "false").lower() == "true":
        return matches
    """Añade Elo features al DataFrame de matches — Sprint 10 Fase 2.

    Recorre matches en orden cronológico actualizando un EloBuilder. Cada
    row recibe feature `elo_home`, `elo_away`, `elo_diff`, `elo_p_home`
    CALCULADOS ANTES del match (anti-leakage: usa ratings pre-partido).

    Args:
        matches: polars DataFrame con al menos [id, home_team_col,
                 away_team_col, home_score_col, away_score_col, time_col].
        sport: 'nba', 'mlb', 'nfl', 'nhl', 'soccer', etc.

    Returns:
        Mismo DataFrame con columnas nuevas: elo_home, elo_away, elo_diff,
        elo_p_home. Si el match no tiene scores (futuro), aplica features
        con ratings actuales sin actualizar.
    """
    from apuestas.features.elo_builder import EloBuilder

    builder = EloBuilder(sport=sport)
    df_sorted = matches.sort(time_col)

    elo_home_col: list[float] = []
    elo_away_col: list[float] = []
    elo_diff_col: list[float] = []
    elo_p_home_col: list[float] = []

    for row in df_sorted.iter_rows(named=True):
        home_id = row.get(home_team_col)
        away_id = row.get(away_team_col)
        # Features ANTES del update (anti-leakage)
        feats = builder.features_for_upcoming(str(home_id), str(away_id))
        elo_home_col.append(feats["elo_home"])
        elo_away_col.append(feats["elo_away"])
        elo_diff_col.append(feats["elo_diff"])
        elo_p_home_col.append(feats["elo_p_home"])
        # Update SOLO si hay scores reales (entrenamiento histórico)
        hs = row.get(home_score_col)
        as_ = row.get(away_score_col)
        if hs is not None and as_ is not None:
            builder.update_match(
                home=str(home_id),
                away=str(away_id),
                home_score=int(hs),
                away_score=int(as_),
            )

    return df_sorted.with_columns(
        [
            pl.Series("elo_home", elo_home_col),
            pl.Series("elo_away", elo_away_col),
            pl.Series("elo_diff", elo_diff_col),
            pl.Series("elo_p_home", elo_p_home_col),
        ]
    )
