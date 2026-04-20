"""Features NBA: Four Factors (Dean Oliver 2004), pace, ORtg/DRtg, rest, altitud.

Entrada esperada: tabla `matches` + boxscores + `team_stats_rolling_*`.
Salida: DataFrame con features tripleta (home/away/diff) listas para modelo.

Principio crítico: todo rolling cierra en t-1 (anti-leakage temporal).
"""

from __future__ import annotations

import polars as pl

from apuestas.features.common import (
    add_home_away_split,
    back_to_back_flag,
    days_since_last,
    diff_features,
    games_in_last_n_days,
    rolling_mean_prev,
)

# Features canónicas del modelo NBA v1. Cualquier cambio debe aumentar version.
FEATURE_SET_NAME = "nba_v1"

# Windows estándar (juegos)
WINDOWS = [5, 10, 20]


def four_factors_from_box(boxscore: pl.DataFrame) -> pl.DataFrame:
    """Calcula Four Factors por equipo-juego a partir de un boxscore estándar.

    Columnas esperadas en boxscore:
        game_id, team_id, is_home, fgm, fga, fg3m, ftm, fta, oreb, dreb,
        tov, pts, possessions
    """
    required = {"fgm", "fga", "fg3m", "ftm", "fta", "oreb", "dreb", "tov", "pts"}
    missing = required - set(boxscore.columns)
    if missing:
        msg = f"Boxscore sin columnas requeridas: {missing}"
        raise ValueError(msg)

    df = boxscore.with_columns(
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        ((pl.col("fgm") + 0.5 * pl.col("fg3m")) / pl.col("fga")).alias("efg_pct"),
        # TOV%: turnovers per possession
        (pl.col("tov") / (pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov"))).alias("tov_pct"),
        # FT rate: FTA / FGA (proxy de free throws earned)
        (pl.col("fta") / pl.col("fga")).alias("ft_rate"),
    )

    # ORB%: OREB / (OREB + DREB_opponent). Aproximación simple: OREB / (OREB + DREB)
    # propios. El cálculo estricto requiere join al equipo contrario.
    df = df.with_columns(
        (pl.col("oreb") / (pl.col("oreb") + pl.col("dreb"))).alias("orb_pct"),
    )

    # Possessions Dean Oliver estimate
    df = df.with_columns(
        (pl.col("fga") - pl.col("oreb") + pl.col("tov") + 0.44 * pl.col("fta")).alias("poss_est")
    )

    # Pace: posesiones por 48 min (asumiendo juego completo; ajustar si hay OT)
    # Aquí aproximamos dividiendo por 48 — en NBA real se ajusta por minutos.
    df = df.with_columns((pl.col("poss_est") / 48.0 * 48.0).alias("pace"))

    # ORtg/DRtg per 100 possessions
    df = df.with_columns(
        (pl.col("pts") / pl.col("poss_est") * 100).alias("ortg"),
    )

    return df


def team_rolling_features(team_games: pl.DataFrame) -> pl.DataFrame:
    """Rolling (5/10/20) sobre Four Factors + pace + ORtg por equipo.

    Entrada: DataFrame con columnas
        team_id, start_time, is_home, efg_pct, tov_pct, orb_pct, ft_rate,
        pace, ortg, drtg, win_margin, total_points.

    Salida: DataFrame original + columnas `{metric}_roll_{w}` (histórico)
    y `{metric}_{home|away}_roll_{w}` (split por condición).
    """
    # Rolling generales (últimos N juegos, cualquier condición)
    metrics = [
        "efg_pct",
        "tov_pct",
        "orb_pct",
        "ft_rate",
        "pace",
        "ortg",
        "drtg",
        "win_margin",
        "total_points",
    ]
    result = team_games.sort(["team_id", "start_time"])
    for metric in metrics:
        if metric not in result.columns:
            continue
        result = rolling_mean_prev(
            result,
            by="team_id",
            order="start_time",
            value=metric,
            windows=WINDOWS,
        )

    # Splits home/away en métricas clave
    for metric in ["ortg", "drtg", "pace", "win_margin"]:
        if metric not in result.columns:
            continue
        result = add_home_away_split(
            result,
            team_col="team_id",
            is_home_col="is_home",
            value_col=metric,
            order="start_time",
            windows=[5, 10],
        )

    # Rest days + B2B + condensed schedule
    result = days_since_last(result, by="team_id", order="start_time", name="rest_days")
    result = back_to_back_flag(
        result, by="team_id", order="start_time", threshold_hours=30.0, name="back_to_back"
    )
    result = games_in_last_n_days(result, by="team_id", order="start_time", n_days=7)
    result = games_in_last_n_days(result, by="team_id", order="start_time", n_days=14)

    return result


def join_home_away(
    matches: pl.DataFrame,
    team_features: pl.DataFrame,
    *,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Une matches con team_features separando home y away con sufijos.

    `matches` tiene home_team_id, away_team_id, start_time.
    `team_features` tiene team_id, start_time y las métricas rolling.
    """
    home_renamed = team_features.select(
        pl.col("team_id").alias("home_team_id"),
        pl.col("start_time"),
        *[pl.col(c).alias(f"{c}_home") for c in feature_cols],
    )
    away_renamed = team_features.select(
        pl.col("team_id").alias("away_team_id"),
        pl.col("start_time"),
        *[pl.col(c).alias(f"{c}_away") for c in feature_cols],
    )

    df = matches.join(home_renamed, on=["home_team_id", "start_time"], how="left")
    df = df.join(away_renamed, on=["away_team_id", "start_time"], how="left")
    return df


def build_nba_feature_frame(
    matches: pl.DataFrame,
    team_games: pl.DataFrame,
    *,
    venue_altitudes: dict[int, int] | None = None,
) -> pl.DataFrame:
    """Pipeline completo NBA: team features → join mirror → diferencias.

    `matches` requiere: id, home_team_id, away_team_id, start_time, venue_id.
    `team_games` requiere: team_id, start_time, is_home, + Four Factors + ortg/drtg.
    """
    team_feats = team_rolling_features(team_games)

    # Columnas a unir (todas las derivadas)
    base_metrics = [
        "efg_pct",
        "tov_pct",
        "orb_pct",
        "ft_rate",
        "pace",
        "ortg",
        "drtg",
        "win_margin",
        "total_points",
    ]
    feature_cols: list[str] = []
    for m in base_metrics:
        for w in WINDOWS:
            col = f"{m}_roll_{w}"
            if col in team_feats.columns:
                feature_cols.append(col)
    for m in ["ortg", "drtg", "pace", "win_margin"]:
        for w in [5, 10]:
            for side in ("home", "away"):
                col = f"{m}_{side}_roll_{w}"
                if col in team_feats.columns:
                    feature_cols.append(col)

    # Features del equipo (no rolling) que queremos arrastrar
    for c in ("rest_days", "back_to_back", "games_last_7d", "games_last_14d"):
        if c in team_feats.columns:
            feature_cols.append(c)

    merged = join_home_away(matches, team_feats, feature_cols=feature_cols)

    # Diferenciales
    diff_candidates = [
        ("ortg_roll_5_home", "ortg_roll_5_away"),
        ("ortg_roll_10_home", "ortg_roll_10_away"),
        ("drtg_roll_5_home", "drtg_roll_5_away"),
        ("drtg_roll_10_home", "drtg_roll_10_away"),
        ("pace_roll_5_home", "pace_roll_5_away"),
        ("win_margin_roll_10_home", "win_margin_roll_10_away"),
        ("rest_days_home", "rest_days_away"),
    ]
    pairs_present = [
        (h, a) for h, a in diff_candidates if h in merged.columns and a in merged.columns
    ]
    if pairs_present:
        merged = diff_features(
            merged,
            home_cols=[h for h, _ in pairs_present],
            away_cols=[a for _, a in pairs_present],
        )

    # Venue: altitud (feature específica NBA para Denver, Utah)
    if venue_altitudes and "venue_id" in merged.columns:
        merged = merged.with_columns(
            pl.col("venue_id")
            .cast(pl.Int64)
            .replace_strict(venue_altitudes, default=0)
            .alias("venue_altitude_m")
        )

    return merged


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Lista columnas que son features válidas para ML (numéricas, no target/ids)."""
    excluded = {
        "id",
        "external_id",
        "home_team_id",
        "away_team_id",
        "venue_id",
        "league_id",
        "start_time",
        "season",
        "stage",
        "status",
        "home_score",
        "away_score",
        "y",
        "sport_code",
        "metadata",
    }
    numeric_types = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Boolean)
    return [c for c in df.columns if c not in excluded and df.schema[c] in numeric_types]
