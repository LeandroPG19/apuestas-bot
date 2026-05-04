"""Soccer xT (Expected Threat) + VAEP proxy — Sprint 11 Fase E.

**Expected Threat (xT)** — Karun Singh (2018) https://karun.in/blog/expected-threat.html:
Es un grid-based value: cada zona del campo tiene un valor precomputado
basado en P(gol | posesión desde esa zona). Cambios de posesión (carry/pass)
entre zonas suman/restan threat.

**VAEP** — Decroos 2019 ("Actions Speak Louder Than Goals"):
Valor de cada acción = P(gol próximos 10 eventos | post-acción) − P(... | pre-acción).
Requiere event-level data (Opta/StatsBomb).

Dado que no tenemos event-level en todas las ligas, este módulo implementa
una **aproximación xT-lite** basada en agregados de partido:
- Posesión % (fbref)
- Shots on target vs total shots
- Progressive passes / carries (si disponibles)
- Distance covered con posesión

xT team rolling se añade como feature rolling window [5, 10] partidos.

Para VAEP real se necesita integración con `socceraction` lib:
    pip install socceraction
    from socceraction.xthreat import ExpectedThreat
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Grid xT precomputado (Karun Singh paper, 12x8 grid, valores normalizados).
# Valores crecientes hacia la portería rival (último tercio). Campo [0-1] x [0-1].
_XT_GRID = np.array(
    [
        [0.006, 0.008, 0.009, 0.011, 0.013, 0.015, 0.019, 0.022, 0.026, 0.031, 0.037, 0.052],
        [0.007, 0.009, 0.010, 0.012, 0.014, 0.017, 0.021, 0.025, 0.030, 0.037, 0.048, 0.075],
        [0.008, 0.010, 0.011, 0.013, 0.016, 0.019, 0.024, 0.029, 0.036, 0.046, 0.064, 0.131],
        [0.009, 0.011, 0.012, 0.014, 0.017, 0.021, 0.026, 0.032, 0.041, 0.056, 0.088, 0.272],
        [0.009, 0.011, 0.012, 0.014, 0.017, 0.021, 0.026, 0.032, 0.041, 0.056, 0.088, 0.272],
        [0.008, 0.010, 0.011, 0.013, 0.016, 0.019, 0.024, 0.029, 0.036, 0.046, 0.064, 0.131],
        [0.007, 0.009, 0.010, 0.012, 0.014, 0.017, 0.021, 0.025, 0.030, 0.037, 0.048, 0.075],
        [0.006, 0.008, 0.009, 0.011, 0.013, 0.015, 0.019, 0.022, 0.026, 0.031, 0.037, 0.052],
    ],
    dtype=float,
)


@dataclass(slots=True)
class MatchThreatStats:
    possession_pct: float  # 0-1
    shots_total: int
    shots_on_target: int
    progressive_passes: int = 0
    progressive_carries: int = 0
    avg_position_x: float = 0.5  # distancia promedio al área rival [0-1]


def approximate_xt(stats: MatchThreatStats) -> float:
    """Aproximación xT del equipo en el match.

    Heurística: posesión ponderada por avg_position_x mapeado al grid xT
    + contribución de shots en zona 2 (último tercio) + progressive moves.

    Retorna threat total en unidades de xT.
    """
    # Posesión mapeada a zona promedio
    x_zone = int(min(max(stats.avg_position_x * 11, 0), 11))
    y_zone = 3  # centro-campo por default
    base_threat = _XT_GRID[y_zone, x_zone] * stats.possession_pct * 100

    # Shots on target tienen valor alto (ya están en área rival)
    sot_threat = stats.shots_on_target * 0.15
    other_shots = (stats.shots_total - stats.shots_on_target) * 0.05

    # Progressive moves añaden threat marginal
    progressive_bonus = (stats.progressive_passes + stats.progressive_carries) * 0.01

    return float(base_threat + sot_threat + other_shots + progressive_bonus)


def add_xt_rolling(
    team_games: pl.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10),
) -> pl.DataFrame:
    """Añade columnas xT rolling al team_games DataFrame.

    Espera columnas: team_id, start_time, possession_pct, shots_total,
    shots_on_target. Si faltan, pone 0.
    """
    required = ["team_id", "start_time"]
    for c in required:
        if c not in team_games.columns:
            logger.warning("soccer_xt.missing_column", col=c)
            return team_games

    # Compute per-row xT
    defaults = {
        "possession_pct": 0.5,
        "shots_total": 0,
        "shots_on_target": 0,
        "progressive_passes": 0,
        "progressive_carries": 0,
        "avg_position_x": 0.5,
    }
    df = team_games
    for col, dflt in defaults.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(dflt).alias(col))

    # xT per row (Python loop acotado por N partidos)
    xt_vals: list[float] = []
    for row in df.iter_rows(named=True):
        stats = MatchThreatStats(
            possession_pct=float(row["possession_pct"] or 0.5),
            shots_total=int(row["shots_total"] or 0),
            shots_on_target=int(row["shots_on_target"] or 0),
            progressive_passes=int(row["progressive_passes"] or 0),
            progressive_carries=int(row["progressive_carries"] or 0),
            avg_position_x=float(row["avg_position_x"] or 0.5),
        )
        xt_vals.append(approximate_xt(stats))

    df = df.with_columns(pl.Series("xt_raw", xt_vals))

    # Rolling sum/mean por team
    df = df.sort(["team_id", "start_time"])
    for w in windows:
        col = f"xt_mean_roll_{w}"
        df = df.with_columns(
            pl.col("xt_raw")
            .shift(1)
            .rolling_mean(window_size=w, min_samples=1)
            .over("team_id")
            .alias(col)
        )
    return df


__all__ = ["MatchThreatStats", "add_xt_rolling", "approximate_xt"]
