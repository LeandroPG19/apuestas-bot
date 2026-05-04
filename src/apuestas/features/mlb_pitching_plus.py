"""MLB Stuff+ / Pitching+ / Location+ — Sprint 11 Fase G.

Métricas SOTA FanGraphs (Driveline-Eno Sarris 2022) para predecir calidad
pitcher granular:

- **Stuff+**: calidad física del pitch (velo, spin, movimiento). 100 = avg.
- **Location+**: dónde ubica el pitch (zona, borde, etc.).
- **Pitching+**: combinación Stuff+ × Location+ ponderada.

Sin acceso directo a FanGraphs API (pagada), implementamos proxies basados
en Baseball Savant Statcast (gratuito con delay):

- Spin rate → proxy Stuff+ (correlación ~0.75 en breakers, ~0.60 fastballs)
- Release point consistency → proxy Command+
- Whiff% + CSW% → resultado combinado Pitching+
- Chase rate (O-Swing%) → deception

Fuente data: pybaseball.statcast_pitcher(player_id) gratuita con rate limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PitcherStuffMetrics:
    """Métricas granulares por pitcher (rolling window)."""

    pitcher_id: int
    spin_rate_avg: float  # rpm
    velo_avg: float  # mph
    whiff_pct: float  # % pitches con swing-and-miss
    csw_pct: float  # called strike + whiff
    chase_pct: float  # O-Swing%
    release_consistency: float  # desv estándar del release point (menor = mejor)
    n_pitches: int


def estimate_stuff_plus(metrics: PitcherStuffMetrics) -> float:
    """Aproximación Stuff+ (100 = liga average).

    Ponderación empírica basada en Sarris 2022:
    - spin_rate: 35% del peso
    - velo: 30%
    - whiff_pct: 35%

    League averages MLB 2024:
    - spin_rate fastball: ~2300 rpm
    - velo fastball: ~93.8 mph
    - whiff%: ~11.5%
    """
    if metrics.n_pitches < 50:
        return 100.0

    # Z-scores vs league avg (aproximados)
    spin_z = (metrics.spin_rate_avg - 2300.0) / 200.0
    velo_z = (metrics.velo_avg - 93.8) / 2.5
    whiff_z = (metrics.whiff_pct - 0.115) / 0.035

    # Combinación ponderada, escalada a media 100 σ 10
    score = 0.35 * spin_z + 0.30 * velo_z + 0.35 * whiff_z
    return float(100.0 + score * 10.0)


def estimate_location_plus(metrics: PitcherStuffMetrics) -> float:
    """Aproximación Location+ via release consistency + chase rate."""
    if metrics.n_pitches < 50:
        return 100.0
    # Release consistency: menor = mejor (invertido)
    release_z = (0.08 - metrics.release_consistency) / 0.04
    chase_z = (metrics.chase_pct - 0.30) / 0.06
    score = 0.60 * release_z + 0.40 * chase_z
    return float(100.0 + score * 10.0)


def estimate_pitching_plus(metrics: PitcherStuffMetrics) -> float:
    """Pitching+ combinado. Producto ponderado Stuff+ × Location+."""
    stuff = estimate_stuff_plus(metrics)
    location = estimate_location_plus(metrics)
    # Geometric mean con peso 60/40 hacia Stuff+
    return float(stuff**0.6 * location**0.4)


def add_pitching_plus_features(
    pitcher_games: pl.DataFrame,
    *,
    windows: tuple[int, ...] = (5, 10),
) -> pl.DataFrame:
    """Añade columnas Stuff+, Location+, Pitching+ rolling al DataFrame.

    Espera pitcher_games con: pitcher_id, game_id, spin_rate_avg, velo_avg,
    whiff_pct, csw_pct, chase_pct, release_consistency, n_pitches.
    """
    required = ("pitcher_id", "spin_rate_avg", "velo_avg", "whiff_pct")
    for col in required:
        if col not in pitcher_games.columns:
            logger.warning("mlb_pitching_plus.missing_col", col=col)
            return pitcher_games

    df = pitcher_games
    for col, dflt in [
        ("csw_pct", 0.28),
        ("chase_pct", 0.30),
        ("release_consistency", 0.08),
        ("n_pitches", 50),
    ]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(dflt).alias(col))

    # Compute per row
    stuff_vals: list[float] = []
    location_vals: list[float] = []
    pitching_vals: list[float] = []
    for row in df.iter_rows(named=True):
        m = PitcherStuffMetrics(
            pitcher_id=int(row["pitcher_id"] or 0),
            spin_rate_avg=float(row["spin_rate_avg"] or 2300.0),
            velo_avg=float(row["velo_avg"] or 93.8),
            whiff_pct=float(row["whiff_pct"] or 0.115),
            csw_pct=float(row["csw_pct"] or 0.28),
            chase_pct=float(row["chase_pct"] or 0.30),
            release_consistency=float(row["release_consistency"] or 0.08),
            n_pitches=int(row["n_pitches"] or 50),
        )
        stuff_vals.append(estimate_stuff_plus(m))
        location_vals.append(estimate_location_plus(m))
        pitching_vals.append(estimate_pitching_plus(m))

    df = df.with_columns(
        [
            pl.Series("stuff_plus", stuff_vals),
            pl.Series("location_plus", location_vals),
            pl.Series("pitching_plus", pitching_vals),
        ]
    )

    df = df.sort(["pitcher_id", "start_time"] if "start_time" in df.columns else ["pitcher_id"])
    for col in ("stuff_plus", "location_plus", "pitching_plus"):
        for w in windows:
            out = f"{col}_roll_{w}"
            df = df.with_columns(
                pl.col(col)
                .shift(1)
                .rolling_mean(window_size=w, min_samples=1)
                .over("pitcher_id")
                .alias(out)
            )

    # Sanity: clip extremos
    for col in ("stuff_plus", "location_plus", "pitching_plus"):
        df = df.with_columns(pl.col(col).clip(50, 150))

    return df


__all__ = [
    "PitcherStuffMetrics",
    "add_pitching_plus_features",
    "estimate_location_plus",
    "estimate_pitching_plus",
    "estimate_stuff_plus",
]
