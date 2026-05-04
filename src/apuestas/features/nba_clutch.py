"""NBA clutch time + lineup efficiency — Sprint 11 Fase F.

Features que mueven la aguja (Valencia-Cardot 2021):

1. **Clutch time splits** (<5 min restantes, margen <5 pts):
   - Equipo clutch ORtg/DRtg
   - Jugadores clutch usage rate
   - FT% clutch (presión psicológica)

2. **Lineup efficiency 5-man units** (nba.com lineup data):
   - Net rating 5-man más usada
   - Minutes together (chemistry proxy)
   - Diff entre starter lineup y backup

3. **On/off court plus-minus** (NBA Advanced):
   - Team net rating when X player on
   - Plus-minus delta starters vs bench

Fuente: Recompute desde play_by_play existente en DB + NBA Stats API
(gratis via nba_api lib si disponible). Fallback: proxies desde features
agregadas (referee_bias.py, coaching_clutch.py ya existen).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ClutchStats:
    """Stats en clutch time (última 5 min, margen ≤5)."""

    team_id: int
    points_clutch: int
    possessions_clutch: int
    ft_made_clutch: int
    ft_att_clutch: int
    turnovers_clutch: int
    minutes_clutch: float

    @property
    def clutch_ortg(self) -> float:
        if self.possessions_clutch <= 0:
            return 100.0
        return 100.0 * self.points_clutch / self.possessions_clutch

    @property
    def clutch_ft_pct(self) -> float:
        if self.ft_att_clutch <= 0:
            return 0.75
        return self.ft_made_clutch / self.ft_att_clutch


def compute_clutch_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Agrega stats clutch desde play-by-play.

    Espera columnas: game_id, team_id, period, time_remaining_sec,
    score_margin, points_scored, is_ft_made, is_ft_attempt, is_turnover.

    Filtra período >= 4 Y time_remaining <= 300s Y |margin| <= 5.
    """
    if pbp.height == 0:
        return pl.DataFrame({"team_id": [], "game_id": []})

    required = {"game_id", "team_id", "period", "time_remaining_sec", "score_margin"}
    missing = required - set(pbp.columns)
    if missing:
        # Downgraded a debug: el caller ya reporta a nivel flow cuando el PBP
        # no está disponible. Esto se dispara N veces por análisis (1 por
        # match NBA) y satura los logs de Telegram sin aportar info nueva.
        logger.debug("nba_clutch.pbp_missing_columns", missing=list(missing))
        return pl.DataFrame({"team_id": [], "game_id": []})

    clutch = pbp.filter(
        (pl.col("period") >= 4)
        & (pl.col("time_remaining_sec") <= 300)
        & (pl.col("score_margin").abs() <= 5)
    )
    if clutch.height == 0:
        return pl.DataFrame({"team_id": [], "game_id": []})

    agg_cols: list[pl.Expr] = []
    for src, out in [
        ("points_scored", "points_clutch"),
        ("is_ft_made", "ft_made_clutch"),
        ("is_ft_attempt", "ft_att_clutch"),
        ("is_turnover", "turnovers_clutch"),
    ]:
        if src in clutch.columns:
            agg_cols.append(pl.col(src).sum().alias(out))

    if not agg_cols:
        return pl.DataFrame({"team_id": [], "game_id": []})

    return clutch.group_by(["game_id", "team_id"]).agg(agg_cols)


def add_clutch_rolling(
    team_games: pl.DataFrame,
    clutch_stats: pl.DataFrame,
    *,
    windows: tuple[int, ...] = (10, 20),
) -> pl.DataFrame:
    """Join clutch stats al team_games + rolling por team."""
    if clutch_stats.height == 0 or "team_id" not in clutch_stats.columns:
        logger.info("nba_clutch.no_stats_skip")
        return team_games

    # Merge por (game_id, team_id)
    join_keys = [
        c for c in ("game_id", "team_id") if c in team_games.columns and c in clutch_stats.columns
    ]
    if len(join_keys) < 2:
        return team_games

    merged = team_games.join(clutch_stats, on=join_keys, how="left")

    # Rolling mean por team en clutch cols
    clutch_cols = [
        c for c in ("points_clutch", "ft_made_clutch", "turnovers_clutch") if c in merged.columns
    ]
    merged = merged.sort(
        ["team_id", "start_time"] if "start_time" in merged.columns else ["team_id"]
    )
    for col in clutch_cols:
        for w in windows:
            out = f"{col}_roll_{w}"
            merged = merged.with_columns(
                pl.col(col)
                .shift(1)
                .rolling_mean(window_size=w, min_samples=1)
                .over("team_id")
                .alias(out)
            )
    return merged


@dataclass(slots=True)
class LineupEfficiency:
    """Eficiencia de una unidad de 5 jugadores."""

    lineup_hash: str  # sorted tuple de player_ids como str
    team_id: int
    minutes: float
    points_for: int
    points_against: int
    possessions: int

    @property
    def net_rating(self) -> float:
        if self.possessions <= 0:
            return 0.0
        return 100.0 * (self.points_for - self.points_against) / self.possessions


def compute_top_lineup_net_rating(
    lineups: list[LineupEfficiency], *, min_minutes: float = 30.0
) -> float:
    """Net rating de la unidad más usada (>min_minutes)."""
    eligible = [lu for lu in lineups if lu.minutes >= min_minutes]
    if not eligible:
        return 0.0
    top = max(eligible, key=lambda lu: lu.minutes)
    return top.net_rating


__all__ = [
    "ClutchStats",
    "LineupEfficiency",
    "add_clutch_rolling",
    "compute_clutch_from_pbp",
    "compute_top_lineup_net_rating",
]
