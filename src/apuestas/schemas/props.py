"""Catálogo de player props por deporte + tipos msgspec para predicciones.

Cada prop tiene:
- `code` único (ej. 'nba_points').
- Distribución base recomendada para el modelo.
- Mercado típico (O/U con line continuo, o Yes/No binario).
- Métrica del player_game_logs.stats JSONB de donde se extrae el target.
- Correlaciones conocidas (para Kelly correlation-aware §17.2).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import msgspec


class PropCategory(StrEnum):
    COUNT = "count"  # variables enteras no negativas (points, Ks)
    CONTINUOUS = "continuous"  # yardage, asistencias float (raras)
    BINARY = "binary"  # anytime_goalscorer, home_run_yes_no
    COMBO = "combo"  # PRA, P+R, P+A, R+A


class PropDistribution(StrEnum):
    POISSON = "poisson"  # enteros independientes con media moderada
    NEG_BINOMIAL = "neg_binomial"  # enteros con overdispersion (NBA points)
    GAMMA = "gamma"  # continuas positivas (yardage NFL)
    NORMAL_TRUNC = "normal_trunc"  # truncada en 0 (minutes NBA)
    BERNOULLI = "bernoulli"  # sí/no (home_run, goal)
    MONTE_CARLO = "monte_carlo"  # Statcast plate appearances / Dixon-Coles
    WEIBULL = "weibull"  # rounds to KO boxing


class PropDef(msgspec.Struct, frozen=True, gc=False):
    """Definición estática de un prop."""

    code: str
    sport_code: str
    display_name: str
    category: PropCategory
    distribution: PropDistribution
    stat_key: str  # llave en player_game_logs.stats JSONB
    role: Literal["batter", "pitcher", "skater", "qb", "rb", "wr_te", "skater", "any"] = "any"
    typical_lines: tuple[float, ...] = ()
    # Correlaciones conocidas (con otros props del MISMO jugador)
    correlated_props: tuple[str, ...] = ()
    # Métricas secundarias para features
    opp_allowed_key: str | None = None
    needs_minutes_projection: bool = False
    needs_park_factor: bool = False
    notes: str = ""


# ═══════════════════════ NBA props ═══════════════════════════════════════

NBA_PROPS: tuple[PropDef, ...] = (
    PropDef(
        code="nba_points",
        sport_code="nba",
        display_name="Puntos",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="points",
        typical_lines=(9.5, 14.5, 19.5, 24.5, 29.5, 34.5),
        correlated_props=("nba_pra", "nba_p_r", "nba_p_a"),
        opp_allowed_key="opp_pts_allowed_by_position",
        needs_minutes_projection=True,
    ),
    PropDef(
        code="nba_rebounds",
        sport_code="nba",
        display_name="Rebotes",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="rebounds",
        typical_lines=(3.5, 5.5, 7.5, 9.5, 11.5),
        correlated_props=("nba_pra", "nba_p_r", "nba_r_a"),
        opp_allowed_key="opp_reb_allowed_by_position",
        needs_minutes_projection=True,
    ),
    PropDef(
        code="nba_assists",
        sport_code="nba",
        display_name="Asistencias",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="assists",
        typical_lines=(2.5, 4.5, 6.5, 8.5, 10.5),
        correlated_props=("nba_pra", "nba_p_a", "nba_r_a"),
        opp_allowed_key="opp_ast_allowed_by_position",
        needs_minutes_projection=True,
    ),
    PropDef(
        code="nba_threes",
        sport_code="nba",
        display_name="3-puntos anotados",
        category=PropCategory.COUNT,
        distribution=PropDistribution.POISSON,
        stat_key="fg3m",
        typical_lines=(0.5, 1.5, 2.5, 3.5, 4.5),
        needs_minutes_projection=True,
    ),
    PropDef(
        code="nba_steals",
        sport_code="nba",
        display_name="Robos",
        category=PropCategory.COUNT,
        distribution=PropDistribution.POISSON,
        stat_key="steals",
        typical_lines=(0.5, 1.5, 2.5),
        needs_minutes_projection=True,
    ),
    PropDef(
        code="nba_blocks",
        sport_code="nba",
        display_name="Bloqueos",
        category=PropCategory.COUNT,
        distribution=PropDistribution.POISSON,
        stat_key="blocks",
        typical_lines=(0.5, 1.5, 2.5),
        needs_minutes_projection=True,
    ),
    PropDef(
        code="nba_pra",
        sport_code="nba",
        display_name="Pts+Reb+Ast (PRA)",
        category=PropCategory.COMBO,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="points,rebounds,assists",
        typical_lines=(19.5, 29.5, 39.5, 49.5),
        correlated_props=("nba_points", "nba_rebounds", "nba_assists"),
        needs_minutes_projection=True,
        notes="Combo = points + rebounds + assists",
    ),
    PropDef(
        code="nba_double_double",
        sport_code="nba",
        display_name="Double-double (Yes/No)",
        category=PropCategory.BINARY,
        distribution=PropDistribution.BERNOULLI,
        stat_key="double_double",
        typical_lines=(),
        correlated_props=("nba_points", "nba_rebounds", "nba_assists"),
        needs_minutes_projection=True,
    ),
)


# ═══════════════════════ MLB props ═══════════════════════════════════════

MLB_PROPS: tuple[PropDef, ...] = (
    PropDef(
        code="mlb_home_run",
        sport_code="mlb",
        display_name="Home run (Yes/No)",
        category=PropCategory.BINARY,
        distribution=PropDistribution.MONTE_CARLO,
        stat_key="home_runs",
        role="batter",
        needs_park_factor=True,
        notes="Monte Carlo PA sim con xwOBA + park factor HR + viento",
    ),
    PropDef(
        code="mlb_total_bases",
        sport_code="mlb",
        display_name="Total bases",
        category=PropCategory.COUNT,
        distribution=PropDistribution.MONTE_CARLO,
        stat_key="total_bases",
        role="batter",
        typical_lines=(0.5, 1.5, 2.5, 3.5),
        correlated_props=("mlb_home_run", "mlb_hits"),
        needs_park_factor=True,
    ),
    PropDef(
        code="mlb_hits",
        sport_code="mlb",
        display_name="Hits",
        category=PropCategory.COUNT,
        distribution=PropDistribution.MONTE_CARLO,
        stat_key="hits",
        role="batter",
        typical_lines=(0.5, 1.5, 2.5),
        correlated_props=("mlb_total_bases", "mlb_runs"),
    ),
    PropDef(
        code="mlb_rbi",
        sport_code="mlb",
        display_name="RBIs",
        category=PropCategory.COUNT,
        distribution=PropDistribution.MONTE_CARLO,
        stat_key="rbi",
        role="batter",
        typical_lines=(0.5, 1.5, 2.5),
    ),
    PropDef(
        code="mlb_runs",
        sport_code="mlb",
        display_name="Runs scored",
        category=PropCategory.COUNT,
        distribution=PropDistribution.POISSON,
        stat_key="runs",
        role="batter",
        typical_lines=(0.5, 1.5),
    ),
    PropDef(
        code="mlb_pitcher_ks",
        sport_code="mlb",
        display_name="Strikeouts (pitcher)",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="strikeouts",
        role="pitcher",
        typical_lines=(3.5, 4.5, 5.5, 6.5, 7.5, 8.5),
        opp_allowed_key="opp_team_k_rate_vs_hand",
        needs_park_factor=True,
    ),
    PropDef(
        code="mlb_pitcher_outs",
        sport_code="mlb",
        display_name="Outs grabados (pitcher)",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="outs_recorded",
        role="pitcher",
        typical_lines=(14.5, 16.5, 17.5, 18.5, 19.5, 20.5),
    ),
)


# ═══════════════════════ NFL props ═══════════════════════════════════════

NFL_PROPS: tuple[PropDef, ...] = (
    PropDef(
        code="nfl_qb_pass_yds",
        sport_code="nfl",
        display_name="Yardas pase (QB)",
        category=PropCategory.CONTINUOUS,
        distribution=PropDistribution.GAMMA,
        stat_key="passing_yards",
        role="qb",
        typical_lines=(199.5, 224.5, 249.5, 274.5, 299.5),
        opp_allowed_key="opp_pass_yds_allowed",
    ),
    PropDef(
        code="nfl_qb_pass_tds",
        sport_code="nfl",
        display_name="TDs pase (QB)",
        category=PropCategory.COUNT,
        distribution=PropDistribution.POISSON,
        stat_key="passing_tds",
        role="qb",
        typical_lines=(0.5, 1.5, 2.5, 3.5),
    ),
    PropDef(
        code="nfl_qb_completions",
        sport_code="nfl",
        display_name="Completions (QB)",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="completions",
        role="qb",
        typical_lines=(17.5, 19.5, 21.5, 23.5, 25.5),
    ),
    PropDef(
        code="nfl_qb_interceptions",
        sport_code="nfl",
        display_name="Interceptions (QB)",
        category=PropCategory.COUNT,
        distribution=PropDistribution.POISSON,
        stat_key="interceptions",
        role="qb",
        typical_lines=(0.5, 1.5),
    ),
    PropDef(
        code="nfl_rb_rush_yds",
        sport_code="nfl",
        display_name="Yardas carrera (RB)",
        category=PropCategory.CONTINUOUS,
        distribution=PropDistribution.GAMMA,
        stat_key="rushing_yards",
        role="rb",
        typical_lines=(39.5, 49.5, 59.5, 69.5, 79.5, 89.5, 99.5),
        opp_allowed_key="opp_rush_yds_allowed",
    ),
    PropDef(
        code="nfl_rb_carries",
        sport_code="nfl",
        display_name="Acarreos (RB)",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="carries",
        role="rb",
        typical_lines=(10.5, 12.5, 14.5, 16.5, 18.5),
    ),
    PropDef(
        code="nfl_wr_rec_yds",
        sport_code="nfl",
        display_name="Yardas recepción (WR/TE)",
        category=PropCategory.CONTINUOUS,
        distribution=PropDistribution.GAMMA,
        stat_key="receiving_yards",
        role="wr_te",
        typical_lines=(29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 89.5),
        opp_allowed_key="opp_pass_yds_allowed_to_position",
    ),
    PropDef(
        code="nfl_wr_receptions",
        sport_code="nfl",
        display_name="Recepciones (WR/TE)",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="receptions",
        role="wr_te",
        typical_lines=(2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5),
    ),
)


# ═══════════════════════ Fútbol props ═══════════════════════════════════

SOCCER_PROPS: tuple[PropDef, ...] = (
    PropDef(
        code="soccer_anytime_goal",
        sport_code="soccer",
        display_name="Anotar gol (Yes)",
        category=PropCategory.BINARY,
        distribution=PropDistribution.MONTE_CARLO,
        stat_key="goals",
        notes="Dixon-Coles + share de xG del jugador",
    ),
    PropDef(
        code="soccer_shots",
        sport_code="soccer",
        display_name="Tiros totales",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="shots",
        typical_lines=(0.5, 1.5, 2.5, 3.5, 4.5),
    ),
    PropDef(
        code="soccer_shots_on_target",
        sport_code="soccer",
        display_name="Tiros a puerta",
        category=PropCategory.COUNT,
        distribution=PropDistribution.NEG_BINOMIAL,
        stat_key="shots_on_target",
        typical_lines=(0.5, 1.5, 2.5),
    ),
    PropDef(
        code="soccer_cards",
        sport_code="soccer",
        display_name="Tarjetas recibidas (Yes)",
        category=PropCategory.BINARY,
        distribution=PropDistribution.BERNOULLI,
        stat_key="yellow_cards",
        notes="Ajustar por árbitro (yellow_avg) y fouls_per_game del jugador",
    ),
    PropDef(
        code="soccer_assists",
        sport_code="soccer",
        display_name="Asistencia (Yes)",
        category=PropCategory.BINARY,
        distribution=PropDistribution.BERNOULLI,
        stat_key="assists",
    ),
)


# ═══════════════════════ Boxeo/MMA props ═══════════════════════════════

BOXING_PROPS: tuple[PropDef, ...] = (
    PropDef(
        code="boxing_method_ko",
        sport_code="boxing",
        display_name="Ganar por KO/TKO",
        category=PropCategory.BINARY,
        distribution=PropDistribution.BERNOULLI,
        stat_key="win_by_ko",
        notes="Logistic con KO% histórico + chin oponente + reach diff",
    ),
    PropDef(
        code="boxing_go_distance",
        sport_code="boxing",
        display_name="Pelea completa la distancia (Yes)",
        category=PropCategory.BINARY,
        distribution=PropDistribution.WEIBULL,
        stat_key="went_distance",
    ),
    PropDef(
        code="boxing_over_rounds",
        sport_code="boxing",
        display_name="Rondas (Over/Under)",
        category=PropCategory.CONTINUOUS,
        distribution=PropDistribution.WEIBULL,
        stat_key="rounds_completed",
        typical_lines=(5.5, 7.5, 9.5, 11.5),
    ),
)


# ═══════════════════════ Registro global ════════════════════════════════

ALL_PROPS: dict[str, PropDef] = {
    p.code: p for p in (*NBA_PROPS, *MLB_PROPS, *NFL_PROPS, *SOCCER_PROPS, *BOXING_PROPS)
}


def get_prop(code: str) -> PropDef:
    if code not in ALL_PROPS:
        msg = f"Prop code desconocido: {code}. Ver ALL_PROPS."
        raise KeyError(msg)
    return ALL_PROPS[code]


def props_for_sport(sport_code: str) -> tuple[PropDef, ...]:
    return tuple(p for p in ALL_PROPS.values() if p.sport_code == sport_code)


# ═══════════════════════ Predicción output struct ═══════════════════════


class PropPrediction(msgspec.Struct, frozen=True, gc=False):
    """Output canónico de un modelo de props."""

    prop_code: str
    player_id: int
    player_name: str
    event_id: int
    line: float | None
    # Distribución
    mean: float
    std: float
    p_over: float | None  # si line != None
    p_under: float | None
    p_exact: float | None  # solo counting con line entero
    # Conformal
    p_over_lower: float | None
    p_over_upper: float | None
    # Metadata
    distribution: PropDistribution
    n_samples_training: int
    model_name: str
    model_version: str
    features_snapshot: dict[str, float] | None = None
    # Warnings/flags
    warnings: tuple[str, ...] = ()
