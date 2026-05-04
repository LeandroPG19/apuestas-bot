"""Features avanzadas multi-deporte — Sprint 11 completamiento.

Consolida features faltantes del research:
- xA (Expected Assists) soccer
- Plus/Minus on/off court NBA
- Umpire strike zone consistency MLB
- Line injury impact NFL
- PDO regression NHL
- Public betting % contrarian signal (global)

Proxies basados en estadísticas agregadas (no event-level data requerido).
Fuente: FBref, Baseball Savant, nflfastR, Natural Stat Trick.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─────────── Soccer xA ───────────
@dataclass(slots=True)
class AssistStats:
    """Expected Assists aproximado por equipo/rolling."""

    key_passes: int
    shots_assisted: int
    xg_assisted: float  # sumatoria xG de shots a partir de pases propios


def approximate_xa(stats: AssistStats) -> float:
    """xA aproximado = xg_assisted (proxy directo).

    Silvestre 2022 (StatsBomb): xA ≈ Σ P(gol) de los shots que el pase creó.
    Si no tenemos event-level, usamos xg_assisted como stand-in.
    """
    return max(0.0, stats.xg_assisted)


# ─────────── NBA Plus/Minus on-off ───────────
@dataclass(slots=True)
class OnOffSplit:
    """Net rating del equipo cuando X jugador está on/off court."""

    player_id: int
    minutes_on: float
    team_net_rating_on: float
    team_net_rating_off: float

    @property
    def impact(self) -> float:
        """Impact = net rating on − off. Positivo = jugador eleva al equipo."""
        return self.team_net_rating_on - self.team_net_rating_off


def star_out_adjustment(team_avg_on_off: float, team_avg_without_star: float) -> float:
    """Ajuste cuando una estrella está out.

    team_avg_on_off: net rating promedio con estrellas on court
    team_avg_without_star: net rating sin la estrella (histórico)

    Retorna delta a aplicar a p_model (probabilidad home win).
    """
    delta_net_rating = team_avg_without_star - team_avg_on_off
    # Aproximación: cada -1 pt net rating ~ -1.2% prob win
    return float(delta_net_rating * 0.012)


# ─────────── MLB Umpire consistency ───────────
@dataclass(slots=True)
class UmpireProfile:
    """Stats del umpire en últimos N partidos."""

    umpire_id: int
    called_strike_pct_out_of_zone: float  # 0-1
    called_ball_pct_in_zone: float
    consistency_score: float  # 0-1, 1 = perfectamente consistente


def umpire_k_adjustment(profile: UmpireProfile) -> float:
    """Ajuste a prob de strikeout.

    Pitcher-friendly umpire: expande zona → más Ks. +0.5% por 5% extra.
    """
    baseline = 0.15  # league avg called strike pct out of zone
    delta = profile.called_strike_pct_out_of_zone - baseline
    return float(delta * 0.10)  # magnitud empírica


# ─────────── NFL Line injury impact ───────────
@dataclass(slots=True)
class LineInjuryImpact:
    """Impact de pérdidas en O-Line/D-Line NFL."""

    team_id: int
    starters_out_oline: int
    starters_out_dline: int
    pff_grade_drop_oline: float  # grade baseline vs current
    pff_grade_drop_dline: float


def line_injury_ev_adjustment(impact: LineInjuryImpact) -> float:
    """Ajuste EV en spreads/totals por lesiones en trenches.

    Aproximación: 1 starter O-line out = −0.5 EPA/play passing.
    """
    epa_loss_pass = impact.starters_out_oline * 0.5
    epa_loss_rush = impact.starters_out_dline * 0.3
    # Converting EPA loss a winprob delta ≈ -1.5% por 1 EPA/game perdido
    total_epa = epa_loss_pass + epa_loss_rush
    return float(-total_epa * 0.015)


# ─────────── NHL PDO regression ───────────
def pdo_regression_signal(pdo_last_10: float, *, mean: float = 1.000, std: float = 0.030) -> float:
    """PDO regression indicator (Natural Stat Trick).

    PDO = on-ice SH% + on-ice SV%. Se regresa a 1.000 en el mediano plazo.
    PDO > 1.030 → sobreperformance → bet contra ese equipo.
    PDO < 0.970 → underperformance → bet a favor (regresa al alza).

    Retorna signal ∈ [-1, 1]:
      -1 = equipo muy afortunado, esperamos regression DOWN
      +1 = equipo muy desafortunado, esperamos regression UP
    """
    z = (pdo_last_10 - mean) / std
    # Invertido: PDO alto = señal negativa (regresará)
    return float(-max(-3.0, min(3.0, z)) / 3.0)


# ─────────── Public betting contrarian ───────────
def contrarian_signal(public_pct_side: float, sharp_pct_side: float) -> float:
    """Sharp-public divergence signal.

    public_pct_side: fraction of tickets on the side (0-1)
    sharp_pct_side: fraction of MONEY on the side (0-1)

    Caso clásico: 80% tickets un lado, pero 30% money = sharp money contra.
    Retorna signal ∈ [-1, 1], positivo = confiar en el lado.
    """
    if public_pct_side == 0.5 or sharp_pct_side == 0.5:
        return 0.0
    divergence = sharp_pct_side - public_pct_side
    # Si sharp > public: sharp está comprando ese lado → señal positiva
    # Si sharp < public: sharp está vendiendo → señal negativa
    return float(max(-1.0, min(1.0, divergence * 4.0)))


__all__ = [
    "AssistStats",
    "LineInjuryImpact",
    "OnOffSplit",
    "UmpireProfile",
    "approximate_xa",
    "contrarian_signal",
    "line_injury_ev_adjustment",
    "pdo_regression_signal",
    "star_out_adjustment",
    "umpire_k_adjustment",
]
