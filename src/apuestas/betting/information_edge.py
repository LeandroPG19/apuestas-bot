"""Information edge — Sprint 11 Fase J.

Señales NO predictivas por modelo, sino por timing de información:

1. **Weather impact quantification** (MLB, NFL, Golf): wind + precip +
   temperature → adjustment multiplier en odds implícitas.
2. **Lineup confirmation status**: "probable" vs "confirmed" permite
   apostar ANTES del ajuste público; crítico en NBA (star out) y soccer.
3. **Injury news scoring**: delta en key-player EV desde último report.
4. **Sharp-public divergence**: línea se mueve contra mayoría pública =
   sharp money ingresando (Levitt 2004).

Uso:
    adj = await weather_impact_adjustment(match_id)
    # adj = {"home": +0.02, "away": -0.015}  (en unidades de prob)

    status = await lineup_confidence_level(match_id)
    # "announced" | "probable" | "uncertain"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


LineupStatus = Literal["announced", "probable", "uncertain", "unknown"]


@dataclass(slots=True)
class WeatherAdjustment:
    home_prob_delta: float  # impacto sobre P(home win)
    total_runs_delta: float  # impacto sobre total scored (MLB/NFL)
    confidence: float  # 0-1
    reason: str


def compute_weather_adjustment_mlb(
    *,
    wind_speed_mph: float,
    wind_direction: str,  # "in", "out", "cross", "calm"
    temperature_f: float,
    humidity_pct: float,
    precip_prob: float,
) -> WeatherAdjustment:
    """Ajuste MLB: wind afecta HR rate dramáticamente.

    Referencias:
    - Nathan 2008 (SABR Baseball Research Journal): wind 10mph out
      → +8% HR rate → totals over boost.
    - Cool temp (<60°F) → ball travels menos → under boost.
    - Alta humedad → ball travel reducido moderado.
    """
    total_delta = 0.0
    home_delta = 0.0
    reason_parts: list[str] = []

    if wind_direction == "out" and wind_speed_mph > 5:
        delta = 0.008 * (wind_speed_mph / 10.0)  # +8% por cada 10mph out
        total_delta += delta
        reason_parts.append(f"wind_out_{wind_speed_mph:.0f}mph")
    elif wind_direction == "in" and wind_speed_mph > 5:
        delta = -0.006 * (wind_speed_mph / 10.0)
        total_delta += delta
        reason_parts.append(f"wind_in_{wind_speed_mph:.0f}mph")

    if temperature_f < 60:
        total_delta -= 0.015
        reason_parts.append(f"cold_{temperature_f:.0f}f")
    elif temperature_f > 85:
        total_delta += 0.010
        reason_parts.append(f"hot_{temperature_f:.0f}f")

    if humidity_pct > 80:
        total_delta -= 0.005
    if precip_prob > 0.4:
        total_delta -= 0.020
        reason_parts.append(f"precip_{precip_prob:.0%}")

    return WeatherAdjustment(
        home_prob_delta=home_delta,
        total_runs_delta=total_delta,
        confidence=min(1.0, len(reason_parts) * 0.25),
        reason=",".join(reason_parts) or "neutral",
    )


def compute_weather_adjustment_nfl(
    *,
    wind_speed_mph: float,
    precip_prob: float,
    temperature_f: float,
) -> WeatherAdjustment:
    """NFL: wind >20mph baja passing EPA, precip penaliza totals."""
    total_delta = 0.0
    home_delta = 0.0
    reason_parts: list[str] = []

    if wind_speed_mph > 20:
        total_delta -= 0.04
        home_delta -= 0.010  # home QB ajustado mejor al entorno
        reason_parts.append(f"high_wind_{wind_speed_mph:.0f}")
    elif wind_speed_mph > 15:
        total_delta -= 0.02

    if precip_prob > 0.5:
        total_delta -= 0.03
        reason_parts.append(f"precip_{precip_prob:.0%}")

    if temperature_f < 20:
        total_delta -= 0.02
        reason_parts.append(f"freezing_{temperature_f:.0f}f")

    return WeatherAdjustment(
        home_prob_delta=home_delta,
        total_runs_delta=total_delta,
        confidence=min(1.0, len(reason_parts) * 0.30),
        reason=",".join(reason_parts) or "neutral",
    )


@dataclass(slots=True)
class LineupInfo:
    status: LineupStatus
    hours_until_kickoff: float
    key_players_missing: int
    star_out_flag: bool
    confidence: float


def score_lineup_confidence(
    *,
    kickoff_utc: datetime,
    now_utc: datetime | None = None,
    has_starter_list: bool = False,
    probable_players_pct: float = 1.0,
) -> LineupInfo:
    """Heurística: mientras más cerca del kickoff + más jugadores confirmados,
    mayor confianza.
    """

    now_utc = now_utc or datetime.now(UTC)
    hrs = (kickoff_utc - now_utc).total_seconds() / 3600.0

    if has_starter_list and probable_players_pct > 0.9:
        status: LineupStatus = "announced"
        conf = 0.95
    elif hrs < 2 and probable_players_pct > 0.7:
        status = "probable"
        conf = 0.65
    elif hrs < 24:
        status = "uncertain"
        conf = 0.30
    else:
        status = "unknown"
        conf = 0.10

    return LineupInfo(
        status=status,
        hours_until_kickoff=hrs,
        key_players_missing=0,  # caller llena si tiene data
        star_out_flag=False,
        confidence=conf,
    )


@dataclass(slots=True)
class SharpDivergence:
    """Detecta si sharp money está contra el público (señal fuerte)."""

    public_pct_home: float  # 0-1
    line_movement_direction: Literal["to_home", "to_away", "flat"]
    is_divergence: bool
    signal_strength: float  # 0-1

    @classmethod
    def compute(
        cls,
        *,
        public_pct_home: float,
        line_movement_home: float,  # negativo = línea se hace más favorable a home
    ) -> SharpDivergence:
        if abs(line_movement_home) < 0.01:
            return cls(
                public_pct_home=public_pct_home,
                line_movement_direction="flat",
                is_divergence=False,
                signal_strength=0.0,
            )
        direction = "to_home" if line_movement_home < 0 else "to_away"
        # Divergencia: público masivo en un lado pero línea se mueve al otro
        if public_pct_home > 0.70 and direction == "to_away":
            return cls(
                public_pct_home=public_pct_home,
                line_movement_direction=direction,
                is_divergence=True,
                signal_strength=(public_pct_home - 0.5)
                * 2
                * min(abs(line_movement_home), 0.1)
                / 0.1,
            )
        if public_pct_home < 0.30 and direction == "to_home":
            return cls(
                public_pct_home=public_pct_home,
                line_movement_direction=direction,
                is_divergence=True,
                signal_strength=(0.5 - public_pct_home)
                * 2
                * min(abs(line_movement_home), 0.1)
                / 0.1,
            )
        return cls(
            public_pct_home=public_pct_home,
            line_movement_direction=direction,
            is_divergence=False,
            signal_strength=0.0,
        )


__all__ = [
    "LineupInfo",
    "LineupStatus",
    "SharpDivergence",
    "WeatherAdjustment",
    "compute_weather_adjustment_mlb",
    "compute_weather_adjustment_nfl",
    "score_lineup_confidence",
]
