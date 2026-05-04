"""Execution timing — Sprint 11 Fase J.

Hipótesis operacional (Levitt 2004, Ed Miller 2019): las líneas de casas
**soft** (no Pinnacle) son **más débiles** en ventanas de staffing reducido:

- 06:00-08:00 UTC (early NY morning): reducción de traders → líneas
  manuales menos sharp, más tardías en reaccionar a steam.
- 30 min antes del kickoff: ritmo de ajustes menor (staff focus en gametime).
- Post-breaking news pero pre-ajuste formal (ventana ~5-15 min).

Este módulo expone helpers para:
1. Scorear si la hora actual es "óptima" para apostar en un deporte/liga.
2. Histórico empírico por (bookmaker, sport_code, hour_of_day) de edge bps
   (reutiliza `book_power_ratings.py`).
3. Recomendación de cuándo postponer vs apostar ya.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# Ventanas óptimas por deporte (en UTC). Aproximación inicial — se
# refina con histórico en `book_power_ratings.py`.
_OPTIMAL_WINDOWS_UTC: dict[str, tuple[tuple[int, int], ...]] = {
    "nba": ((6, 10), (21, 23)),  # early NY morning, 2h antes tip-off típico
    "nfl": ((6, 10), (14, 17)),  # early morning + antes de Sunday slate
    "mlb": ((6, 11),),  # early morning antes líneas ajustadas
    "nhl": ((6, 10),),
    "soccer": ((5, 9), (10, 12)),  # antes de mañana europea + pre-kickoff
    "tennis": ((5, 8), (10, 13)),
}

# Multiplicador de edge esperado por ventana (empírico, refinar con data).
_WINDOW_EDGE_MULTIPLIER = 1.30  # 30% más edge en ventana óptima
_LATE_KICKOFF_PENALTY = 0.60  # 40% menos edge si <30 min kickoff


@dataclass(slots=True)
class TimingScore:
    in_optimal_window: bool
    hours_until_kickoff: float
    edge_multiplier: float
    reason: str


def score_timing(
    *,
    sport_code: str,
    kickoff_utc: datetime,
    now_utc: datetime | None = None,
) -> TimingScore:
    """Evalúa si el momento es óptimo para enviar el pick al usuario."""
    now_utc = now_utc or datetime.now(UTC)
    hrs_until = (kickoff_utc - now_utc).total_seconds() / 3600.0

    if hrs_until < 0.5:
        return TimingScore(
            in_optimal_window=False,
            hours_until_kickoff=hrs_until,
            edge_multiplier=_LATE_KICKOFF_PENALTY,
            reason="late_kickoff_window",
        )

    hour = now_utc.hour
    windows = _OPTIMAL_WINDOWS_UTC.get(sport_code.lower(), ())
    in_window = any(start <= hour < end for (start, end) in windows)

    if in_window:
        return TimingScore(
            in_optimal_window=True,
            hours_until_kickoff=hrs_until,
            edge_multiplier=_WINDOW_EDGE_MULTIPLIER,
            reason=f"early_morning_window_{hour:02d}utc",
        )
    return TimingScore(
        in_optimal_window=False,
        hours_until_kickoff=hrs_until,
        edge_multiplier=1.0,
        reason="neutral",
    )


def recommend_delay(
    *,
    sport_code: str,
    kickoff_utc: datetime,
    now_utc: datetime | None = None,
) -> tuple[bool, int]:
    """¿Esperar a ventana óptima? Retorna (should_delay, seconds_to_wait)."""
    now_utc = now_utc or datetime.now(UTC)
    windows = _OPTIMAL_WINDOWS_UTC.get(sport_code.lower(), ())
    if not windows:
        return (False, 0)

    hrs_until = (kickoff_utc - now_utc).total_seconds() / 3600.0
    if hrs_until < 2:  # demasiado cerca del kickoff, no esperar
        return (False, 0)

    # ¿Hay ventana próxima dentro de las siguientes 4h?
    hour = now_utc.hour
    for start, end in windows:
        if hour < start and start - hour <= 4:
            delay = (start - hour) * 3600 - now_utc.minute * 60
            return (True, max(delay, 300))
    return (False, 0)


__all__ = ["TimingScore", "recommend_delay", "score_timing"]
