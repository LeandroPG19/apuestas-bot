"""Etiqueta de confianza multi-componente (Sprint 2 final).

Reemplaza el bug original de `telegram.py:2189` donde `ev_raw >= 0.05`
mapeaba directamente a "Muy alta" sin importar la probabilidad calibrada.
La fórmula nueva tiene 5 componentes con pesos que suman 1.0 exacto:

  edge_c   = min(EV, 0.15) / 0.15 * 0.40
  prob_c   = min(p_blended - 0.50, 0.25) / 0.25 * 0.20
  cert_c   = max(0, 0.20 - (p_up - p_low)) / 0.20 * 0.15
  cal_c    = max(0, 0.08 - rolling_ece_30d) / 0.08 * 0.15
  cons_c   = max(0, 0.08 - market_consensus_delta) / 0.08 * 0.10
  score    = suma (∈ [0, 1])

Soft tags `pricing_error` o `stale_line` multiplican por 0.80.

Referencias:
  - Walsh & Joshi 2024, ML with Applications v19 (calibración > accuracy)
  - plan §4.1 (Sprint 2)
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConfidenceLabel:
    stars: str
    label: str
    score: float  # [0, 1]


_TIERS: tuple[tuple[float, str, str], ...] = (
    (0.75, "⭐⭐⭐⭐⭐", "Muy alta"),
    (0.55, "⭐⭐⭐⭐", "Alta"),
    (0.35, "⭐⭐⭐", "Media"),
    (0.18, "⭐⭐", "Marginal"),
    (0.00, "⭐", "Baja"),
)

_SOFT_TAG_PENALTY = frozenset({"pricing_error", "stale_line"})
_SOFT_TAG_MULTIPLIER = 0.80


def classify_confidence(
    ev_raw: float,
    p_blended: float,
    *,
    p_lower: float | None = None,
    p_upper: float | None = None,
    rolling_ece_30d: float = 0.05,
    market_consensus_delta: float = 0.0,
    soft_tags: frozenset[str] = frozenset(),
) -> ConfidenceLabel:
    """Score multidimensional. Los parámetros por defecto (ECE=0.05,
    consensus=0.0) reflejan "no hay señal de calidad/consenso"; el
    componente respectivo contribuye positivamente sólo cuando la métrica
    real es mejor que el default.
    """
    interval_width = 0.20
    if p_lower is not None and p_upper is not None:
        interval_width = max(0.0, float(p_upper) - float(p_lower))

    edge_c = min(max(ev_raw, 0.0), 0.15) / 0.15 * 0.40
    prob_c = min(max(p_blended - 0.50, 0.0), 0.25) / 0.25 * 0.20
    cert_c = max(0.0, (0.20 - interval_width) / 0.20) * 0.15
    cal_c = max(0.0, (0.08 - rolling_ece_30d) / 0.08) * 0.15
    cons_c = max(0.0, (0.08 - market_consensus_delta) / 0.08) * 0.10

    score = edge_c + prob_c + cert_c + cal_c + cons_c

    if _SOFT_TAG_PENALTY & soft_tags:
        score *= _SOFT_TAG_MULTIPLIER

    score = max(0.0, min(1.0, score))

    for threshold, stars, label in _TIERS:
        if score >= threshold:
            return ConfidenceLabel(stars=stars, label=label, score=score)

    # Defensivo — nunca debería alcanzarse porque el último tier tiene threshold=0.
    return ConfidenceLabel(stars="⭐", label="Baja", score=score)


async def fetch_rolling_ece(session: AsyncSession, sport: str | None) -> float:
    """Lee `calibration_rolling` para obtener el ECE 30d del deporte.

    Devuelve 0.05 (default conservador) si no hay datos.
    """
    if not sport:
        return 0.05
    row = (
        await session.execute(
            text(
                """
                SELECT AVG(calibration_gap) AS ece
                FROM calibration_rolling
                WHERE sport_code = :sport
                  AND window_days = 30
                """
            ),
            {"sport": sport},
        )
    ).first()
    if row is None or row.ece is None:
        return 0.05
    try:
        return float(row.ece)
    except (TypeError, ValueError):
        return 0.05


__all__ = ["ConfidenceLabel", "classify_confidence", "fetch_rolling_ece"]
