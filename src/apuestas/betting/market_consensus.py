"""Consensus sharp de 3 fuentes: Pinnacle + Polymarket + Kalshi (Sprint 6c).

Combina probabilidades devigged de Pinnacle con midpoints CLOB de
Polymarket y Kalshi. Si las 3 coinciden (±5pp), el consenso es fuerte.
Si el modelo propio diverge >8pp del consenso, se marca el pick con
`soft_tag='market_disagreement'` para que `classify_confidence` baje
un tier (ya soportado por soft_tags).

Decisiones (plan §6):
  - Pinnacle peso 0.50 (más líquido, market maker).
  - Polymarket peso 0.30 si volume > $10k (R7).
  - Kalshi peso 0.20 si disponible para el deporte.
  - Si falta una fuente, se renormalizan los pesos sobre las presentes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    p_consensus: float
    dispersion: float  # desviación estándar de las fuentes usadas
    sources: int  # cuántas fuentes contribuyeron (1..3)
    pinnacle: float | None
    polymarket: float | None
    kalshi: float | None


_MIN_POLYMARKET_VOLUME = 10_000.0


def compute_consensus_sharp(
    *,
    pinnacle_devigged: float | None,
    polymarket_mid: float | None,
    kalshi_mid: float | None,
    polymarket_volume_usd: float | None = None,
) -> ConsensusResult:
    """Retorna el consenso ponderado + dispersión.

    Args:
        pinnacle_devigged: prob fair de Pinnacle tras devig Power/Shin.
        polymarket_mid: midpoint CLOB Polymarket (0..1).
        kalshi_mid: midpoint CLOB Kalshi (0..1).
        polymarket_volume_usd: volumen del mercado Polymarket. Si es
            menor a $10k, el midpoint es ruido y se excluye (R7).

    Returns:
        ConsensusResult; p_consensus=0.5 + 0 sources si nada aporta.
    """
    weights: list[tuple[float, float]] = []  # (value, weight)
    pm_used: float | None = None
    if pinnacle_devigged is not None:
        weights.append((float(pinnacle_devigged), 0.50))
    if polymarket_mid is not None:
        vol_ok = polymarket_volume_usd is None or polymarket_volume_usd >= _MIN_POLYMARKET_VOLUME
        if vol_ok:
            weights.append((float(polymarket_mid), 0.30))
            pm_used = float(polymarket_mid)
    if kalshi_mid is not None:
        weights.append((float(kalshi_mid), 0.20))

    if not weights:
        return ConsensusResult(
            p_consensus=0.5,
            dispersion=0.0,
            sources=0,
            pinnacle=pinnacle_devigged,
            polymarket=polymarket_mid,
            kalshi=kalshi_mid,
        )

    total_w = sum(w for _, w in weights)
    p = sum(v * w for v, w in weights) / total_w
    values_only = [v for v, _ in weights]
    if len(values_only) >= 2:
        mean = sum(values_only) / len(values_only)
        var = sum((v - mean) ** 2 for v in values_only) / len(values_only)
        dispersion = var**0.5
    else:
        dispersion = 0.0
    return ConsensusResult(
        p_consensus=float(p),
        dispersion=float(dispersion),
        sources=len(weights),
        pinnacle=pinnacle_devigged,
        polymarket=pm_used,
        kalshi=kalshi_mid,
    )


def consensus_delta(p_model: float, consensus: ConsensusResult) -> float:
    """Diferencia absoluta entre p_model y p_consensus.

    Umbral 8pp (0.08) → disagreement. Lo consume `classify_confidence`
    como `market_consensus_delta` y reduce el `cons_c` component.
    """
    return abs(float(p_model) - consensus.p_consensus)


def is_significant_disagreement(
    p_model: float, consensus: ConsensusResult, *, threshold_pp: float = 0.08
) -> bool:
    """True si el modelo diverge >threshold_pp del consensus y hay ≥2 fuentes.

    Con una sola fuente el "consenso" no es robusto; no marcamos disagreement.
    """
    if consensus.sources < 2:
        return False
    return consensus_delta(p_model, consensus) >= threshold_pp


__all__ = [
    "ConsensusResult",
    "compute_consensus_sharp",
    "consensus_delta",
    "is_significant_disagreement",
]
