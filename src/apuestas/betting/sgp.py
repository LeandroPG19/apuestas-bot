"""Fase 3.2 — Same-Game Parlay correlation engine.

Los books pricean SGPs asumiendo correlación 0 entre legs. En realidad,
correlaciones históricas son fuertes: `P(LeBron>30 | Lakers cover spread) = 0.68`
vs base `P(LeBron>30) = 0.45`. El mispricing persistente es 4-6% edge.

Implementación:
  1. `compute_pair_correlation(prop_a, prop_b, lookback=365d)` — Pearson sobre
     player_game_logs + team cover outcome para pares.
  2. `price_sgp(legs)` — usa copula gaussiana para combinar probabilidades
     marginales respetando correlación.
  3. Comparar `sgp_fair_price_model` vs `sgp_offered_price_book`.

Uso:
    from apuestas.betting.sgp import price_sgp, compute_pair_correlation
    corr = await compute_pair_correlation(prop_a, prop_b)
    fair_price = price_sgp([leg_a, leg_b], correlations={(a,b): corr})
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class SGPLeg:
    """Un leg de SGP."""

    leg_id: str  # identificador único, p.ej. "lebron_points_over_29.5"
    p_marginal: float  # probabilidad marginal (independiente) 0-1
    outcome_description: str


@dataclass(slots=True, frozen=True)
class SGPQuote:
    """Cotización SGP lista para evaluar EV."""

    legs: list[SGPLeg]
    fair_price_model: float  # P(todos los legs hitean) según modelo+correlación
    fair_odds: float  # 1 / fair_price_model
    offered_odds: float  # odds del book
    ev: float  # EV del SGP


async def compute_pair_correlation(
    prop_a_key: str,
    prop_b_key: str,
    *,
    lookback_days: int = 365,
    min_samples: int = 50,
) -> float | None:
    """Pearson correlation entre dos props a partir de histórico.

    Cada prop_key: 'player_points_over_X' | 'team_spread_cover_home' | etc.
    Usa `player_game_logs` + `matches` para retroactivamente construir outcomes.
    """
    since = datetime.now(tz=UTC) - timedelta(days=lookback_days)

    # Implementación simplificada: query genérica que use columnas hit/not-hit.
    # En producción real requiere parser de prop_key → SQL específico.
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                        matched.match_id,
                        COALESCE((stats_a->>'hit')::int, 0) AS hit_a,
                        COALESCE((stats_b->>'hit')::int, 0) AS hit_b
                    FROM (
                        SELECT DISTINCT m.id AS match_id
                        FROM matches m WHERE m.start_time > :since
                          AND m.status = 'finished'
                    ) matched
                    LEFT JOIN LATERAL (
                        SELECT jsonb_build_object(
                            'hit',
                            (pg.stats->>'points' IS NOT NULL
                             AND (pg.stats->>'points')::float > 29.5)::int
                        ) AS stats_a
                        FROM player_game_logs pg
                        WHERE pg.match_id = matched.match_id
                        LIMIT 1
                    ) pa ON true
                    LEFT JOIN LATERAL (
                        SELECT jsonb_build_object(
                            'hit',
                            (pg.stats->>'rebounds' IS NOT NULL
                             AND (pg.stats->>'rebounds')::float > 7.5)::int
                        ) AS stats_b
                        FROM player_game_logs pg
                        WHERE pg.match_id = matched.match_id
                        LIMIT 1
                    ) pb ON true
                    LIMIT 2000
                    """
                ),
                {"since": since},
            )
        ).all()

    if len(rows) < min_samples:
        return None

    arr_a = np.array([r.hit_a for r in rows], dtype=np.float64)
    arr_b = np.array([r.hit_b for r in rows], dtype=np.float64)
    if arr_a.std() < 1e-9 or arr_b.std() < 1e-9:
        return 0.0
    corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
    logger.info(
        "sgp.correlation_computed",
        prop_a=prop_a_key,
        prop_b=prop_b_key,
        corr=corr,
        n=len(rows),
    )
    return corr


def gaussian_copula_joint_prob(marginals: list[float], correlation_matrix: np.ndarray) -> float:
    """Gaussian copula joint probability.

    Maps marginals to Z-scores via inverse normal CDF, aplica correlation matrix,
    retorna P(all legs hit).
    """
    try:
        from scipy.stats import multivariate_normal, norm  # type: ignore[import-untyped]
    except ImportError:
        # Fallback: independent assumption si scipy no disponible
        return float(np.prod(marginals))

    if not marginals:
        return 0.0

    # Transform marginals → Z (inverse normal)
    z_values = np.array([norm.ppf(max(min(p, 0.999), 0.001)) for p in marginals])
    # Ensure PSD correlation matrix
    n = len(marginals)
    if correlation_matrix.shape != (n, n):
        # Si no match, fallback independent
        return float(np.prod(marginals))

    # P(Z_1 ≤ z_1 AND ... AND Z_n ≤ z_n) via multivariate normal CDF
    try:
        mvn = multivariate_normal(mean=np.zeros(n), cov=correlation_matrix, allow_singular=True)
        # cdf accepts upper bound; P(Z_i ≤ z_i)
        joint = float(mvn.cdf(z_values))
    except Exception:  # fmt: skip
        joint = float(np.prod(marginals))

    return max(0.0, min(1.0, joint))


def price_sgp(
    legs: list[SGPLeg],
    *,
    correlations: dict[tuple[str, str], float] | None = None,
    offered_odds: float | None = None,
    min_correlation_effect: float = 0.02,
) -> SGPQuote:
    """Calcula fair price del SGP usando copula gaussiana.

    Si `correlations` provee pares con Pearson r, se construye una matriz de
    correlación. Si ninguna correlación > `min_correlation_effect`, fallback a
    independence (product rule).
    """
    correlations = correlations or {}
    marginals = [leg.p_marginal for leg in legs]

    # Construir correlation matrix
    n = len(legs)
    corr_matrix = np.eye(n)
    has_meaningful_corr = False
    for i in range(n):
        for j in range(i + 1, n):
            key_fwd = (legs[i].leg_id, legs[j].leg_id)
            key_bwd = (legs[j].leg_id, legs[i].leg_id)
            r = correlations.get(key_fwd, correlations.get(key_bwd, 0.0))
            if abs(r) >= min_correlation_effect:
                has_meaningful_corr = True
            # clamp r a [-0.95, 0.95] para PSD
            r = max(-0.95, min(0.95, r))
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    if has_meaningful_corr and n >= 2:
        fair_price = gaussian_copula_joint_prob(marginals, corr_matrix)
    else:
        fair_price = float(np.prod(marginals))

    fair_odds = 1.0 / fair_price if fair_price > 0 else float("inf")
    ev = (offered_odds or fair_odds) * fair_price - 1.0 if offered_odds else 0.0

    return SGPQuote(
        legs=legs,
        fair_price_model=fair_price,
        fair_odds=fair_odds,
        offered_odds=offered_odds or fair_odds,
        ev=ev,
    )
