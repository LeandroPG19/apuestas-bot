"""Model hierarchy resolver — Sprint 13 Capa 3.

Selecciona el modelo correcto para (sport, market, league_id) usando
priority-based fallback:

1. Modelo específico: (sport + market + league_id exacto) priority=0
2. Modelo sport-wide: (sport + market + league_id=NULL) priority=10-50
3. Catchall fallback: priority=99

El detector llama `resolve_and_load_model(sport, market, league_id)` y
siempre recibe un modelo (catchall si no hay otro).

Uso:
    info, estimator = await resolve_and_load_model(
        session, sport_code='soccer', market='h2h', league_id=6
    )
    # info.model_name → 'soccer_league_6' (Premier)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ResolvedModel:
    model_name: str
    model_version: str
    stage: str
    priority: int
    league_id: int | None
    is_catchall: bool


async def resolve_model_chain(
    session: Any,
    *,
    sport_code: str,
    market: str,
    league_id: int | None,
) -> list[ResolvedModel]:
    """Retorna cadena de modelos candidatos por priority ascendente.

    Primer resultado = modelo más específico. Último = catchall.
    Si model_hierarchy está vacío, retorna solo catchall_baseline.
    """
    try:
        # 1. Intentar modelo específico con league_id exacto
        rows_specific = []
        if league_id is not None:
            r = await session.execute(
                text(
                    """
                    SELECT mh.model_name, mh.priority, mh.league_id,
                           mrm.model_version, mrm.stage
                    FROM model_hierarchy mh
                    LEFT JOIN model_registry_meta mrm
                        ON mrm.model_name = mh.model_name AND mrm.stage = 'production'
                    WHERE mh.sport_code = :sport AND mh.market = :mkt
                      AND mh.league_id = :lg AND mh.active = true
                    ORDER BY mh.priority ASC
                    """
                ),
                {"sport": sport_code, "mkt": market, "lg": league_id},
            )
            rows_specific = r.fetchall()

        # 2. Sport-wide (league_id NULL)
        r2 = await session.execute(
            text(
                """
                SELECT mh.model_name, mh.priority, mh.league_id,
                       mrm.model_version, mrm.stage
                FROM model_hierarchy mh
                LEFT JOIN model_registry_meta mrm
                    ON mrm.model_name = mh.model_name AND mrm.stage = 'production'
                WHERE mh.sport_code = :sport AND mh.market = :mkt
                  AND mh.league_id IS NULL AND mh.active = true
                ORDER BY mh.priority ASC
                """
            ),
            {"sport": sport_code, "mkt": market},
        )
        rows_wide = r2.fetchall()
    except Exception as exc:
        logger.debug("hierarchy.query_fail", error=str(exc)[:100])
        rows_specific = []
        rows_wide = []

    chain: list[ResolvedModel] = []
    for row in list(rows_specific) + list(rows_wide):
        chain.append(
            ResolvedModel(
                model_name=str(row.model_name),
                model_version=str(row.model_version or "v1"),
                stage=str(row.stage or "none"),
                priority=int(row.priority),
                league_id=int(row.league_id) if row.league_id else None,
                is_catchall=str(row.model_name) == "catchall_baseline",
            )
        )

    if not chain:
        # Último fallback absoluto: catchall in-memory
        chain.append(
            ResolvedModel(
                model_name="catchall_baseline",
                model_version="v1",
                stage="builtin",
                priority=99,
                league_id=None,
                is_catchall=True,
            )
        )
    return chain


async def resolve_and_load_model(
    session: Any,
    *,
    sport_code: str,
    market: str,
    league_id: int | None,
) -> tuple[ResolvedModel, Any] | None:
    """Selecciona el mejor modelo disponible y lo carga.

    Intenta cada modelo en la cadena priority-ascending. El primero que
    cargue OK se devuelve. Si ninguno carga (ni catchall), devuelve None.
    """
    chain = await resolve_model_chain(
        session, sport_code=sport_code, market=market, league_id=league_id
    )
    if not chain:
        return None

    from apuestas.ml.registry import load_production_model

    for candidate in chain:
        # Catchall: instancia directa (no MLflow)
        if candidate.is_catchall:
            from apuestas.ml.catchall_baseline import CatchallBaselineModel

            logger.info(
                "hierarchy.resolved_catchall",
                sport=sport_code,
                market=market,
                league_id=league_id,
            )
            return candidate, CatchallBaselineModel()

        # Fase 2 — Bayesian xG soccer runtime (Scholtes-Karakuş 2025)
        if candidate.model_name.startswith("bayesian_xg_league_"):
            try:
                from apuestas.ml.bayesian_xg_runtime import BayesianXGModel

                lg_id = int(candidate.model_name.replace("bayesian_xg_league_", ""))
                model = BayesianXGModel(league_id=lg_id)
                logger.info(
                    "hierarchy.resolved_bayesian_xg",
                    league=lg_id,
                    sport=sport_code,
                )
                return candidate, model
            except Exception as exc:
                logger.debug("hierarchy.bayesian_xg_fail", error=str(exc)[:80])
                continue

        # Dixon-Coles cross-liga (495 teams cubiertos via team_strength_bayesian).
        # Para ligas sin modelo dedicado (UCL/UEL/Liga Portugal/Turkey/etc).
        if candidate.model_name == "dixon_coles_crossleague":
            try:
                from apuestas.ml.dixon_coles_runtime import (
                    DixonColesCrossLeagueModel,
                )

                model = DixonColesCrossLeagueModel()
                logger.info(
                    "hierarchy.resolved_dc_crossleague",
                    sport=sport_code,
                    market=market,
                    league_id=league_id,
                )
                return candidate, model
            except Exception as exc:
                logger.debug("hierarchy.dc_crossleague_fail", error=str(exc)[:80])
                continue

        # Modelos MLflow-registered — buscar por model_name exacto del hierarchy.
        # Fix #11: pattern incluye `%` wrap + exact-match guard post-load.
        try:
            exact_pattern = f"%{candidate.model_name}%"
            loaded = await load_production_model(
                sport_code, market, model_name_pattern=exact_pattern
            )
            if loaded is not None:
                info, estimator = loaded
                # Guard exact match (LIKE puede traer colisiones)
                if info.model_name != candidate.model_name:
                    logger.warning(
                        "hierarchy.name_mismatch_skip",
                        wanted=candidate.model_name,
                        got=info.model_name,
                        sport=sport_code,
                        league_id=league_id,
                    )
                    continue
                logger.info(
                    "hierarchy.resolved",
                    model=candidate.model_name,
                    priority=candidate.priority,
                    sport=sport_code,
                    market=market,
                    league_id=league_id,
                )
                return candidate, estimator
        except Exception as exc:
            logger.debug(
                "hierarchy.load_fail",
                model=candidate.model_name,
                error=str(exc)[:80],
            )
            continue

    # Ningún modelo cargó → catchall in-memory
    from apuestas.ml.catchall_baseline import CatchallBaselineModel

    fallback = (
        chain[-1]
        if chain
        else ResolvedModel(
            model_name="catchall_baseline",
            model_version="v1",
            stage="builtin",
            priority=99,
            league_id=None,
            is_catchall=True,
        )
    )
    logger.warning(
        "hierarchy.fallback_to_builtin_catchall",
        sport=sport_code,
        market=market,
    )
    return fallback, CatchallBaselineModel()


__all__ = [
    "ResolvedModel",
    "resolve_and_load_model",
    "resolve_model_chain",
]
