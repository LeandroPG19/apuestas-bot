"""Análisis on-demand multi-señal de un partido específico.

A diferencia del flow batch `deep_analysis_flow` que skipea silente cuando le
falta una señal, aquí ejecutamos TODAS las señales disponibles en paralelo y
fusionamos las que retornaron datos. Graceful degradation: si una señal falla,
las demás aumentan su peso relativo y reportamos al usuario qué se usó.

Pipeline (~30-60s por match):
  1. Resolución del match (id directo o "Home vs Away" + fecha próxima)
  2. Recolección de señales en paralelo:
     - Modelo production por sport+market+league (vía hierarchy resolver)
     - Dixon-Coles cross-liga (team_strength_bayesian)
     - Bayesian xG runtime si la liga lo tiene
     - Pinnacle de-vigged (sharp anchor universal)
     - Polymarket / Kalshi cuando exista
     - Lesiones + lineup status (mirror_check + injuries)
     - Clima venue (Open-Meteo)
     - Movimiento de línea 24h
     - LLM análisis cualitativo (DeepSeek pre_match)
  3. Fusión bayesiana ponderada por confianza histórica de cada señal
  4. Decisión por mercado disponible (h2h, totals, BTTS, AH)
  5. Reporte estructurado para el caller
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import text

from apuestas.betting.detector import EventOdds
from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


# ─── Result types ────────────────────────────────────────────────────────


@dataclass(slots=True)
class SignalProbs:
    """Una señal probabilística h2h por outcome (home/draw/away o home/away)."""

    name: str
    probs: dict[str, float]
    weight: float = 1.0
    confidence: float = 0.5  # 0-1, ajusta peso final
    note: str = ""


@dataclass(slots=True)
class PickRecommendation:
    market: str
    outcome: str
    line: float | None
    book: str
    odds: float
    p_fused: float
    edge: float
    ev: float
    confidence: str  # low/medium/high
    reasoning: str
    p_low: float | None = None  # B4 conformal band lower
    p_high: float | None = None  # B4 conformal band upper
    anticipated_clv: float | None = None  # B6 closing_line_predictor signal
    book_edge_bps: float | None = None  # B6 book_power_ratings rank
    kelly_quarter_pct: float | None = None  # B6 ¼ Kelly stake hint
    stale_warning: str | None = None  # EV > cap → posible stale book vs sharp fresh


@dataclass(slots=True)
class ExistingPickInfo:
    """Pick ya emitido por el detector batch para este match (para mostrar en /analizar)."""

    pick_id: int
    market: str
    outcome: str
    line: float | None
    book: str
    odds_placed: float
    placed_at: datetime
    status: str  # pending/confirmed/not_taken
    outcome_result: str | None  # won/lost/push/null
    p_consensus_sharp: float | None


@dataclass(slots=True)
class MatchAnalysisReport:
    match_id: int
    sport_code: str
    home_name: str
    away_name: str
    league_name: str | None
    start_time: datetime
    market: str = "h2h"
    signals_used: list[SignalProbs] = field(default_factory=list)
    fused_probs: dict[str, float] = field(default_factory=dict)
    fused_bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    picks: list[PickRecommendation] = field(default_factory=list)
    skipped_signals: list[str] = field(default_factory=list)
    skip_reasons: dict[str, str] = field(default_factory=dict)
    odds_freshness_warning: str | None = None
    existing_picks: list[ExistingPickInfo] = field(default_factory=list)
    llm_reasoning: dict[str, Any] | None = None
    ambiguous_candidates: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0
    summary_es: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "sport_code": self.sport_code,
            "home": self.home_name,
            "away": self.away_name,
            "league": self.league_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "market": self.market,
            "signals": [
                {
                    "name": s.name,
                    "probs": s.probs,
                    "confidence": s.confidence,
                    "weight": s.weight,
                    "note": s.note,
                }
                for s in self.signals_used
            ],
            "fused_probs": self.fused_probs,
            "fused_bands": {k: list(v) for k, v in self.fused_bands.items()},
            "picks": [
                {
                    "market": p.market,
                    "outcome": p.outcome,
                    "line": p.line,
                    "book": p.book,
                    "odds": p.odds,
                    "p_fused": p.p_fused,
                    "p_low": p.p_low,
                    "p_high": p.p_high,
                    "edge": p.edge,
                    "ev": p.ev,
                    "confidence": p.confidence,
                    "reasoning": p.reasoning,
                    "anticipated_clv": p.anticipated_clv,
                    "book_edge_bps": p.book_edge_bps,
                    "kelly_quarter_pct": p.kelly_quarter_pct,
                    "stale_warning": p.stale_warning,
                }
                for p in self.picks
            ],
            "skipped_signals": self.skipped_signals,
            "skip_reasons": self.skip_reasons,
            "odds_freshness_warning": self.odds_freshness_warning,
            "existing_picks": [
                {
                    "pick_id": ep.pick_id,
                    "market": ep.market,
                    "outcome": ep.outcome,
                    "line": ep.line,
                    "book": ep.book,
                    "odds_placed": ep.odds_placed,
                    "placed_at": ep.placed_at.isoformat() if ep.placed_at else None,
                    "status": ep.status,
                    "outcome_result": ep.outcome_result,
                    "p_consensus_sharp": ep.p_consensus_sharp,
                }
                for ep in self.existing_picks
            ],
            "llm_reasoning": self.llm_reasoning,
            "ambiguous_candidates": self.ambiguous_candidates,
            "duration_s": self.duration_s,
            "summary_es": self.summary_es,
        }


# ─── Match resolver ──────────────────────────────────────────────────────


# Aliases comunes que la gente escribe vs el nombre canónico en DB.
# Si tu query menciona la izquierda, expandimos el LIKE pattern a la derecha.
_TEAM_ALIASES: dict[str, list[str]] = {
    "psg": ["paris saint-germain", "paris sg", "paris", "parisinos"],
    "bayern": ["bayern munich", "fc bayern", "bayern münchen"],
    "real": ["real madrid"],
    "real madrid": ["real madrid"],
    "merengues": ["real madrid"],
    "real sociedad": ["real sociedad"],
    "barca": ["barcelona", "fc barcelona", "barça", "blaugranas"],
    "city": ["manchester city", "man city", "citizens"],
    "united": ["manchester united", "man united", "man utd", "red devils"],
    "atletico": ["atletico madrid", "atlético madrid", "atleti", "colchoneros"],
    "atleti": ["atletico madrid", "atlético madrid"],
    "atletico de madrid": ["atletico madrid", "atlético madrid"],
    "inter": ["inter milan", "internazionale"],
    "inter miami": ["inter miami"],
    "milan": ["ac milan"],
    "juve": ["juventus"],
    "boca": ["boca juniors"],
    "river": ["river plate"],
    "santos": ["santos fc"],
    "sao paulo": ["sao paulo", "são paulo"],
    "flamengo": ["cr flamengo"],
    "arsenal": ["arsenal", "gunners"],
    "gunners": ["arsenal"],
    "tottenham": ["tottenham hotspur", "spurs"],
    "spurs": ["tottenham hotspur"],
    "chelsea": ["chelsea", "blues"],
    "blues": ["chelsea"],
    "liverpool": ["liverpool", "reds"],
    "yankees": ["new york yankees"],
    "red sox": ["boston red sox"],
    "dodgers": ["los angeles dodgers"],
    "lakers": ["los angeles lakers"],
    "celtics": ["boston celtics"],
    "warriors": ["golden state warriors"],
    "tigres": ["tigres uanl"],
    "america": ["club america", "club américa"],
    "chivas": ["guadalajara"],
}

# Términos ambiguos que pueden referirse a >1 team. Si la query usa solo
# uno de estos sin mayor contexto, el resolver debería pedir clarificación.
_AMBIGUOUS_TERMS = frozenset({"real", "inter"})


def _expand_query(q: str) -> list[str]:
    """Devuelve la query original + aliases conocidos como patterns LIKE."""
    q_lower = q.strip().lower()
    candidates = [q_lower]
    for alias, expansions in _TEAM_ALIASES.items():
        if alias in q_lower:
            candidates.extend(expansions)
        for exp in expansions:
            if exp in q_lower:
                candidates.append(alias)
    return list({c for c in candidates if c})


async def resolve_match(query: str | int) -> dict[str, Any] | None:
    """Resuelve match por id (int) o por "Home vs Away" (string).

    Para strings:
      1. Parte por separadores (vs, v, x, -)
      2. Expande aliases conocidos (PSG→Paris Saint-Germain, etc.)
      3. Busca matches scheduled próximas 7 días con substring + trigram
      4. Si hay múltiples, prefiere por cobertura de books y proximidad
    """
    if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
        match_id = int(query)
        async with session_scope() as session:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT m.id, m.sport_code, m.start_time, m.league_id,
                               m.home_team_id, m.away_team_id, m.status,
                               ht.name AS home_name, at.name AS away_name,
                               l.name AS league_name
                        FROM matches m
                        JOIN teams ht ON ht.id = m.home_team_id
                        JOIN teams at ON at.id = m.away_team_id
                        LEFT JOIN leagues l ON l.id = m.league_id
                        WHERE m.id = :mid
                        """
                    ),
                    {"mid": match_id},
                )
            ).first()
            return dict(row._mapping) if row else None

    raw = str(query).strip().lower()
    home_q: str | None = None
    away_q: str | None = None
    for sep in (" vs ", " v ", " - ", " x ", " contra ", " versus "):
        if sep in raw:
            parts = raw.split(sep, 1)
            home_q, away_q = parts[0].strip(), parts[1].strip()
            break
    if not home_q or not away_q:
        return None

    home_candidates = _expand_query(home_q)
    away_candidates = _expand_query(away_q)
    home_likes = [f"%{c}%" for c in home_candidates]
    away_likes = [f"%{c}%" for c in away_candidates]

    # SCORING ESTRICTO: el match SOLO califica si AMBOS teams matchean (en
    # cualquier orden home/away). Antes con `OR LIKE ANY` cualquier match con
    # uno solo de los nombres pasaba — ej. "Real vs Barcelona" devolvía
    # "Osasuna vs Barcelona" (sólo Barcelona match) en vez de Real Madrid.
    # Score = 0..2: cuenta cuántos teams matchearon vía alias O substring.
    # Sin esto la query ambigua siempre cae a partidos populares en el log.
    async with session_scope() as session:
        result = await session.execute(
            text(
                """
                WITH scored AS (
                  SELECT m.id, m.sport_code, m.start_time, m.league_id,
                         m.home_team_id, m.away_team_id, m.status,
                         ht.name AS home_name, at.name AS away_name,
                         l.name AS league_name,
                         (
                           SELECT COUNT(DISTINCT bookmaker) FROM odds_history oh
                           WHERE oh.match_id = m.id
                             AND oh.ts > NOW() - INTERVAL '24 hours'
                         ) AS n_books,
                         -- Score por team: 1 si calza con cualquier alias o
                         -- substring de los términos de la query, 0 si no.
                         CASE
                           WHEN LOWER(ht.name) LIKE ANY(:home_likes)
                                OR similarity(LOWER(ht.name), :hq) >= 0.4 THEN 1
                           ELSE 0
                         END AS home_match_normal,
                         CASE
                           WHEN LOWER(at.name) LIKE ANY(:away_likes)
                                OR similarity(LOWER(at.name), :aq) >= 0.4 THEN 1
                           ELSE 0
                         END AS away_match_normal,
                         CASE
                           WHEN LOWER(ht.name) LIKE ANY(:away_likes)
                                OR similarity(LOWER(ht.name), :aq) >= 0.4 THEN 1
                           ELSE 0
                         END AS home_match_flipped,
                         CASE
                           WHEN LOWER(at.name) LIKE ANY(:home_likes)
                                OR similarity(LOWER(at.name), :hq) >= 0.4 THEN 1
                           ELSE 0
                         END AS away_match_flipped,
                         GREATEST(
                           similarity(LOWER(ht.name), :hq) + similarity(LOWER(at.name), :aq),
                           similarity(LOWER(ht.name), :aq) + similarity(LOWER(at.name), :hq)
                         ) AS sim
                  FROM matches m
                  JOIN teams ht ON ht.id = m.home_team_id
                  JOIN teams at ON at.id = m.away_team_id
                  LEFT JOIN leagues l ON l.id = m.league_id
                  WHERE m.status = 'scheduled'
                    AND m.start_time BETWEEN NOW() - INTERVAL '2 hours'
                                         AND NOW() + INTERVAL '7 days'
                )
                SELECT *,
                       GREATEST(
                         home_match_normal + away_match_normal,
                         home_match_flipped + away_match_flipped
                       ) AS team_score
                FROM scored
                -- AMBOS teams deben matchear (score=2 en cualquier orientación)
                WHERE GREATEST(
                        home_match_normal + away_match_normal,
                        home_match_flipped + away_match_flipped
                      ) = 2
                ORDER BY n_books DESC, sim DESC, start_time ASC
                LIMIT 5
                """
            ),
            {
                "hq": home_q,
                "aq": away_q,
                "home_likes": home_likes,
                "away_likes": away_likes,
            },
        )
        rows = result.fetchall()
        if not rows:
            return None
        top = dict(rows[0]._mapping)
        # B9 ambigüedad: si query usa término ambiguo SIN otro discriminador
        # y hay >=2 candidatos con sim cercano, exponer alternativas para que
        # el caller (cmd_analizar) las muestre al user.
        ambiguous_query = any(
            t in (home_q, away_q) or t in (home_q.split() + away_q.split())
            for t in _AMBIGUOUS_TERMS
        )
        if ambiguous_query and len(rows) >= 2:
            top_sim = float(getattr(rows[0], "sim", 0.0) or 0.0)
            second_sim = float(getattr(rows[1], "sim", 0.0) or 0.0)
            # Si el segundo está dentro del 15% del top → ambiguo
            if top_sim > 0 and (top_sim - second_sim) / top_sim < 0.15:
                top["_ambiguous_candidates"] = [dict(r._mapping) for r in rows[:5]]
        return top


# ─── Señales individuales (cada una graceful degradation) ────────────────


async def _signal_production_model(
    sport_code: str,
    market: str,
    league_id: int | None,
    event: dict[str, Any],
    odds: EventOdds | None = None,
) -> SignalProbs | None:
    """Carga modelo production via hierarchy resolver y predict.

    Soporta 4 dispatch paths:
      1. BayesianXGModel (soccer Bayesian xG por liga)
      2. _IndependentPoissonModel / _DCModelWithMap (DC con map team_id→strength)
      3. CatchallBaselineModel (fallback Pinnacle-prior calibrado)
      4. sklearn estándar (LGBM/CatBoost para mlb_*, nba_moneyline) via build_match_features

    Retorna None solo si NINGUNO de los dispatch aplica o todos fallan.
    """
    try:
        from apuestas.ml.model_hierarchy_resolver import resolve_and_load_model

        async with session_scope() as session:
            resolved = await resolve_and_load_model(
                session, sport_code=sport_code, market=market, league_id=league_id
            )
        if resolved is None:
            return None
        info, raw_obj = resolved
        if isinstance(raw_obj, dict):
            estimator = raw_obj.get("estimator") or raw_obj.get("model")
            feature_names = raw_obj.get("feature_names") or []
        else:
            estimator = raw_obj
            feature_names = getattr(raw_obj, "feature_names_in_", []) or []
        if estimator is None:
            return None
        # Fallback: si raw_obj era dict sin feature_names pero el estimator
        # tiene feature_names_in_ (sklearn estándar), usarlo.
        if not feature_names:
            fn_attr = getattr(estimator, "feature_names_in_", None)
            if fn_attr is not None:
                try:
                    feature_names = list(fn_attr)
                except Exception:
                    feature_names = []

        from apuestas.ml.bayesian_xg_runtime import BayesianXGModel
        from apuestas.ml.catchall_baseline import CatchallBaselineModel

        # Path 1: BayesianXG (soccer por liga)
        if isinstance(estimator, BayesianXGModel):
            home_id = int(event["home_team_id"])
            away_id = int(event["away_team_id"])
            proba = estimator.predict_proba(np.array([[home_id, away_id]]))
            return SignalProbs(
                name=f"production:{info.model_name}",
                probs={
                    "away": float(proba[0, 0]),
                    "draw": float(proba[0, 1]),
                    "home": float(proba[0, 2]),
                },
                weight=1.2,
                confidence=0.75,
                note="Bayesian xG league posterior",
            )

        # Path 2: Independent Poisson / DC map
        cls_name = getattr(estimator, "__class__", type).__name__
        if cls_name in {"_IndependentPoissonModel", "_DCModelWithMap"} and hasattr(
            estimator, "predict"
        ):
            home_id = int(event["home_team_id"])
            away_id = int(event["away_team_id"])
            try:
                pred = await asyncio.to_thread(estimator.predict, home_id, away_id)
                p_h, p_d, p_a = pred.home_draw_away
                return SignalProbs(
                    name=f"production:{info.model_name}",
                    probs={
                        "home": float(p_h),
                        "draw": float(p_d),
                        "away": float(p_a),
                    },
                    weight=1.0,
                    confidence=0.65,
                    note="Independent Poisson / DC map",
                )
            except Exception as exc:
                logger.debug("agent.indep_poisson_fail", error=str(exc)[:120])

        # Path 3: Catchall baseline (Pinnacle prior recalibrado).
        # SOLO h2h: catchall predice home/away, no over/under ni spread.
        # Para markets distintos no aplica (devolvería probs h2h confundiendo
        # la fusión que espera over/under).
        if isinstance(estimator, CatchallBaselineModel) and market == "h2h":
            pinn_p_home = None
            if odds is not None and "pinnacle" in odds.quotes_by_bookmaker:
                pinn_quotes = odds.quotes_by_bookmaker["pinnacle"]
                # 2-way: pinn_quotes[0]=home odds. Implied home = 1/odds_home /
                # (1/odds_home + 1/odds_away) — sin de-vig fancy, suficiente prior.
                try:
                    if (
                        len(pinn_quotes) >= 2
                        and pinn_quotes[0]
                        and pinn_quotes[1]
                        and pinn_quotes[0] > 1.0
                        and pinn_quotes[1] > 1.0
                    ):
                        ih = 1.0 / float(pinn_quotes[0])
                        ia = 1.0 / float(pinn_quotes[1])
                        pinn_p_home = ih / (ih + ia)
                except Exception:
                    pinn_p_home = None
            if pinn_p_home is None:
                return None
            try:
                proba = await asyncio.to_thread(
                    estimator.predict_proba, np.array([[float(pinn_p_home)]])
                )
                p_home = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
                return SignalProbs(
                    name="production:catchall_baseline",
                    probs={"home": p_home, "away": 1.0 - p_home},
                    weight=0.6,  # solo recalibra Pinnacle, peso bajo
                    confidence=0.45,
                    note="Catchall: Pinnacle prior recalibrado",
                )
            except Exception as exc:
                logger.debug("agent.catchall_fail", error=str(exc)[:120])
                return None

        # Path 4: sklearn estándar (LGBM/CatBoost) — requiere features rolling
        if hasattr(estimator, "predict_proba") and feature_names:
            try:
                from apuestas.features.feature_store import build_match_features

                X_vec = await build_match_features(
                    sport_code=sport_code,
                    home_team_id=int(event["home_team_id"]),
                    away_team_id=int(event["away_team_id"]),
                    match_start=event.get("start_time"),
                    feature_names=list(feature_names),
                )
                if X_vec is None:
                    logger.debug("agent.sklearn_features_unavailable")
                    return None
                X = X_vec.reshape(1, -1) if X_vec.ndim == 1 else X_vec
                proba = await asyncio.to_thread(estimator.predict_proba, X)
                classes = list(getattr(estimator, "classes_", [0, 1]))
                if len(classes) == 2:
                    idx_home = classes.index(1) if 1 in classes else 1
                    p_home = float(proba[0, idx_home])
                    return SignalProbs(
                        name=f"production:{info.model_name}",
                        probs={"home": p_home, "away": 1.0 - p_home},
                        weight=1.2,
                        confidence=0.70,
                        note=f"sklearn {cls_name} 2-way",
                    )
                if len(classes) == 3:
                    return SignalProbs(
                        name=f"production:{info.model_name}",
                        probs={
                            "home": float(proba[0, 0]),
                            "draw": float(proba[0, 1]),
                            "away": float(proba[0, 2]),
                        },
                        weight=1.2,
                        confidence=0.70,
                        note=f"sklearn {cls_name} 3-way",
                    )
            except Exception as exc:
                logger.debug("agent.sklearn_dispatch_fail", error=str(exc)[:120])
        return None
    except Exception as exc:
        logger.debug("agent.production_model.fail", error=str(exc)[:120])
        return None


async def _signal_dixon_coles(
    event: dict[str, Any],
    market: str = "h2h",
    *,
    line: float | None = None,
) -> SignalProbs | None:
    """Dixon-Coles cross-liga via team_strength_bayesian.

    Markets soportados: h2h (1X2), totals (over/under con `line`), btts (yes/no).
    Para spreads/runline soccer no hay handicap nativo en DC: skip.
    """
    if event.get("sport_code") not in _SOCCER_SPORTS:
        return None
    home_id = int(event["home_team_id"])
    away_id = int(event["away_team_id"])
    try:
        if market == "h2h":
            from apuestas.features.soccer import dixon_coles_predict

            result = await asyncio.to_thread(dixon_coles_predict, home_id, away_id)
            if result is None:
                return None
            return SignalProbs(
                name="dixon_coles_crossleague",
                probs={
                    "home": result["p_home"],
                    "draw": result["p_draw"],
                    "away": result["p_away"],
                },
                weight=1.0,
                confidence=0.70,
                note="DC cross-liga sobre strength bayesian",
            )
        if market in ("totals", "totals_team"):
            from apuestas.features.soccer import dixon_coles_predict_total

            ln = float(line) if line is not None else 2.5
            result = await asyncio.to_thread(dixon_coles_predict_total, home_id, away_id, ln)
            if result is None:
                return None
            return SignalProbs(
                name="dixon_coles_crossleague",
                probs={"over": result["over"], "under": result["under"]},
                weight=1.0,
                confidence=0.65,
                note=f"DC cross-liga totals @ {ln}",
            )
        if market == "btts":
            from apuestas.features.soccer import dixon_coles_predict_btts

            result = await asyncio.to_thread(dixon_coles_predict_btts, home_id, away_id)
            if result is None:
                return None
            return SignalProbs(
                name="dixon_coles_crossleague",
                probs={"yes": result["yes"], "no": result["no"]},
                weight=1.0,
                confidence=0.65,
                note="DC cross-liga BTTS",
            )
        return None
    except Exception as exc:
        logger.debug("agent.dixon_coles.fail", error=str(exc)[:120])
        return None


async def _noop_signal() -> SignalProbs | None:
    """Placeholder para gather() cuando un signal no aplica al market actual."""
    return None


async def _signal_statsbomb_form(event: dict[str, Any]) -> SignalProbs | None:
    """Forma reciente vía StatsBomb event-level (xG, progressive passes).

    Computa rolling de últimos 10 matches para home/away y emite señal direccional:
    si home_xg_mean > away_xg_mean por margen >0.4 → boost home (+5pp).
    Solo aplica para soccer h2h. Requiere mapeo team_external_id source='statsbomb'.

    Diseño defensivo: si los teams no están mapeados o no hay rolling, retorna None
    silentemente (anchor pseudo-sharp neutraliza cuando hay otras señales).
    """
    if event.get("sport_code") not in _SOCCER_SPORTS:
        return None
    home_id = int(event["home_team_id"])
    away_id = int(event["away_team_id"])
    try:
        from apuestas.features.statsbomb_features import compute_team_rolling_from_sb

        async with session_scope() as session:
            sb_home_row = (
                await session.execute(
                    text(
                        """
                        SELECT external_id FROM team_external_id
                        WHERE team_id = :tid AND source = 'statsbomb'
                        ORDER BY confidence DESC NULLS LAST LIMIT 1
                        """
                    ),
                    {"tid": home_id},
                )
            ).first()
            sb_away_row = (
                await session.execute(
                    text(
                        """
                        SELECT external_id FROM team_external_id
                        WHERE team_id = :tid AND source = 'statsbomb'
                        ORDER BY confidence DESC NULLS LAST LIMIT 1
                        """
                    ),
                    {"tid": away_id},
                )
            ).first()
            if sb_home_row is None or sb_away_row is None:
                return None
            sb_home = int(sb_home_row.external_id)
            sb_away = int(sb_away_row.external_id)
            # match_id grande para "no filtrar matches futuros" (anti-leakage trivial)
            cutoff = 10**9
            # Cap a 5s cada side: agregación de event_jsonb en 10 matches puede
            # ser pesada (100k+ events/match). En timeout devolvemos None → no
            # contamina fusion con datos parciales.
            try:
                home_form, away_form = await asyncio.wait_for(
                    asyncio.gather(
                        compute_team_rolling_from_sb(session, sb_home, cutoff, window=5),
                        compute_team_rolling_from_sb(session, sb_away, cutoff, window=5),
                    ),
                    timeout=8.0,
                )
            except TimeoutError:
                logger.debug("agent.statsbomb_form.timeout", home=sb_home, away=sb_away)
                return None

        if not home_form or not away_form:
            return None
        if home_form.get("sb_n_matches", 0) < 5 or away_form.get("sb_n_matches", 0) < 5:
            return None

        h_xg = float(home_form.get("sb_xg_mean", 0.0))
        a_xg = float(away_form.get("sb_xg_mean", 0.0))
        # Mapping xG diff → probabilidad direccional. xG diff +0.5 ≈ +5pp swing,
        # +1.0 ≈ +10pp. Cap en ±15pp para evitar over-confidence.
        diff = h_xg - a_xg
        swing = max(-0.15, min(0.15, diff * 0.10))
        # Base 1X2 (sin info): home 0.45 / draw 0.27 / away 0.28 (avg ligas)
        p_home = 0.45 + swing
        p_away = 0.28 - swing
        p_draw = max(0.0, 1.0 - p_home - p_away)
        s = p_home + p_draw + p_away
        return SignalProbs(
            name="statsbomb_form",
            probs={"home": p_home / s, "draw": p_draw / s, "away": p_away / s},
            weight=0.5,
            confidence=0.40,
            note=f"SB form xG_diff={diff:+.2f} (h_n={home_form['sb_n_matches']:.0f}, "
            f"a_n={away_form['sb_n_matches']:.0f})",
        )
    except Exception as exc:
        logger.debug("agent.statsbomb_form.fail", error=str(exc)[:120])
        return None


async def _pinnacle_quote_age_hours(match_id: int) -> float | None:
    """Edad en horas de la última quote de Pinnacle para este match."""
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT EXTRACT(EPOCH FROM (NOW() - MAX(ts))) / 3600.0 AS h
                    FROM odds_history
                    WHERE match_id = :mid AND bookmaker = 'pinnacle'
                    """
                ),
                {"mid": match_id},
            )
        ).first()
    if row is None or row.h is None:
        return None
    return float(row.h)


async def _signal_pinnacle_devigged(
    odds: EventOdds | None, match_id: int | None = None
) -> SignalProbs | None:
    """Pinnacle de-vigged como sharp anchor universal.

    Staleness check: si la quote de Pinnacle tiene >2h baja confidence,
    >6h skipea. Pinnacle ajusta agresivo cerca del kickoff; un fair de hace
    8h puede estar desactualizado por lineups/lesiones publicadas después.
    """
    if odds is None or "pinnacle" not in odds.quotes_by_bookmaker:
        return None
    try:
        from apuestas.betting.devig import shin

        quotes = odds.quotes_by_bookmaker["pinnacle"]
        valid = [q for q in quotes if q and q > 1.0]
        if len(valid) < 2:
            return None
        fair = shin(valid)
        probs: dict[str, float] = {}
        for i, outcome in enumerate(odds.outcomes):
            if i < len(fair):
                probs[outcome] = float(fair[i])
        if not probs:
            return None

        # Staleness: ajustar confidence/weight según edad
        confidence = 0.85
        weight = 1.5
        note = "Sharp anchor — Shin de-vigging"
        age_h: float | None = None
        if match_id is not None:
            age_h = await _pinnacle_quote_age_hours(match_id)
            if age_h is not None:
                if age_h > 6.0:
                    logger.debug("agent.pinnacle_stale_skip", age_h=age_h)
                    return None
                if age_h > 2.0:
                    confidence = 0.60
                    weight = 1.0
                    note = f"Pinnacle de-vigged (stale {age_h:.1f}h)"
        return SignalProbs(
            name="pinnacle_devig",
            probs=probs,
            weight=weight,
            confidence=confidence,
            note=note,
        )
    except Exception as exc:
        logger.debug("agent.pinnacle_devig.fail", error=str(exc)[:120])
        return None


async def _polymarket_volume_24h(match_id: int) -> float | None:
    """Liquidez 24h USD del market polymarket asociado al match (si existe)."""
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT pm.volume_24h_usd
                    FROM polymarket_markets pm
                    JOIN matches m ON m.sport_code = pm.sport_code
                    WHERE m.id = :mid
                      AND pm.end_date BETWEEN m.start_time - INTERVAL '12 hours'
                                           AND m.start_time + INTERVAL '12 hours'
                    ORDER BY pm.last_updated DESC
                    LIMIT 1
                    """
                ),
                {"mid": match_id},
            )
        ).first()
    if row is None or row.volume_24h_usd is None:
        return None
    return float(row.volume_24h_usd)


async def _signal_polymarket(
    odds: EventOdds | None, match_id: int | None = None
) -> SignalProbs | None:
    """Polymarket retail anchor cuando esté disponible.

    Liquidez check: <$100 24h → señal demasiado ruidosa, skip.
    $100-$1k → confidence 0.40 (peso bajo).
    >$1k → confidence 0.65 default.
    """
    if odds is None or "polymarket" not in odds.quotes_by_bookmaker:
        return None
    try:
        quotes = odds.quotes_by_bookmaker["polymarket"]
        valid = [q for q in quotes if q and q > 1.0]
        if len(valid) < 2:
            return None
        implied = [1.0 / q for q in valid]
        s = sum(implied)
        if s <= 0:
            return None
        normalized = [v / s for v in implied]
        probs = {odds.outcomes[i]: normalized[i] for i in range(len(normalized))}

        confidence = 0.65
        note = "Retail prediction market"
        if match_id is not None:
            vol = await _polymarket_volume_24h(match_id)
            if vol is not None:
                if vol < 100.0:
                    logger.debug("agent.polymarket_illiquid_skip", vol=vol)
                    return None
                if vol < 1000.0:
                    confidence = 0.40
                    note = f"Polymarket low-liquidity (${vol:.0f}/24h)"
                else:
                    note = f"Polymarket retail (${vol:.0f}/24h)"
        return SignalProbs(
            name="polymarket",
            probs=probs,
            weight=0.8,
            confidence=confidence,
            note=note,
        )
    except Exception as exc:
        logger.debug("agent.polymarket.fail", error=str(exc)[:120])
        return None


_SOCCER_SPORTS = frozenset({"soccer", "epl", "laliga", "bundesliga", "seriea", "ligue1", "liga_mx"})


async def _fetch_rag_snippets(event: dict[str, Any], max_chars: int = 1500) -> str:
    """RAG context para el LLM (B6): últimas noticias de los teams + liga.

    Hybrid search BGE-M3 dense + BM25 sparse + RRF (Cormack 2009).
    Si embed cae a zeros (B7 fallback), el RRF degrada pero sparse sigue
    aportando. Si todo falla retorna "" → LLM razona con event metadata.
    """
    try:
        from apuestas.llm.embed import EmbedClient
        from apuestas.llm.rag import RAGRetriever

        home = str(event.get("home_name") or "").strip()
        away = str(event.get("away_name") or "").strip()
        league = str(event.get("league_name") or "").strip()
        if not home or not away:
            return ""
        query = f"{home} vs {away} {league}".strip()
        team_ids = [
            int(event["home_team_id"]),
            int(event["away_team_id"]),
        ]
        sport_code = str(event.get("sport_code") or "")
        async with EmbedClient() as embed_client:
            retr = RAGRetriever(embed_client=embed_client)
            hits = await retr.hybrid_search(
                query,
                top_k=5,
                sports=[sport_code] if sport_code else None,
                team_ids=team_ids,
            )
            if not hits:
                return ""
            return retr.format_snippets(hits, max_chars=max_chars // max(1, len(hits)))
    except Exception as exc:
        logger.debug("agent.rag_fetch_fail", error=str(exc)[:120])
        return ""


async def _signal_llm_qualitative(
    event: dict[str, Any], market: str = "h2h"
) -> tuple[SignalProbs, dict[str, Any]] | None:
    """LLM análisis cualitativo: direccional adjustment + reasoning chain.

    Solo emite probs en mercado h2h (su prior es 3-way o 2-way de ganador).
    Para totals/spreads/btts retorna None pero igual ejecuta el LLM para
    capturar reasoning chain (key_factors, risks) que se persiste como
    contexto sin contaminar `_fuse_signals` con probs del market equivocado.

    Prior por sport:
      - 3-way (soccer h2h): home 0.45 / draw 0.27 / away 0.28
      - 2-way (NBA/MLB/NFL/tennis h2h): home 0.55 / away 0.45
    """
    try:
        from apuestas.flows.deep_analysis import llm_analyze_event

        sport_code = str(event.get("sport_code") or "").lower()
        is_3way = sport_code in _SOCCER_SPORTS

        # B6 fetch_rag: inyectar contexto de noticias relevantes
        rag_snippets = await _fetch_rag_snippets(event)
        # `llm_analyze_event` es un @task de Prefect. Cuando se invoca fuera
        # de un flow context (caso del agente on-demand), Prefect intenta
        # contactar el server (PREFECT_API_URL=http://prefect:4200/api/) y falla.
        # `.fn` devuelve la función plana sin la capa de Prefect runtime.
        llm_fn = getattr(llm_analyze_event, "fn", llm_analyze_event)
        result = await llm_fn(event, rag_snippets=rag_snippets, correlation_id="agent")
        if result is None:
            return None
        direction = result.get("edge_direction", "neutral")
        confidence = result.get("confidence", "low")
        shift = {
            ("home", "high"): 0.10,
            ("home", "medium"): 0.05,
            ("home", "low"): 0.02,
            ("away", "high"): -0.10,
            ("away", "medium"): -0.05,
            ("away", "low"): -0.02,
        }.get((direction, confidence), 0.0)
        # Solo emit probs en h2h. Para markets distintos (totals/spreads/btts)
        # devolvemos SignalProbs vacío (probs={}) → no contribuye al fusion
        # pero el reasoning chain igual se persiste en el reporte.
        if market != "h2h":
            sig = SignalProbs(
                name="llm_qualitative",
                probs={},
                weight=0.0,
                confidence=0.0,
                note=f"LLM dir={direction} conf={confidence} (no probs para market={market})",
            )
        else:
            if is_3way:
                base_home = max(0.10, min(0.85, 0.45 + shift))
                probs_dict = {
                    "home": base_home,
                    "draw": 0.27,
                    "away": max(0.05, 1.0 - base_home - 0.27),
                }
            else:
                base_home = max(0.15, min(0.85, 0.55 + shift))
                probs_dict = {"home": base_home, "away": 1.0 - base_home}
            sig = SignalProbs(
                name="llm_qualitative",
                probs=probs_dict,
                weight=0.6,
                confidence={"high": 0.55, "medium": 0.40, "low": 0.25}.get(confidence, 0.25),
                note=f"LLM dir={direction} conf={confidence}",
            )
        # B8 reasoning chain completa. Las llaves del dict vienen de
        # `llm_analyze_event` (deep_analysis.py:904-932): summary_es,
        # confidence, edge_direction, line_movement, home/away (cada uno con
        # key_injuries, rest_days, b2b, momentum, travel_km, altitude_delta_m),
        # contradictions_found.
        home_info = result.get("home") or {}
        away_info = result.get("away") or {}
        key_factors: list[str] = []
        risks: list[str] = []
        # Extraer factores narrativos de momentum + injuries
        for side, info in (("local", home_info), ("visitante", away_info)):
            momentum = info.get("momentum")
            if momentum:
                key_factors.append(f"Momentum {side}: {momentum}")
            for inj in info.get("key_injuries") or []:
                player = inj.get("player") or "?"
                sev = inj.get("severity") or "?"
                risks.append(f"Lesión {side}: {player} ({sev})")
            rest = info.get("rest_days")
            if rest is not None and rest <= 1:
                risks.append(f"Descanso corto {side}: {rest}d")
            if info.get("b2b"):
                risks.append(f"Back-to-back {side}")
        for c in result.get("contradictions_found") or []:
            risks.append(f"Contradicción: {c}")

        reasoning = {
            "edge_direction": direction,
            "confidence": confidence,
            "shift_applied": shift,
            "rag_snippets_chars": len(rag_snippets),
            "line_movement": result.get("line_movement"),
            "key_factors": key_factors,
            "risks": risks,
            "summary": result.get("summary_es") or "",
            "home_analysis": home_info,
            "away_analysis": away_info,
        }
        return sig, reasoning
    except Exception as exc:
        logger.debug("agent.llm.fail", error=str(exc)[:120])
        return None


# ─── Fusión bayesiana ────────────────────────────────────────────────────


_SHARP_ANCHOR_NAME = "pinnacle_devig"


def _fuse_signals(signals: list[SignalProbs]) -> dict[str, float]:
    """Promedio ponderado con shrinkage cuadrático cuando una señal diverge
    fuerte del consenso sharp (B4).

    Si Pinnacle de-vigged está disponible se usa como ancla; cualquier señal
    con |Δ p| > 0.08 en su outcome más divergente recibe shrinkage cuadrático
    sobre su weight efectivo (mismo criterio que `detector.py:631-640`):

      |Δ| ≤ 0.05      → weight intacto
      0.05 < |Δ| ≤ 0.08 → factor lineal max(0.1, 1 − (Δ−0.05)·4.5)
      |Δ| > 0.08      → factor cuadrático max(0.04, (1 − (Δ−0.05)·4.5)²)

    Si Pinnacle no está, se usa fallback al promedio de las otras señales
    como ancla pseudo-sharp (sin shrinkage cuando solo hay una señal).
    """
    if not signals:
        return {}
    # Filtrar signals "vacías" (placeholders sin probs, e.g. LLM en non-h2h
    # markets). Conservan reasoning chain pero no contribuyen al fusion.
    signals = [s for s in signals if s.probs and s.weight > 0 and s.confidence > 0]
    if not signals:
        return {}
    common = set(signals[0].probs.keys())
    for s in signals[1:]:
        common &= set(s.probs.keys())
    if not common:
        return {}

    anchor: dict[str, float] | None = None
    anchor_is_sharp = False
    for s in signals:
        if s.name == _SHARP_ANCHOR_NAME:
            anchor = {oc: s.probs.get(oc, 0.0) for oc in common}
            anchor_is_sharp = True
            break

    # Fallback pseudo-sharp: si no hay Pinnacle pero hay >=2 señales,
    # usamos el promedio simple (uniforme, sin pesos) como ancla. Aplica
    # shrinkage atenuado (escalas más permisivas: 0.10/0.15) porque el
    # ancla no es realmente sharp.
    pseudo_sharp = False
    if anchor is None and len(signals) >= 2:
        anchor = {oc: sum(s.probs.get(oc, 0.0) for s in signals) / len(signals) for oc in common}
        pseudo_sharp = True

    fused: dict[str, float] = dict.fromkeys(common, 0.0)
    total_w = 0.0
    for s in signals:
        w_eff = s.weight * s.confidence
        if w_eff <= 0:
            continue
        if anchor is not None and (not anchor_is_sharp or s.name != _SHARP_ANCHOR_NAME):
            # Máxima divergencia outcome-by-outcome contra ancla
            delta_max = max(abs(s.probs.get(oc, 0.0) - anchor[oc]) for oc in common)
            if anchor_is_sharp:
                hi, lo = 0.08, 0.05
            else:
                # Pseudo-sharp: tolerancia mayor (la "media" aún tiene varianza)
                hi, lo = 0.15, 0.10
            if delta_max > hi:
                shrink = max(0.04, (1.0 - (delta_max - lo) * 4.5) ** 2)
                w_eff *= shrink
            elif delta_max > lo:
                shrink = max(0.1, 1.0 - (delta_max - lo) * 4.5)
                w_eff *= shrink
        if w_eff <= 0:
            continue
        for oc in common:
            fused[oc] += s.probs.get(oc, 0.0) * w_eff
        total_w += w_eff
    _ = pseudo_sharp  # informativo, podría loggearse en analytics futuro
    if total_w <= 0:
        return {}
    for oc in common:
        fused[oc] /= total_w
    s_sum = sum(fused.values())
    if s_sum > 0:
        for oc in common:
            fused[oc] /= s_sum
    return fused


def _conformal_band(
    signals: list[SignalProbs], fused: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """Banda no-paramétrica via min/max ponderado entre señales (B4).

    Para cada outcome devuelve (low, high) usando ±1 desviación ponderada
    de las señales contribuyentes. No es Monte-Carlo PAC formal (no tenemos
    distribución posterior), pero da al usuario una idea de la dispersión
    cuando las señales están muy distribuidas vs convergentes.
    """
    out: dict[str, tuple[float, float]] = {}
    if not signals or not fused:
        return out
    common = set(fused.keys())
    for oc in common:
        # weighted mean (ya en fused) + weighted variance
        ws: list[tuple[float, float]] = []
        for s in signals:
            if oc not in s.probs:
                continue
            w = max(0.0, s.weight * s.confidence)
            if w <= 0:
                continue
            ws.append((w, s.probs[oc]))
        if len(ws) < 2:
            # Sin dispersión computable → banda mínima ±5pp
            p = fused[oc]
            out[oc] = (max(0.0, p - 0.05), min(1.0, p + 0.05))
            continue
        total = sum(w for w, _ in ws)
        mean = sum(w * p for w, p in ws) / total
        var = sum(w * (p - mean) ** 2 for w, p in ws) / total
        sigma = var**0.5
        # Ensanchar levemente para underestimation con n pequeño
        sigma_eff = max(0.02, sigma * (len(ws) / max(1, len(ws) - 1)) ** 0.5)
        out[oc] = (max(0.0, fused[oc] - sigma_eff), min(1.0, fused[oc] + sigma_eff))
    return out


# ─── Pick extraction ─────────────────────────────────────────────────────


def _extract_picks(
    fused: dict[str, float],
    odds: EventOdds | None,
    *,
    sport_code: str,
    league_id: int | None,
    signals_used: list[SignalProbs] | None = None,
    skip_reasons: dict[str, str] | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    league_name: str | None = None,
    stage: str | None = None,
    clv_hints: dict[str, float] | None = None,
) -> list[PickRecommendation]:
    """Para cada outcome con probs fusionadas, busca el book que mejor pague
    sobre el fair y emit pick si edge > threshold dinámico.

    Guarda anti-Pinnacle-only: si la única señal aportante es `pinnacle_devig`,
    el "fair" es literalmente Pinnacle fair → arbitrar el vig del soft book.
    Es el bug "MLB picks sin modelo" del 22-23 abr; el detector batch lo bloquea
    en `detector.py:642-680`. Aquí igualamos comportamiento.
    """
    if skip_reasons is None:
        skip_reasons = {}
    if not fused or odds is None:
        if not fused:
            skip_reasons["__all__"] = "no_fused_probs"
        else:
            skip_reasons["__all__"] = "no_odds_available"
        return []

    # Guarda anti-sharp-derivative-only: las señales independientes deben
    # aportar al menos UN modelo o anchor con info independiente. catchall_
    # baseline literalmente recalibra Pinnacle prior → es sharp-derivativa.
    # llm_qualitative es prior baseline + shift cualitativo, no modelo numérico.
    # Sin BayesianXG / sklearn LGBM / Dixon-Coles real / Polymarket líquido,
    # NO emit picks (escenario UCL hoy: catchall + LLM darían +50% EV irreal
    # porque ambos heredan el ruido de Pinnacle sin compensación independiente).
    if signals_used:
        SHARP_DERIVATIVE = {
            "pinnacle_devig",
            "polymarket",
            "production:catchall_baseline",
            "llm_qualitative",
        }
        # Solo considerar señales que aportaron probs (filtrar placeholders).
        active = [s for s in signals_used if s.probs and s.weight > 0 and s.confidence > 0]
        independent = [s for s in active if s.name not in SHARP_DERIVATIVE]
        if not independent:
            skip_reasons["__all__"] = "only_sharp_derivative_no_independent_model"
            return []

    # Threshold dinámico por sport/stage/market/league (B6 reusa ev_thresholds.yaml).
    try:
        from apuestas.betting.ev_thresholds import ev_threshold_for

        threshold = ev_threshold_for(
            sport=sport_code,
            stage=stage,
            market=getattr(odds, "market", None),
            league_id=league_id,
            fallback=float(os.environ.get("APUESTAS_AGENT_EV_THRESHOLD", "0.03")),
        )
    except Exception:
        threshold = float(os.environ.get("APUESTAS_AGENT_EV_THRESHOLD", "0.03"))
    # Sanity cap (B3): EV > 25% es casi siempre stale odds / data corrupta /
    # pricing error del book. No debe emit picks con EV irrealistas.
    ev_max = float(os.environ.get("APUESTAS_AGENT_EV_MAX", "0.25"))
    picks: list[PickRecommendation] = []
    SHARP_BOOKS = {"pinnacle", "polymarket", "smarkets", "betfair_ex_eu", "matchbook"}

    # B6: book_power_ratings — preferir books con edge histórico positivo.
    try:
        from apuestas.betting.book_power_ratings import get_cached_edge

        _book_edge_lookup = lambda bm: get_cached_edge(bm, league_name) if league_name else 0.0
    except Exception:
        _book_edge_lookup = lambda _bm: 0.0

    for outcome, p_fair in fused.items():
        if p_fair <= 0 or p_fair >= 1:
            skip_reasons[outcome] = f"prob_out_of_range:{p_fair:.3f}"
            continue
        try:
            idx = odds.outcomes.index(outcome)
        except ValueError:
            skip_reasons[outcome] = "outcome_not_in_market"
            continue
        best_book = ""
        best_odds = 0.0
        best_book_edge_bps = 0.0
        for bm, quotes in odds.quotes_by_bookmaker.items():
            if bm in SHARP_BOOKS:
                continue
            if idx >= len(quotes):
                continue
            o = quotes[idx]
            if o is None or o <= 1.0:
                continue
            # B6: priorizar por (odds, book_edge_bps). Books con edge positivo
            # histórico ganan en empates de odds; en odds desiguales ganan odds.
            book_edge = _book_edge_lookup(bm)
            tiebreak = book_edge / 1000.0  # bps→puntuación marginal sub-1
            score = float(o) + tiebreak
            cur_score = best_odds + best_book_edge_bps / 1000.0 if best_book else -1
            if score > cur_score:
                best_odds = float(o)
                best_book = bm
                best_book_edge_bps = book_edge
        if not best_book or best_odds == 0:
            skip_reasons[outcome] = "no_soft_book_offer"
            continue
        implied_book = 1.0 / best_odds
        edge = (p_fair / implied_book) - 1.0
        ev = (p_fair * best_odds) - 1.0
        if ev < threshold:
            skip_reasons[outcome] = f"ev_below_threshold:{ev * 100:+.2f}%<{threshold * 100:+.2f}%"
            continue
        # Si EV excede el cap, NO skipeamos: emitimos pick pero con
        # `stale_warning` honesto. Caso típico: Pinnacle se movió fuerte hoy
        # pero los soft books son del día anterior (budget Odds API agotado).
        # El usuario debe verificar en el book antes de apostar; el agente no
        # puede saber si el precio sigue vivo.
        stale_warn: str | None = None
        if ev > ev_max:
            stale_warn = (
                f"EV anormalmente alto ({ev * 100:+.2f}% > cap {ev_max * 100:.0f}%). "
                f"Probable stale soft book vs Pinnacle fresco. "
                f"Verifica en {best_odds:.2f}@{best_book} antes de apostar — "
                f"si el precio cambió a ≤ {1.0 / p_fair * (1 + threshold):.2f} ya no hay edge."
            )
        confidence_label = (
            "stale" if stale_warn else "high" if ev >= 0.08 else "medium" if ev >= 0.05 else "low"
        )

        # B4 conformal band para este outcome.
        band = bands.get(outcome) if bands else None
        p_low_v = float(band[0]) if band else None
        p_high_v = float(band[1]) if band else None

        # B6 anticipated_clv si se calculó previamente.
        clv_v = clv_hints.get(outcome) if clv_hints else None

        # B6 ¼ Kelly hint: f* = (p·o − 1) / (o − 1); ¼ del óptimo, cap 5%.
        kelly_full = (p_fair * best_odds - 1.0) / max(best_odds - 1.0, 0.001)
        kelly_quarter = max(0.0, min(0.05, 0.25 * kelly_full))

        picks.append(
            PickRecommendation(
                market=odds.market,
                outcome=outcome,
                line=odds.lines[idx] if odds.lines else None,
                book=best_book,
                odds=best_odds,
                p_fused=p_fair,
                edge=edge,
                ev=ev,
                confidence=confidence_label,
                reasoning=(
                    f"p_fused={p_fair:.1%} > p_book={implied_book:.1%} "
                    f"(edge {edge * 100:+.2f}%, EV {ev * 100:+.2f}%)"
                ),
                p_low=p_low_v,
                p_high=p_high_v,
                anticipated_clv=clv_v,
                book_edge_bps=float(best_book_edge_bps) if best_book_edge_bps else None,
                kelly_quarter_pct=float(kelly_quarter * 100.0),
                stale_warning=stale_warn,
            )
        )
    picks.sort(key=lambda p: p.ev, reverse=True)
    return picks[:3]  # top 3 por EV


# ─── Orchestrator principal ──────────────────────────────────────────────


async def _fetch_existing_picks(match_id: int) -> list[ExistingPickInfo]:
    """Lee picks ya emitidos por el detector batch para este match.

    Esto evita que el agente `/analizar` aparente "sin picks" cuando en
    realidad ya hay uno vivo. Incluye pendings, confirmed y resueltos
    recientes (últimas 72h) para dar contexto histórico.
    """
    async with session_scope() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id, market, outcome, line, bookmaker, odds_placed,
                           placed_at, status, outcome_result, p_consensus_sharp
                    FROM pick_alerts
                    WHERE match_id = :mid
                      AND placed_at >= NOW() - INTERVAL '7 days'
                    ORDER BY placed_at DESC
                    """
                ),
                {"mid": match_id},
            )
        ).all()
    out: list[ExistingPickInfo] = []
    for r in rows:
        out.append(
            ExistingPickInfo(
                pick_id=int(r.id),
                market=str(r.market),
                outcome=str(r.outcome),
                line=float(r.line) if r.line is not None else None,
                book=str(r.bookmaker or ""),
                odds_placed=float(r.odds_placed),
                placed_at=r.placed_at,
                status=str(r.status or "pending"),
                outcome_result=r.outcome_result,
                p_consensus_sharp=(
                    float(r.p_consensus_sharp) if r.p_consensus_sharp is not None else None
                ),
            )
        )
    return out


async def _max_odds_age_hours(match_id: int) -> float | None:
    """Devuelve cuántas horas tiene la odds más reciente del match. None si no hay."""
    async with session_scope() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT EXTRACT(EPOCH FROM (NOW() - MAX(ts))) / 3600.0 AS hours_old
                    FROM odds_history
                    WHERE match_id = :mid
                    """
                ),
                {"mid": match_id},
            )
        ).first()
    if row is None or row.hours_old is None:
        return None
    return float(row.hours_old)


async def _refresh_odds_on_demand(sport_code: str) -> dict[str, Any]:
    """Dispara catchup_pinnacle_guest + odds_api_optimized para 1 sport.

    Bajo costo (~5-10 cred Odds API). Solo se llama cuando las odds del match
    pedido tienen >2h. Falla silenciosa: si refresh falla, seguimos con lo que
    ya hay en DB.
    """
    refreshed: dict[str, Any] = {"pinnacle": False, "odds_api": False}
    try:
        from apuestas.flows.catchup import catchup_pinnacle_guest

        r = await catchup_pinnacle_guest.fn()
        refreshed["pinnacle"] = bool(r)
    except Exception as exc:
        logger.debug("agent.refresh_pinnacle_fail", error=str(exc)[:120])
    try:
        from apuestas.flows.catchup import catchup_odds_api

        r2 = await catchup_odds_api.fn()
        refreshed["odds_api"] = bool(r2)
    except Exception as exc:
        logger.debug("agent.refresh_odds_api_fail", error=str(exc)[:120])
    logger.info("agent.refresh_odds_on_demand", sport=sport_code, **refreshed)
    return refreshed


# B8: rate limit por user (en memoria, reset 1h rolling).
_RATE_LIMIT_WINDOW_S = 3600
_RATE_LIMIT_MAX = int(os.environ.get("APUESTAS_AGENT_RATE_LIMIT_PER_HOUR", "10"))
_RATE_LIMIT_TS: dict[str, list[float]] = {}


def _rate_limit_check(user_key: str) -> tuple[bool, int]:
    """True si el user puede hacer otra llamada. Devuelve (ok, remaining)."""
    import time as _t

    now = _t.time()
    bucket = _RATE_LIMIT_TS.setdefault(user_key, [])
    # Drop expirados
    cutoff = now - _RATE_LIMIT_WINDOW_S
    bucket[:] = [t for t in bucket if t > cutoff]
    if len(bucket) >= _RATE_LIMIT_MAX:
        return False, 0
    bucket.append(now)
    return True, _RATE_LIMIT_MAX - len(bucket)


# B8: agent_run_log JSONL local (sin migración Alembic). Cada análisis deja
# fila con timestamps, fused_probs, signals, picks, skip_reasons, llm_reasoning.
# Migrable a tabla DB cuando convenga (futuro).
_AGENT_LOG_PATH = "logs/agent_run_log.jsonl"


def _persist_agent_run_log(report: MatchAnalysisReport, *, user_key: str | None = None) -> None:
    import json as _json
    from pathlib import Path as _Path

    try:
        _Path(_AGENT_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        record = report.to_dict()
        record["user_key"] = user_key
        record["logged_at"] = datetime.now(tz=UTC).isoformat()
        with open(_AGENT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.debug("agent.log_persist_fail", error=str(exc)[:120])


async def _compute_anticipated_clv(
    match_id: int,
    sport_code: str,
    league_id: int | None,
    odds: EventOdds | None,
    fused: dict[str, float],
    market: str,
    start_time: Any,
) -> dict[str, float]:
    """Anticipated CLV por outcome via ClosingLinePredictor (B6).

    Necesita: odds actual, sharp_book_consensus (Pinnacle), hours_until_start.
    Retorna {outcome: anticipated_clv} para outcomes con datos suficientes.
    """
    out: dict[str, float] = {}
    if odds is None or not fused:
        return out
    try:
        from apuestas.betting.closing_line_predictor import (
            extract_features_for_match,
            load_fitted_predictor,
        )

        predictor = load_fitted_predictor(sport_code)
        # hours until start
        try:
            now = datetime.now(tz=UTC)
            hrs = max(0.1, (start_time - now).total_seconds() / 3600.0) if start_time else 24.0
        except Exception:
            hrs = 24.0
        # Pinnacle sharp consensus para cada outcome
        pinn = odds.quotes_by_bookmaker.get("pinnacle", [])
        async with session_scope() as session:
            for outcome, _p in fused.items():
                try:
                    idx = odds.outcomes.index(outcome)
                except ValueError:
                    continue
                if idx >= len(pinn):
                    continue
                pinn_odds = pinn[idx]
                if pinn_odds is None or pinn_odds <= 1.0:
                    continue
                # Soft book best ofreciendo este outcome
                best = 0.0
                for bm, qq in odds.quotes_by_bookmaker.items():
                    if bm in {"pinnacle", "polymarket", "smarkets", "betfair_ex_eu", "matchbook"}:
                        continue
                    if idx < len(qq) and qq[idx] and qq[idx] > best:
                        best = float(qq[idx])
                if best <= 1.0:
                    continue
                feats = await extract_features_for_match(
                    session=session,
                    match_id=match_id,
                    market=market,
                    outcome=outcome,
                    current_odds=best,
                    sharp_book_consensus=float(pinn_odds),
                    hours_until_start=hrs,
                    sport_code=sport_code,
                    league_id=league_id,
                )
                out[outcome] = float(predictor.anticipated_clv(feats))
    except Exception as exc:
        logger.debug("agent.clv_compute_fail", error=str(exc)[:120])
    return out


async def analyze_single_match(
    query: str | int,
    *,
    market: str = "h2h",
    user_key: str | None = None,
) -> MatchAnalysisReport | None:
    """Pipeline completo on-demand. Retorna None si no resuelve match.

    `market`: h2h | totals | spreads | runline | btts (B5 cobertura mercados).
    `user_key`: identificador para rate limit (B8). None desactiva limit.
    """
    # B8: rate limit
    if user_key is not None:
        ok, remaining = _rate_limit_check(user_key)
        if not ok:
            logger.warning("agent.rate_limit_exceeded", user_key=user_key)
            return None

    started = datetime.now(tz=UTC)
    event = await resolve_match(query)
    if event is None:
        return None

    sport_code = str(event.get("sport_code") or "")
    league_id = event.get("league_id")
    league_name = event.get("league_name")
    stage = event.get("stage")
    match_id = int(event["id"])

    from apuestas.flows.deep_analysis import collect_odds_for_event

    existing_picks = await _fetch_existing_picks(match_id)

    odds_freshness_warning: str | None = None
    age_h = await _max_odds_age_hours(match_id)
    if age_h is None or age_h > 2.0:
        await _refresh_odds_on_demand(sport_code)
        age_h = await _max_odds_age_hours(match_id)
    if age_h is not None and age_h > 6.0:
        odds_freshness_warning = (
            f"Las odds más recientes para este partido tienen {age_h:.1f}h de antigüedad. "
            "Verifica el precio actual en el book antes de apostar."
        )

    # Ventana adaptativa: si las odds más recientes son viejas (budget Odds API
    # agotado, partido en liga sin refresh reciente), abrimos la ventana hasta
    # cubrir esas odds. Antes era fija 48h y devolvía None aunque hubiera odds
    # de 76h → reporte "1/6 señales" sin Pinnacle. Ahora: ventana = max(48,
    # ceil(age_h)+2) hasta tope 168h (7 días). Compensamos la pérdida de
    # frescura mostrando warning al usuario (ya hace).
    freshness = 48
    if age_h is not None and age_h > freshness:
        freshness = min(168, int(age_h) + 2)
    odds = await collect_odds_for_event.fn(match_id, freshness_hours=freshness)

    # B5: si market != h2h, buscar en EventOdds.additional_markets (lista).
    # collect_odds_for_event guarda totals/spreads/team_totals/etc en
    # `additional_markets: list[EventOdds]` (`detector.py:84`). Para que el
    # agente analice esos markets debe encontrar el EventOdds correcto en la
    # lista por nombre de market.
    if odds is not None and market != "h2h" and getattr(odds, "market", None) != market:
        extras = list(getattr(odds, "additional_markets", []) or [])
        match_extra = next((e for e in extras if getattr(e, "market", None) == market), None)
        if match_extra is not None:
            odds = match_extra
        else:
            available = [getattr(odds, "market", "?")] + [getattr(e, "market", "?") for e in extras]
            logger.debug(
                "agent.market_unavailable",
                market=market,
                match_id=match_id,
                available=available,
            )
            odds = None

    # Línea totals/spreads: tomar la primera no-None del odds (todos los
    # outcomes de un mismo market suelen compartir la misma línea para soccer/NBA).
    dc_line: float | None = None
    if odds is not None and market in ("totals", "spreads", "runline"):
        lines_seq = getattr(odds, "lines", None) or []
        for ln in lines_seq:
            if ln is not None:
                dc_line = float(ln)
                break

    signals_raw = await asyncio.gather(
        _signal_production_model(sport_code, market, league_id, event, odds),
        _signal_dixon_coles(event, market, line=dc_line),
        _signal_pinnacle_devigged(odds, match_id),
        _signal_polymarket(odds, match_id),
        _signal_llm_qualitative(event, market),
        _signal_statsbomb_form(event) if market == "h2h" else _noop_signal(),
        return_exceptions=True,
    )
    used: list[SignalProbs] = []
    skipped: list[str] = []
    llm_reasoning: dict[str, Any] | None = None
    expected_names = [
        "production_model",
        "dixon_coles",
        "pinnacle_devig",
        "polymarket",
        "llm_qualitative",
        "statsbomb_form",
    ]
    for name, s in zip(expected_names, signals_raw, strict=True):
        if isinstance(s, Exception):
            skipped.append(f"{name}:exception")
            continue
        if s is None:
            skipped.append(f"{name}:not_available")
            continue
        # llm_qualitative ahora devuelve tupla (SignalProbs, reasoning_dict)
        if name == "llm_qualitative" and isinstance(s, tuple):
            sig_obj, llm_reasoning = s
            used.append(sig_obj)
        else:
            used.append(s)  # type: ignore[arg-type]

    # B4: shrinkage + conformal band
    fused = _fuse_signals(used)
    bands = _conformal_band(used, fused)

    # B6: anticipated CLV por outcome
    clv_hints = await _compute_anticipated_clv(
        match_id,
        sport_code,
        league_id,
        odds,
        fused,
        market,
        event.get("start_time"),
    )

    skip_reasons: dict[str, str] = {}
    picks = _extract_picks(
        fused,
        odds,
        sport_code=sport_code,
        league_id=league_id,
        signals_used=used,
        skip_reasons=skip_reasons,
        bands=bands,
        league_name=league_name,
        stage=stage,
        clv_hints=clv_hints,
    )

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    summary_lines: list[str] = []
    if fused:
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        most_likely_oc, most_likely_p = ordered[0]
        band = bands.get(most_likely_oc)
        if band is not None:
            summary_lines.append(
                f"Resultado más probable: {most_likely_oc} ({most_likely_p * 100:.1f}% "
                f"[{band[0] * 100:.1f}-{band[1] * 100:.1f}%])"
            )
        else:
            summary_lines.append(
                f"Resultado más probable: {most_likely_oc} ({most_likely_p * 100:.1f}%)"
            )
    if picks:
        summary_lines.append(f"Picks recomendados: {len(picks)}")
    else:
        global_reason = skip_reasons.get("__all__")
        if global_reason == "only_sharp_derivative_no_independent_model":
            summary_lines.append(
                "Sin picks: solo señales sharp-derivativas (Pinnacle / Catchall / "
                "Polymarket / LLM prior). Sin modelo independiente (Bayesian xG, "
                "Dixon-Coles real, sklearn) cualquier 'edge' que aparezca es ruido "
                "del propio Pinnacle. Guarda anti-pattern activa."
            )
        elif global_reason == "no_odds_available":
            summary_lines.append(
                f"Sin picks: no hay odds disponibles para este partido (market={market}) "
                f"en la ventana de {freshness}h."
            )
        elif global_reason == "no_fused_probs":
            summary_lines.append("Sin picks: todas las señales fallaron. No se pudo computar fair.")
        else:
            summary_lines.append("Sin picks de valor: mercado eficiente o señales insuficientes.")
    summary_lines.append(
        f"Señales usadas: {len(used)}/{len(expected_names)} | duración: {elapsed:.1f}s"
    )
    if odds_freshness_warning:
        summary_lines.append(f"⚠ {odds_freshness_warning}")

    report = MatchAnalysisReport(
        match_id=match_id,
        sport_code=sport_code,
        home_name=str(event.get("home_name") or ""),
        away_name=str(event.get("away_name") or ""),
        league_name=league_name,
        start_time=event.get("start_time") or started,
        market=market,
        signals_used=used,
        fused_probs=fused,
        fused_bands=bands,
        picks=picks,
        skipped_signals=skipped,
        skip_reasons=skip_reasons,
        odds_freshness_warning=odds_freshness_warning,
        existing_picks=existing_picks,
        llm_reasoning=llm_reasoning,
        ambiguous_candidates=event.get("_ambiguous_candidates") or [],
        duration_s=elapsed,
        summary_es="\n".join(summary_lines),
    )
    # B8: persist run log
    _persist_agent_run_log(report, user_key=user_key)
    return report


__all__ = [
    "MatchAnalysisReport",
    "PickRecommendation",
    "SignalProbs",
    "analyze_single_match",
    "resolve_match",
]
