"""Correlation filter para picks del mismo evento — Sprint 10 Fase 1 (Mejora #3).

Lógica: `h2h/away` y `spreads/away +1.5` del mismo match son señales
~85% correlacionadas (Koopman & Lit 2015, bivariate Poisson). Emitir
ambos duplica exposición al mismo riesgo subyacente.

Política: por (event_id, outcome_side), conservar SOLO el pick con mayor
edge entre `h2h`, `spreads` y `totals`/`team_totals`. Se emite uno.

Casos del dataset 22-23 abr que elimina:
- #33 TEX-PIT h2h/home (lost) + #40 TEX-PIT spreads/home +1.5 (lost):
  redundantes. Por edge mayor se emitiría solo 1, reduciendo exposición.
- #29 BOS-NYY h2h/away (won) + #37 BOS-NYY spreads/home +1.5 (lost):
  eran OPUESTOS (bot se apostaba contra sí mismo en el mismo match).
  El filter los detecta como contradictorios — el que baja edge se mata.
"""

from __future__ import annotations

from dataclasses import dataclass

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


_SPREAD_MARKETS = frozenset({"spreads", "handicap", "ah", "runline", "puckline"})
_H2H_MARKETS = frozenset({"h2h", "moneyline", "ml"})
_TOTAL_MARKETS = frozenset({"totals", "total", "over_under"})


def _market_family(market: str) -> str:
    m = market.lower()
    if m in _SPREAD_MARKETS:
        return "spread"
    if m in _H2H_MARKETS:
        return "h2h"
    if m in _TOTAL_MARKETS:
        return "total"
    return m  # props u otros mercados: no agrupar


def _pick_side(outcome: str) -> str:
    """Devuelve 'home'|'away'|'draw'|'over'|'under' normalizado."""
    o = outcome.lower()
    if o in ("home", "1"):
        return "home"
    if o in ("away", "2"):
        return "away"
    if o == "draw":
        return "draw"
    if o == "over":
        return "over"
    if o == "under":
        return "under"
    return o


@dataclass(slots=True, frozen=True)
class _Key:
    event_id: int
    family: str  # "h2h"|"spread"|"total"
    side: str  # "home"|"away"|"draw"|"over"|"under"


def filter_correlated_picks(picks: list) -> tuple[list, list[dict]]:
    """Elimina picks correlacionados del mismo evento.

    Reglas:
    1. Para `(event_id, side)` donde side ∈ {home, away}:
       - Si hay h2h Y spread del mismo side → conservar el de mayor edge.
       - Si hay picks OPUESTOS (home h2h + away h2h) del mismo match,
         conservar el de mayor edge (evita bot apostando contra sí mismo).
    2. Para totals (`over` vs `under`): si ambos aparecen → contradictorio,
       conservar solo el de mayor edge.
    3. Props u otros mercados: no tocar (no correlacionados por construcción).

    Args:
        picks: lista de `ValueBet` o dicts con `event_id`, `market`,
               `outcome`, `edge` como keys.

    Returns:
        (kept, dropped_info) donde `dropped_info` es lista de dicts
        `{pick_id, reason, kept_instead_of}` para logging/audit.
    """
    if not picks:
        return [], []

    # Agrupar por (event_id, family)
    from collections import defaultdict

    buckets: dict[tuple[int, str], list] = defaultdict(list)
    dropped: list[dict] = []
    untouched: list = []

    for p in picks:
        event_id = _get(p, "event_id")
        market = _get(p, "market")
        if event_id is None or not market:
            untouched.append(p)
            continue
        family = _market_family(market)
        if family not in ("h2h", "spread", "total"):
            # props y exóticos pasan directo
            untouched.append(p)
            continue
        buckets[(event_id, family)].append(p)

    kept: list = list(untouched)

    # Paso 1: dentro de cada family del mismo evento, resolver contradictorios
    per_event: dict[int, list] = defaultdict(list)
    for (event_id, _family), group in buckets.items():
        if len(group) <= 1:
            per_event[event_id].extend(group)
            continue
        # Varios picks del mismo (event, family): pueden ser lados opuestos.
        # Conservar el de mayor edge.
        best = max(group, key=lambda p: _get(p, "edge") or 0.0)
        for p in group:
            if p is not best:
                dropped.append(
                    {
                        "event_id": event_id,
                        "market": _get(p, "market"),
                        "outcome": _get(p, "outcome"),
                        "edge": _get(p, "edge"),
                        "reason": "same_family_lower_edge",
                        "kept_instead_of": {
                            "market": _get(best, "market"),
                            "outcome": _get(best, "outcome"),
                            "edge": _get(best, "edge"),
                        },
                    }
                )
        per_event[event_id].append(best)

    # Paso 2: entre families distintas del mismo evento+side (h2h/home vs spread/home),
    # conservar solo 1.
    for event_id, group in per_event.items():
        by_side: dict[str, list] = defaultdict(list)
        for p in group:
            side = _pick_side(_get(p, "outcome") or "")
            by_side[side].append(p)
        for side, side_picks in by_side.items():
            if len(side_picks) == 1:
                kept.append(side_picks[0])
                continue
            best = max(side_picks, key=lambda p: _get(p, "edge") or 0.0)
            kept.append(best)
            for p in side_picks:
                if p is not best:
                    dropped.append(
                        {
                            "event_id": event_id,
                            "market": _get(p, "market"),
                            "outcome": _get(p, "outcome"),
                            "edge": _get(p, "edge"),
                            "reason": "correlated_cross_family",
                            "kept_instead_of": {
                                "market": _get(best, "market"),
                                "outcome": _get(best, "outcome"),
                                "edge": _get(best, "edge"),
                            },
                        }
                    )

    if dropped:
        logger.info(
            "correlation_filter.applied",
            kept=len(kept),
            dropped=len(dropped),
            events=len({d["event_id"] for d in dropped}),
        )
    return kept, dropped


def _get(pick, attr: str):  # type: ignore[no-untyped-def]
    """Acceso universal: dataclass o dict."""
    if isinstance(pick, dict):
        return pick.get(attr)
    return getattr(pick, attr, None)


__all__ = ["filter_correlated_picks"]
