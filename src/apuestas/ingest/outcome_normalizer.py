"""Normalizador de outcomes de odds externas a schema interno.

APIs externas devuelven outcomes como nombres de equipos:
    The Odds API → "Crystal Palace", "Draw", "West Ham United"
    football-data.org → "HOME_TEAM", "DRAW", "AWAY_TEAM"
    API-Football → "Home", "Draw", "Away"

Nuestro schema interno de `predictions.outcome` y `bets.outcome` usa:
    "home" | "away" | "draw"

Este módulo mapea según el contexto del match (home_team_name, away_team_name).
Indispensable antes de persistir en `odds_history` u operar settlement.
"""

from __future__ import annotations

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


DRAW_ALIASES = frozenset({"draw", "tie", "empate", "x", "d", "the draw", "drawn"})


def normalize_h2h_outcome(
    *,
    raw_outcome: str,
    home_team_name: str,
    away_team_name: str,
) -> str | None:
    """Convierte outcome raw a 'home'|'away'|'draw'. Retorna None si no matchea.

    Estrategia:
    1. Lowercase normalize ambos lados.
    2. Si está en DRAW_ALIASES → 'draw'.
    3. Si matchea (substring bidireccional) home_team_name → 'home'.
    4. Si matchea away_team_name → 'away'.
    5. Casos especiales: 'HOME_TEAM'/'AWAY_TEAM'/'HOME'/'AWAY' directos.
    """
    if not raw_outcome:
        return None

    raw_lower = raw_outcome.strip().lower()
    home_lower = (home_team_name or "").strip().lower()
    away_lower = (away_team_name or "").strip().lower()

    # Casos especiales API-Football + football-data.org
    if raw_lower in {"home", "home_team", "home team"}:
        return "home"
    if raw_lower in {"away", "away_team", "away team"}:
        return "away"
    if raw_lower in DRAW_ALIASES:
        return "draw"

    # Matching fuzzy por substring (orden: más específico primero)
    def _fuzzy_match(needle: str, haystack: str) -> bool:
        if not needle or not haystack:
            return False
        if needle == haystack:
            return True
        # Eliminar sufijos comunes "FC", "AFC", "United FC", etc.
        suffixes = (" fc", " afc", " cf", " club", " deportivo")
        for suf in suffixes:
            if needle.endswith(suf):
                needle = needle[: -len(suf)]
            if haystack.endswith(suf):
                haystack = haystack[: -len(suf)]
        return needle in haystack or haystack in needle

    if _fuzzy_match(raw_lower, home_lower):
        return "home"
    if _fuzzy_match(raw_lower, away_lower):
        return "away"

    logger.debug(
        "outcome_normalizer.unmatched",
        raw=raw_outcome,
        home=home_team_name,
        away=away_team_name,
    )
    return None


def normalize_totals_outcome(*, raw_outcome: str) -> str | None:
    """O/U totals: 'Over'/'Under' → 'over'/'under'."""
    if not raw_outcome:
        return None
    low = raw_outcome.strip().lower()
    if low.startswith("over") or low == "o":
        return "over"
    if low.startswith("under") or low == "u":
        return "under"
    return None


def normalize_spread_outcome(
    *,
    raw_outcome: str,
    home_team_name: str,
    away_team_name: str,
) -> str | None:
    """Spread handicap: igual que h2h (home/away)."""
    return normalize_h2h_outcome(
        raw_outcome=raw_outcome,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
    )


def normalize_btts_outcome(*, raw_outcome: str) -> str | None:
    """BTTS: 'Yes'/'No'."""
    if not raw_outcome:
        return None
    low = raw_outcome.strip().lower()
    if low in {"yes", "y", "gg", "btts_yes"}:
        return "yes"
    if low in {"no", "n", "ng", "btts_no"}:
        return "no"
    return None


_MARKET_NORMALIZERS = {
    "h2h": normalize_h2h_outcome,
    "moneyline": normalize_h2h_outcome,
    "1x2": normalize_h2h_outcome,
    "totals": normalize_totals_outcome,
    "total": normalize_totals_outcome,
    "ou": normalize_totals_outcome,
    "spreads": normalize_spread_outcome,
    "spread": normalize_spread_outcome,
    "runline": normalize_spread_outcome,
    "puckline": normalize_spread_outcome,
    "btts": normalize_btts_outcome,
}


def normalize_outcome(
    *,
    market: str,
    raw_outcome: str,
    home_team_name: str = "",
    away_team_name: str = "",
) -> str | None:
    """Dispatcher: normaliza outcome según el mercado."""
    normalizer = _MARKET_NORMALIZERS.get(market.lower())
    if normalizer is None:
        return None

    # h2h/spreads necesitan nombres de equipos; otros no.
    if normalizer in (normalize_h2h_outcome, normalize_spread_outcome):
        return normalizer(
            raw_outcome=raw_outcome,
            home_team_name=home_team_name,
            away_team_name=away_team_name,
        )
    return normalizer(raw_outcome=raw_outcome)
