"""Sport taxonomy central — Sprint 13 Capa 5.

Source of truth para `sport_code_canonical` + flags modelables + aliases
comunes por fuentes externas. Elimina la fragmentación (laliga/epl/soccer
como sport_code distinto pese a ser mismo deporte).

Reglas:
- `canonical`: única clave que usa el bot para lookup de modelos
- `aliases`: variantes que las fuentes externas pueden enviar
- `modelable`: si el bot debe emitir picks para este sport
- `markets`: markets soportados por deporte

Uso:
    from apuestas.sports import canonical_sport_code, is_modelable

    code = canonical_sport_code('laliga')  # → 'soccer'
    if is_modelable(code) and code not in ('esports',):
        # proceder con análisis
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SportDefinition:
    canonical: str
    aliases: frozenset[str]
    modelable: bool = True
    markets: frozenset[str] = field(default_factory=lambda: frozenset({"h2h"}))
    description: str = ""


SPORT_TAXONOMY: dict[str, SportDefinition] = {
    "soccer": SportDefinition(
        canonical="soccer",
        aliases=frozenset(
            {
                "soccer",
                "football",
                "futbol",
                "fútbol",
                "laliga",
                "epl",
                "premier_league",
                "liga_mx",
                "mls",
                "bundesliga",
                "seriea",
                "serie_a",
                "serieb",
                "serie_b",
                "ligue1",
                "ligue_1",
                "ligue2",
                "ligue_2",
                "eredivisie",
                "liga_portugal",
                "liga_expansion",
                "championship",
                "scottish_premier",
                "turkey_super_lig",
                "greek_super_league",
                "soccer_ucl",
                "soccer_uel",
                "soccer_fa_cup",
                "soccer_copa_libertadores",
                "soccer_copa_sudamericana",
                "soccer_brasileirao",
                "soccer_argentina",
                "soccer_chile",
                "soccer_saudi",
                "soccer_j_league",
                "soccer_k_league",
                "soccer_australia_aleague",
                "soccer_china_superleague",
                "soccer_ufa_conference",
            }
        ),
        modelable=True,
        markets=frozenset({"h2h", "spreads", "totals", "btts", "asian_handicap"}),
        description="Soccer/Football global (todas las ligas)",
    ),
    "nba": SportDefinition(
        canonical="nba",
        aliases=frozenset({"nba", "basketball", "basketball_nba"}),
        modelable=True,
        markets=frozenset({"h2h", "spreads", "totals"}),
    ),
    "mlb": SportDefinition(
        canonical="mlb",
        aliases=frozenset({"mlb", "baseball", "baseball_mlb"}),
        modelable=True,
        markets=frozenset({"h2h", "spreads", "totals", "runline"}),
    ),
    "nfl": SportDefinition(
        canonical="nfl",
        aliases=frozenset({"nfl", "american-football", "american_football"}),
        modelable=True,
        markets=frozenset({"h2h", "spreads", "totals", "ats"}),
    ),
    "nhl": SportDefinition(
        canonical="nhl",
        aliases=frozenset({"nhl", "ice-hockey", "hockey", "ice_hockey"}),
        modelable=True,
        markets=frozenset({"h2h", "spreads", "totals", "puckline"}),
    ),
    "tennis": SportDefinition(
        canonical="tennis",
        aliases=frozenset({"tennis", "atp", "wta", "tennis_atp", "tennis_wta"}),
        modelable=True,
        markets=frozenset({"h2h", "match_winner", "set_betting"}),
    ),
    "mma": SportDefinition(
        canonical="mma",
        aliases=frozenset({"mma", "ufc", "mma_mixed_martial_arts"}),
        modelable=True,
        markets=frozenset({"h2h", "method", "rounds"}),
    ),
    "boxing": SportDefinition(
        canonical="boxing",
        aliases=frozenset({"boxing", "boxing_boxing"}),
        modelable=True,
        markets=frozenset({"h2h", "method", "rounds"}),
    ),
    "esports": SportDefinition(
        canonical="esports",
        aliases=frozenset(
            {
                "esports",
                "lol",
                "dota2",
                "cs2",
                "csgo",
                "valorant",
                "efootball",
                "e-football",
                "fifa_esports",
                "nba_2k",
                "madden_esports",
            }
        ),
        modelable=False,  # NO emitir picks
        description="Esports — bot NO emite picks",
    ),
}


# Lookup rápido: alias → canonical
_ALIAS_INDEX: dict[str, str] = {}
for canonical, sdef in SPORT_TAXONOMY.items():
    for alias in sdef.aliases:
        _ALIAS_INDEX[alias.lower()] = canonical


def canonical_sport_code(raw: str | None) -> str:
    """Normaliza cualquier sport_code a su canonical.

    Returns 'unknown' si no matchea nada.
    """
    if not raw:
        return "unknown"
    key = str(raw).lower().strip()
    return _ALIAS_INDEX.get(key, key)


def is_modelable(sport_code: str | None) -> bool:
    """True si el bot debe emitir picks para este sport."""
    if not sport_code:
        return False
    canonical = canonical_sport_code(sport_code)
    sdef = SPORT_TAXONOMY.get(canonical)
    return sdef is not None and sdef.modelable


def supported_markets(sport_code: str | None) -> frozenset[str]:
    """Markets soportados para un sport."""
    if not sport_code:
        return frozenset()
    canonical = canonical_sport_code(sport_code)
    sdef = SPORT_TAXONOMY.get(canonical)
    return sdef.markets if sdef else frozenset()


def is_esports_team(team_name: str | None) -> bool:
    """Detecta team names de esports (cyber, virtual, efootball, etc.)."""
    if not team_name:
        return False
    n = team_name.lower()
    esports_markers = (
        "cyber",
        "esports",
        "(cs)",
        "virtual",
        "e-football",
        "efootball",
        "fifa ",
        "nba 2k",
        "madden ",
        "(e)",
        "(sim)",
    )
    return any(marker in n for marker in esports_markers)


__all__ = [
    "SPORT_TAXONOMY",
    "SportDefinition",
    "canonical_sport_code",
    "is_esports_team",
    "is_modelable",
    "supported_markets",
]
