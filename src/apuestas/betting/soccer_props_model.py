"""Modelo Poisson para props de soccer — goles/corners/shots/fouls/cards.

Dixon-Coles + team rolling averages. Input: stats promedio últimos 10 partidos
del equipo. Output: distribución Poisson de la prop.

Fórmula:
    λ_team = avg_stat_team × (avg_stat_opponent_allowed / league_avg)
    P(X ≥ line) = 1 - CDF_poisson(line, λ_team)

Ejemplo corners Chelsea vs Arsenal:
    Chelsea avg corners/game = 5.2 (últimos 10)
    Arsenal allowed corners/game = 4.8
    League avg = 5.0
    λ = 5.2 × (4.8 / 5.0) = 4.99
    P(Chelsea corners ≥ 5.5) = 1 - Poisson.cdf(5, 4.99) ≈ 41%

Librería: scipy.stats.poisson (ya instalada).
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import poisson  # type: ignore[import-untyped]

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# League averages típicas soccer (basadas en EPL/LaLiga 2023-2024)
LEAGUE_AVERAGES = {
    "goals_per_team": 1.4,
    "corners_per_team": 5.0,
    "shots_per_team": 12.0,
    "shots_on_target_per_team": 4.5,
    "fouls_per_team": 10.5,
    "yellow_cards_per_team": 1.8,
}


@dataclass(slots=True, frozen=True)
class PropProbability:
    prop_name: str  # corners_for, shots_for, fouls_for, etc.
    team: str  # home | away
    lambda_rate: float  # λ Poisson
    over_prob_for_line: dict[float, float]  # {line: P(X > line)}


def compute_prop_lambda(
    team_stat_avg: float,
    opponent_allowed_avg: float,
    league_avg: float,
) -> float:
    """λ Poisson ajustado por oponente.

    Si el team promedia 6 corners y el opponent permite 4 (vs liga 5),
    λ = 6 × (4/5) = 4.8.
    """
    if league_avg <= 0:
        return team_stat_avg
    opponent_factor = opponent_allowed_avg / league_avg
    return team_stat_avg * opponent_factor


def compute_prop_distribution(lam: float, lines: list[float]) -> dict[float, float]:
    """P(X > line) para varias líneas. Usa 0.5 (e.g. 5.5 → ≥ 6)."""
    result = {}
    for line in lines:
        # Over 5.5 significa X ≥ 6. sf es 1 - cdf(floor(line)).
        threshold = int(line)  # truncar para .5
        prob_over = float(poisson.sf(threshold, lam))
        result[line] = prob_over
    return result


async def compute_soccer_props(
    home_stats: dict[str, float],
    away_stats: dict[str, float],
    *,
    prop_types: tuple[str, ...] = ("corners", "shots", "shots_on_target", "fouls"),
    lines: list[float] | None = None,
    sport_code: str = "soccer",
    league_id: int = 0,
) -> list[PropProbability]:
    """Computa props para ambos equipos.

    Zero-hardcoded: lee league averages desde tabla `league_stats_averages`
    (poblada por `scripts/compute_league_averages.py`). Fallback al dict
    LEAGUE_AVERAGES si no hay data.

    Args:
        home_stats / away_stats: output de compute_team_rolling_stats()
            con keys 'corners', 'shots_for', 'shots_on_target_for', 'fouls', etc.
        prop_types: qué props calcular.
        lines: líneas a evaluar (default 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5).
        sport_code: para leer league_averages específico (soccer/epl/laliga/etc.)
        league_id: opcional, para averages por liga específica (0 = sport-wide).
    """
    from apuestas.scripts.compute_league_averages import get_league_average

    if lines is None:
        lines = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]

    results: list[PropProbability] = []
    mapping = {
        "corners": ("corners", "corners_per_team"),
        "shots": ("shots_for", "shots_per_team"),
        "shots_on_target": ("shots_on_target_for", "shots_on_target_per_team"),
        "fouls": ("fouls", "fouls_per_team"),
        "yellow_cards": ("yellow_cards", "yellow_cards_per_team"),
    }

    for prop in prop_types:
        stat_key, lg_key = mapping.get(prop, (prop, f"{prop}_per_team"))
        # Zero-hardcoded: leer desde DB, fallback a constante si no hay data
        lg_avg_fallback = LEAGUE_AVERAGES.get(lg_key, 5.0)
        lg_avg = await get_league_average(
            sport_code, lg_key, league_id=league_id, fallback=lg_avg_fallback
        )

        # Home team prop
        home_avg = home_stats.get(stat_key, 0.0)
        away_allowed = away_stats.get(
            stat_key, lg_avg
        )  # proxy: lo que genera el rival es lo que permite
        lam_home = compute_prop_lambda(home_avg, away_allowed, lg_avg)
        if lam_home > 0:
            results.append(
                PropProbability(
                    prop_name=prop,
                    team="home",
                    lambda_rate=lam_home,
                    over_prob_for_line=compute_prop_distribution(lam_home, lines),
                )
            )

        # Away team prop
        away_avg = away_stats.get(stat_key, 0.0)
        home_allowed = home_stats.get(stat_key, lg_avg)
        lam_away = compute_prop_lambda(away_avg, home_allowed, lg_avg)
        if lam_away > 0:
            results.append(
                PropProbability(
                    prop_name=prop,
                    team="away",
                    lambda_rate=lam_away,
                    over_prob_for_line=compute_prop_distribution(lam_away, lines),
                )
            )

    return results


def compute_match_total_prop(
    home_lambda: float, away_lambda: float, lines: list[float]
) -> dict[float, float]:
    """P(total > line) donde total = home + away (Poisson sum).

    Suma de Poissons independientes es Poisson con λ = λ_home + λ_away.
    """
    lam_total = home_lambda + away_lambda
    return compute_prop_distribution(lam_total, lines)


if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        # Test: Chelsea vs Arsenal con stats ficticias
        chelsea = {"corners": 5.2, "shots_for": 13.0, "fouls": 10.0, "shots_on_target_for": 4.8}
        arsenal = {"corners": 4.5, "shots_for": 11.5, "fouls": 11.2, "shots_on_target_for": 4.2}
        props = await compute_soccer_props(chelsea, arsenal)
        for p in props[:6]:
            print(f"{p.team} {p.prop_name}: λ={p.lambda_rate:.2f}")
            for line, prob in sorted(p.over_prob_for_line.items())[:4]:
                print(f"  P(>{line}) = {prob:.3f}")

    asyncio.run(_test())
