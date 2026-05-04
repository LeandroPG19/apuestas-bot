"""Motor de mercados player props + integración regional MX/US (§23.5 + §22).

Dado PropPrediction + quotes de libros, produce ValueBet con EV/Kelly
conformal-filtered, usando line shopping y cross-region recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apuestas.betting.ev import (
    BookmakerQuote,
    line_shopping,
)
from apuestas.betting.regional import RegionalRecommendation, compare_regions
from apuestas.obs.logging import get_logger
from apuestas.schemas.props import PropPrediction

logger = get_logger(__name__)


@dataclass(slots=True)
class PropLineQuote:
    bookmaker: str
    line: float
    over_odds: float | None
    under_odds: float | None


@dataclass(slots=True)
class PropValueBet:
    """Alerta de valor para un player prop.

    Post-pivote 2026-04-23: sin stake/kelly (el bot ya no dimensiona stake).
    """

    prediction: PropPrediction
    side: str  # 'over' | 'under'
    bookmaker: str
    odds: float
    line: float
    edge: float
    ev: float
    regional: RegionalRecommendation | None = None
    conformal_passed: bool = True
    skip_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


def _select_side_probability(
    prediction: PropPrediction,
    *,
    side: str,
    line: float,
) -> tuple[float, float | None]:
    """Retorna (p_point, p_lower) para el side dado."""
    if side == "over":
        p_point = prediction.p_over if prediction.p_over is not None else 0.5
        p_low = prediction.p_over_lower
    else:
        p_point = prediction.p_under if prediction.p_under is not None else 0.5
        p_low = 1.0 - prediction.p_over_upper if prediction.p_over_upper is not None else None
    return p_point, p_low


def evaluate_prop_side(
    prediction: PropPrediction,
    *,
    side: str,
    line: float,
    quotes: list[BookmakerQuote],
    include_regional: bool = True,
    conformal_margin: float = 0.01,
) -> PropValueBet | None:
    """Pipeline end-to-end para UN side (over/under) de un prop."""
    p_point, p_low = _select_side_probability(prediction, side=side, line=line)

    # Conformal filter
    implied_if_fair = 1 / min(q.odds for q in quotes) if quotes else 1.0
    conformal_ok = True
    if p_low is not None and p_low <= implied_if_fair + conformal_margin:
        conformal_ok = False

    offer = line_shopping(quotes, p_fair=p_point, exclude_sharp=True)
    if offer is None:
        return PropValueBet(
            prediction=prediction,
            side=side,
            bookmaker="",
            odds=0.0,
            line=line,
            edge=0.0,
            ev=0.0,
            conformal_passed=conformal_ok,
            skip_reason="no_qualifying_offer",
        )

    if not conformal_ok:
        return PropValueBet(
            prediction=prediction,
            side=side,
            bookmaker=offer.bookmaker,
            odds=offer.odds,
            line=line,
            edge=offer.edge,
            ev=offer.ev,
            conformal_passed=False,
            skip_reason="conformal_width",
        )

    regional_rec: RegionalRecommendation | None = None
    if include_regional:
        regional_rec = compare_regions(
            event_id=prediction.event_id,
            market=f"player_{prediction.prop_code}_{side}",
            outcome=f"{prediction.player_name}_{side}_{line}",
            p_fair=p_point,
            quotes=quotes,
        )

    warnings = list(prediction.warnings)
    return PropValueBet(
        prediction=prediction,
        side=side,
        bookmaker=offer.bookmaker,
        odds=offer.odds,
        line=line,
        edge=offer.edge,
        ev=offer.ev,
        regional=regional_rec,
        conformal_passed=True,
        skip_reason=None,
        warnings=warnings,
    )


def detect_prop_value_bets(
    prediction: PropPrediction,
    lines_offered: list[PropLineQuote],
    *,
    include_regional: bool = True,
) -> list[PropValueBet]:
    """Evalúa todas las líneas disponibles para un prop_code y jugador."""
    results: list[PropValueBet] = []
    for line_data in lines_offered:
        quotes_over = (
            [
                BookmakerQuote(
                    bookmaker=line_data.bookmaker, odds=line_data.over_odds, line=line_data.line
                )
            ]
            if line_data.over_odds
            else []
        )
        quotes_under = (
            [
                BookmakerQuote(
                    bookmaker=line_data.bookmaker, odds=line_data.under_odds, line=line_data.line
                )
            ]
            if line_data.under_odds
            else []
        )

        if quotes_over:
            over_bet = evaluate_prop_side(
                prediction,
                side="over",
                line=line_data.line,
                quotes=quotes_over,
                include_regional=include_regional,
            )
            if over_bet is not None:
                results.append(over_bet)

        if quotes_under:
            under_bet = evaluate_prop_side(
                prediction,
                side="under",
                line=line_data.line,
                quotes=quotes_under,
                include_regional=include_regional,
            )
            if under_bet is not None:
                results.append(under_bet)
    return results


def format_prop_telegram(bet: PropValueBet) -> str:
    """Formato reporte §23.7."""
    pred = bet.prediction
    line_str = f"{bet.side.upper()} {bet.line}"
    ci_str = ""
    if pred.p_over_lower is not None and pred.p_over_upper is not None:
        ci_str = f"\n   CI 90%: [{pred.p_over_lower:.2f}, {pred.p_over_upper:.2f}]"

    emoji = {
        "nba": "🏀",
        "mlb": "⚾",
        "nfl": "🏈",
        "soccer": "⚽",
        "tennis": "🎾",
        "nhl": "🏒",
        "boxing": "🥊",
    }.get(pred.prop_code.split("_")[0], "🎯")

    lines = [
        f"{emoji} *{pred.player_name}* — {pred.prop_code}",
        f"Prop: {line_str} @ {bet.odds:.2f} ({bet.bookmaker})",
        "",
        f"📊 Modelo: μ={pred.mean:.2f} σ={pred.std:.2f}",
        f"P({bet.side}) = {pred.p_over if bet.side == 'over' else pred.p_under or 0:.3f}{ci_str}",
        "",
        f"🎯 EV: *{bet.ev:+.2%}* · Edge: {bet.edge:+.2%}",
    ]
    if bet.regional and bet.regional.cross_recommendation in {"MX", "US"}:
        lines.append(
            f"\n🌎 Regional: **{bet.regional.cross_recommendation}** "
            f"(+{abs(bet.regional.expected_profit_diff_pct):.2%})"
        )
    if bet.warnings:
        lines.append(f"\n⚠️ {', '.join(bet.warnings[:3])}")
    return "\n".join(lines)
