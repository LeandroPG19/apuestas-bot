"""Análisis regional MX + US — mejor casa para cada pick por jurisdicción.

NUEVO requisito del usuario: el bot debe analizar casas de apuestas de
México Y Estados Unidos, y para cada pick recomendar en cuál conviene
apostar en cada región para maximizar ganancias.

Datos considerados por casa:
- Odds ofrecidas (prime para line shopping)
- Límites típicos por sport/mercado (capacidad)
- Velocidad de pago
- Tolerancia a ganadores (historial)
- Disponibilidad geo (MX / US state)
- Costo de transacción (OXXO/SPEI vs Venmo/ACH)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from apuestas.betting.ev import (
    BestOffer,
    BookmakerQuote,
    evaluate_offer,
)
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


class Region(StrEnum):
    MX = "MX"
    US = "US"
    BOTH = "BOTH"
    # OFFSHORE = Pinnacle / Betfair / Circa. Sharp benchmark, NO apostable
    # legalmente desde MX o US por residentes regulares. Solo fair value.
    OFFSHORE = "OFFSHORE"


@dataclass(slots=True, frozen=True)
class BookProfile:
    slug: str
    display_name: str
    region: Region
    segob_license: bool = False  # solo aplica MX
    typical_limit_usd: int = 500  # orden de magnitud postload
    pro_tolerance: Literal["high", "medium", "low"] = "medium"
    payout_speed_hours: int = 48
    margin_pct_typical: float = 0.05
    mobile_app: bool = True
    supports_oxxo_spei: bool = False  # relevante MX
    supports_ach_venmo: bool = False  # relevante US
    # Fase 5.10/5.11 — nuevas flags para line shopping multi-jurisdicción
    apostable_mx: bool = False  # accesible desde IP mexicana sin VPN
    apostable_us: bool = False  # accesible desde IP USA sin VPN
    soft_book: bool = False  # líneas blandas (público casual) vs sharp
    deposit_methods: tuple[str, ...] = ()  # ("fiat",) | ("crypto_only",) | mixed
    notes: str = ""


# ═══════════════════════ Catálogo México ═══════════════════════════════════

MX_BOOKS: dict[str, BookProfile] = {
    "caliente": BookProfile(
        slug="caliente",
        display_name="Caliente.mx",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=3000,
        pro_tolerance="medium",  # limita eventualmente
        payout_speed_hours=24,
        margin_pct_typical=0.05,
        supports_oxxo_spei=True,
        notes="Dominante MX. Patrocina Xolos. Mejor cobertura Liga MX.",
    ),
    "strendus": BookProfile(
        slug="strendus",
        display_name="Strendus",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=2000,
        pro_tolerance="medium",
        payout_speed_hours=24,
        margin_pct_typical=0.03,
        supports_oxxo_spei=True,
        notes="Grupo Caliente. Online-first. Programa de lealtad.",
    ),
    "codere": BookProfile(
        slug="codere",
        display_name="Codere.mx",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=2000,
        pro_tolerance="low",
        payout_speed_hours=48,
        margin_pct_typical=0.05,
        supports_oxxo_spei=True,
        notes="OXXO, menos mercados exóticos. Tolerancia baja a sharps.",
    ),
    "betway_mx": BookProfile(
        slug="betway_mx",
        display_name="Betway México",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=1500,
        pro_tolerance="low",
        payout_speed_hours=48,
        margin_pct_typical=0.04,
        supports_oxxo_spei=True,
        notes="Operador europeo con permiso local.",
    ),
    "betano_mx": BookProfile(
        slug="betano_mx",
        display_name="Betano México",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=1500,
        pro_tolerance="low",
        payout_speed_hours=36,
        margin_pct_typical=0.04,
        supports_oxxo_spei=True,
    ),
    "betsson_mx": BookProfile(
        slug="betsson_mx",
        display_name="Betsson México",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=1500,
        pro_tolerance="low",
        margin_pct_typical=0.05,
        supports_oxxo_spei=True,
    ),
    "bwin_mx": BookProfile(
        slug="bwin_mx",
        display_name="bwin México",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=1500,
        pro_tolerance="low",
        margin_pct_typical=0.05,
        supports_oxxo_spei=True,
    ),
    "novibet_mx": BookProfile(
        slug="novibet_mx",
        display_name="Novibet MX",
        region=Region.MX,
        segob_license=True,
        typical_limit_usd=1500,
        pro_tolerance="low",
        margin_pct_typical=0.05,
        supports_oxxo_spei=True,
    ),
}

# ═══════════════════════ Catálogo Estados Unidos ═══════════════════════════

US_BOOKS: dict[str, BookProfile] = {
    "draftkings": BookProfile(
        slug="draftkings",
        display_name="DraftKings",
        region=Region.US,
        typical_limit_usd=2500,
        pro_tolerance="medium",
        payout_speed_hours=24,
        margin_pct_typical=0.045,
        supports_ach_venmo=True,
        notes="Amplia disponibilidad geo US. Mejor cobertura de player props NBA/NFL.",
    ),
    "fanduel": BookProfile(
        slug="fanduel",
        display_name="FanDuel",
        region=Region.US,
        typical_limit_usd=2500,
        pro_tolerance="medium",
        payout_speed_hours=24,
        margin_pct_typical=0.045,
        supports_ach_venmo=True,
        notes="Same Game Parlay fuerte. Popular entre casuales.",
    ),
    "betmgm": BookProfile(
        slug="betmgm",
        display_name="BetMGM",
        region=Region.US,
        typical_limit_usd=5000,
        pro_tolerance="medium",
        payout_speed_hours=48,
        margin_pct_typical=0.05,
        supports_ach_venmo=True,
        notes="Límites altos. Promociones frecuentes.",
    ),
    "caesars": BookProfile(
        slug="caesars",
        display_name="Caesars Sportsbook",
        region=Region.US,
        typical_limit_usd=3000,
        pro_tolerance="medium",
        payout_speed_hours=48,
        margin_pct_typical=0.05,
        supports_ach_venmo=True,
    ),
    "pointsbet": BookProfile(
        slug="pointsbet",
        display_name="PointsBet (Fanatics)",
        region=Region.US,
        typical_limit_usd=2000,
        pro_tolerance="high",  # reputación amigable a sharps
        margin_pct_typical=0.04,
        supports_ach_venmo=True,
        notes="PointsBetting exclusivo. Rebrandeado como Fanatics.",
    ),
    "betrivers": BookProfile(
        slug="betrivers",
        display_name="BetRivers",
        region=Region.US,
        typical_limit_usd=1500,
        pro_tolerance="medium",
        margin_pct_typical=0.05,
        supports_ach_venmo=True,
    ),
    "espnbet": BookProfile(
        slug="espnbet",
        display_name="ESPN BET",
        region=Region.US,
        typical_limit_usd=2000,
        pro_tolerance="medium",
        margin_pct_typical=0.05,
        supports_ach_venmo=True,
        notes="Relanzado 2023 de Barstool. Integrado con ESPN scores.",
    ),
    "hardrock": BookProfile(
        slug="hardrock",
        display_name="Hard Rock Bet",
        region=Region.US,
        typical_limit_usd=1500,
        pro_tolerance="medium",
        margin_pct_typical=0.05,
        supports_ach_venmo=True,
        notes="Solo en ciertos estados US.",
    ),
    "circa": BookProfile(
        slug="circa",
        display_name="Circa Sports",
        region=Region.US,  # Regulado Nevada — apostable con viaje o residencia NV
        typical_limit_usd=10000,
        pro_tolerance="high",
        margin_pct_typical=0.02,
        supports_ach_venmo=True,
        notes="Sharp book Nevada. NO limita a sharps. Apostable solo in-state NV.",
    ),
}

# Books OFFSHORE = sharp benchmarks NO apostables regularmente desde MX/US.
# Se usan para de-vigging Shin (§7) y line movement tracking.
# Fase 5.10: añadidos soft-books offshore APOSTABLES desde MX+US sin VPN
# (BetUS, BetWhale, Everygame, SportsBetting.ag, BC.GAME).
OFFSHORE_BOOKS: dict[str, BookProfile] = {
    "pinnacle": BookProfile(
        slug="pinnacle",
        display_name="Pinnacle",
        region=Region.OFFSHORE,
        typical_limit_usd=20000,
        pro_tolerance="high",
        margin_pct_typical=0.02,
        apostable_mx=False,
        apostable_us=False,
        soft_book=False,
        notes="Curaçao. Sharp benchmark para de-vigging. NO operaciones MX/US.",
    ),
    "betfair": BookProfile(
        slug="betfair",
        display_name="Betfair Exchange",
        region=Region.OFFSHORE,
        typical_limit_usd=50000,
        pro_tolerance="high",
        margin_pct_typical=0.015,
        apostable_mx=False,
        apostable_us=False,
        soft_book=False,
        notes="UK-regulado. Exchange. Delayed API key gratis. Solo benchmark.",
    ),
    "betus": BookProfile(
        slug="betus",
        display_name="BetUS",
        region=Region.OFFSHORE,
        typical_limit_usd=2000,
        pro_tolerance="medium",
        margin_pct_typical=0.05,
        apostable_mx=True,
        apostable_us=True,
        soft_book=True,
        deposit_methods=("fiat", "crypto"),
        notes="Panamá, SBR 3.3/5, bonos agresivos, soft-book accesible MX+US.",
    ),
    "betwhale": BookProfile(
        slug="betwhale",
        display_name="BetWhale",
        region=Region.OFFSHORE,
        typical_limit_usd=1500,
        pro_tolerance="medium",
        margin_pct_typical=0.055,
        apostable_mx=True,
        apostable_us=True,
        soft_book=True,
        deposit_methods=("fiat", "crypto"),
        notes="Curaçao 2023+, bono 125% hasta $1250, SBR sin reviews aún.",
    ),
    "everygame": BookProfile(
        slug="everygame",
        display_name="Everygame (Intertops)",
        region=Region.OFFSHORE,
        typical_limit_usd=2000,
        pro_tolerance="medium",
        margin_pct_typical=0.05,
        apostable_mx=True,
        apostable_us=True,
        soft_book=True,
        deposit_methods=("fiat", "crypto"),
        notes="Malta+Curaçao, veterano desde 1996, SBR 2.8/5 (mejor del lote soft).",
    ),
    "sportsbetting_ag": BookProfile(
        slug="sportsbetting_ag",
        display_name="SportsBetting.ag",
        region=Region.OFFSHORE,
        typical_limit_usd=2000,
        pro_tolerance="medium",
        margin_pct_typical=0.05,
        apostable_mx=True,
        apostable_us=True,
        soft_book=True,
        deposit_methods=("fiat", "crypto"),
        notes="Panamá+Curaçao 2003, mismo grupo BetOnline, cobertura eSports, SBR 2.0/5.",
    ),
    "bc_game": BookProfile(
        slug="bc_game",
        display_name="BC.GAME",
        region=Region.OFFSHORE,
        typical_limit_usd=5000,
        pro_tolerance="medium",
        margin_pct_typical=0.045,
        apostable_mx=True,
        apostable_us=True,
        soft_book=True,
        deposit_methods=("crypto_only",),
        notes="Curaçao, crypto-only (BTC/ETH/USDT). Edge en eSports + crypto users.",
    ),
}

# Fase 5.11: añadir 3 books MX regulados SEGOB adicionales (Winpot, CampoBet, JugaBet)
MX_BOOKS.update(
    {
        "winpot": BookProfile(
            slug="winpot",
            display_name="Winpot",
            region=Region.MX,
            segob_license=True,
            typical_limit_usd=1500,
            pro_tolerance="medium",
            margin_pct_typical=0.07,
            supports_oxxo_spei=True,
            apostable_mx=True,
            soft_book=True,
            deposit_methods=("fiat",),
            notes="Codere Group, SEGOB MX regulado desde 2019, bono $3k MXN.",
        ),
        "campobet": BookProfile(
            slug="campobet",
            display_name="CampoBet",
            region=Region.MX,
            segob_license=True,
            typical_limit_usd=1500,
            pro_tolerance="medium",
            margin_pct_typical=0.07,
            supports_oxxo_spei=True,
            apostable_mx=True,
            soft_book=True,
            deposit_methods=("fiat",),
            notes="SEGOB MX regulado, bono $5k MXN. Cobertura amplia Liga MX.",
        ),
        "jugabet": BookProfile(
            slug="jugabet",
            display_name="JugaBet",
            region=Region.MX,
            segob_license=True,
            typical_limit_usd=1500,
            pro_tolerance="medium",
            margin_pct_typical=0.07,
            supports_oxxo_spei=True,
            apostable_mx=True,
            soft_book=True,
            deposit_methods=("fiat",),
            notes="SEGOB MX regulado, bono $5k MXN, cobertura Liga MX completa.",
        ),
    }
)

ALL_BOOKS: dict[str, BookProfile] = {**MX_BOOKS, **US_BOOKS, **OFFSHORE_BOOKS}


def get_profile(slug: str) -> BookProfile | None:
    return ALL_BOOKS.get(slug)


def filter_by_region(
    books: frozenset[str] | list[str] | set[str], region: Region
) -> frozenset[str]:
    """Devuelve solo los slugs disponibles en la región indicada."""
    if region == Region.BOTH:
        return frozenset(books)
    return frozenset(b for b in books if (p := ALL_BOOKS.get(b)) is not None and p.region == region)


# ═══════════════════════ Recomendación regional ═══════════════════════════


@dataclass(slots=True)
class RegionalOffer:
    region: Region
    best_offer: BestOffer | None
    all_offers: list[BestOffer] = field(default_factory=list)
    profile: BookProfile | None = None
    expected_net_profit_pct: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegionalRecommendation:
    event_id: int | str
    market: str
    outcome: str
    p_fair: float
    mx: RegionalOffer
    us: RegionalOffer
    cross_recommendation: str  # "MX"|"US"|"tie"|"neither"
    reason: str
    expected_profit_diff_pct: float  # mx_ev - us_ev (en puntos pct)


def _net_profit_adjustment(profile: BookProfile) -> float:
    """Ajuste heurístico de EV por fricciones prácticas de la casa.

    - Tolerancia a pros: sharp-friendly preserva EV más tiempo.
    - Velocidad pago: lento reduce VNA.
    """
    adj = 0.0
    tolerance_adj = {
        "high": 0.0,
        "medium": -0.002,
        "low": -0.008,  # cuenta cerrada probable reduce EV realizable
    }
    adj += tolerance_adj.get(profile.pro_tolerance, -0.005)
    if profile.payout_speed_hours > 48:
        adj -= 0.001
    return adj


def find_best_regional_offer(
    quotes: list[BookmakerQuote],
    *,
    p_fair: float,
    region: Region,
) -> RegionalOffer:
    """Evalúa todas las cuotas válidas en la región y devuelve best + ranking.

    Post-pivote 2026-04-23: sin bankroll/stake (la signature conserva la
    compatibilidad mínima con tests actualizados en Sprint 1).
    """
    allowed = filter_by_region([q.bookmaker for q in quotes], region)
    filtered = [q for q in quotes if q.bookmaker in allowed]
    if not filtered:
        return RegionalOffer(region=region, best_offer=None)

    offers: list[BestOffer] = []
    for q in filtered:
        offer = evaluate_offer(p_fair=p_fair, quote=q)
        if offer is None:
            continue
        offers.append(offer)

    if not offers:
        return RegionalOffer(region=region, best_offer=None)

    offers.sort(key=lambda o: o.ev, reverse=True)
    best = offers[0]
    profile = ALL_BOOKS.get(best.bookmaker)
    net_adj = _net_profit_adjustment(profile) if profile else 0.0
    expected_net = best.ev + net_adj

    warnings: list[str] = []
    if profile is None:
        warnings.append("profile_missing")
    else:
        if profile.pro_tolerance == "low":
            warnings.append("low_sharp_tolerance_account_risk")
        if profile.typical_limit_usd < 1000:
            warnings.append(f"low_limit_{profile.typical_limit_usd}usd")

    return RegionalOffer(
        region=region,
        best_offer=best,
        all_offers=offers,
        profile=profile,
        expected_net_profit_pct=expected_net,
        warnings=warnings,
    )


def compare_regions(
    *,
    event_id: int | str,
    market: str,
    outcome: str,
    p_fair: float,
    quotes: list[BookmakerQuote],
) -> RegionalRecommendation:
    """Produce recomendación cross-región MX vs US."""
    mx_offer = find_best_regional_offer(quotes, p_fair=p_fair, region=Region.MX)
    us_offer = find_best_regional_offer(quotes, p_fair=p_fair, region=Region.US)

    mx_net = mx_offer.expected_net_profit_pct or -1.0
    us_net = us_offer.expected_net_profit_pct or -1.0

    if mx_offer.best_offer is None and us_offer.best_offer is None:
        rec = "neither"
        reason = "no_qualifying_offer_either_region"
    elif mx_offer.best_offer is None:
        rec = "US"
        reason = "only_us_has_qualifying_offer"
    elif us_offer.best_offer is None:
        rec = "MX"
        reason = "only_mx_has_qualifying_offer"
    else:
        delta = mx_net - us_net
        if abs(delta) < 0.005:
            rec = "tie"
            reason = f"similar_ev_delta_{delta:.4f}"
        elif delta > 0:
            rec = "MX"
            reason = f"mx_net_better_by_{delta:.4f}"
        else:
            rec = "US"
            reason = f"us_net_better_by_{-delta:.4f}"

    logger.info(
        "regional.comparison",
        event_id=event_id,
        market=market,
        outcome=outcome,
        recommendation=rec,
        mx_book=mx_offer.best_offer.bookmaker if mx_offer.best_offer else None,
        us_book=us_offer.best_offer.bookmaker if us_offer.best_offer else None,
        mx_net=mx_net,
        us_net=us_net,
    )

    return RegionalRecommendation(
        event_id=event_id,
        market=market,
        outcome=outcome,
        p_fair=p_fair,
        mx=mx_offer,
        us=us_offer,
        cross_recommendation=rec,
        reason=reason,
        expected_profit_diff_pct=mx_net - us_net,
    )


def format_regional_summary(rec: RegionalRecommendation) -> str:
    """Formato legible para Telegram message."""
    lines = [
        f"🌎 Análisis regional · {rec.market} · {rec.outcome}",
        f"P fair: {rec.p_fair:.3f}",
        "",
    ]

    def _fmt_region(offer: RegionalOffer, emoji: str, label: str) -> list[str]:
        if offer.best_offer is None:
            return [f"{emoji} {label}: sin oferta +EV elegible"]
        b = offer.best_offer
        prof = offer.profile
        warn = f" ⚠️ {','.join(offer.warnings)}" if offer.warnings else ""
        return [
            f"{emoji} {label}: {b.bookmaker} @ {b.odds:.3f}",
            f"   EV={b.ev:+.2%} · Edge={b.edge:+.2%}",
            f"   Límite típico: ${prof.typical_limit_usd if prof else '?'} · "
            f"Tolerancia: {prof.pro_tolerance if prof else '?'}{warn}",
        ]

    lines.extend(_fmt_region(rec.mx, "🇲🇽", "MX"))
    lines.append("")
    lines.extend(_fmt_region(rec.us, "🇺🇸", "US"))
    lines.append("")

    if rec.cross_recommendation == "tie":
        lines.append("📝 Recomendación: EQUIVALENTE (apostar en la casa donde tengas cuenta).")
    elif rec.cross_recommendation in ("MX", "US"):
        lines.append(
            f"📝 Recomendación: **{rec.cross_recommendation}** "
            f"(+{abs(rec.expected_profit_diff_pct):.2%} vs alternativa)"
        )
        lines.append(f"   Razón: {rec.reason}")
    else:
        lines.append("📝 Recomendación: NO APOSTAR (sin oferta +EV en ninguna región).")

    return "\n".join(lines)
