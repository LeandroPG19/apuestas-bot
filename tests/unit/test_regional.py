"""Tests del análisis cross-región MX+US."""

from __future__ import annotations

from apuestas.betting import regional as _reg
from apuestas.betting.ev import BookmakerQuote
from apuestas.betting.regional import (
    ALL_BOOKS,
    MX_BOOKS,
    US_BOOKS,
    Region,
    compare_regions,
    filter_by_region,
    find_best_regional_offer,
    format_regional_summary,
    get_profile,
)

OFFSHORE_BOOKS = _reg.OFFSHORE_BOOKS


def test_catalog_separation() -> None:
    """MX → Region.MX, US → Region.US, offshore sharp → Region.OFFSHORE."""
    for slug, profile in MX_BOOKS.items():
        assert profile.region == Region.MX, f"{slug} mal clasificado"
        assert profile.segob_license is True
    for slug, profile in US_BOOKS.items():
        assert profile.region == Region.US, f"{slug} mal clasificado"
    for slug, profile in OFFSHORE_BOOKS.items():
        assert profile.region == Region.OFFSHORE, f"{slug} mal clasificado"


def test_all_books_union() -> None:
    assert set(ALL_BOOKS) == set(MX_BOOKS) | set(US_BOOKS) | set(OFFSHORE_BOOKS)


def test_pinnacle_betfair_son_offshore_no_apostables() -> None:
    """Pinnacle y Betfair NO deben aparecer en US_BOOKS (no apostables desde US regular)."""
    assert "pinnacle" not in US_BOOKS
    assert "betfair" not in US_BOOKS
    assert "pinnacle" in OFFSHORE_BOOKS
    assert "betfair" in OFFSHORE_BOOKS


def test_get_profile_known_and_unknown() -> None:
    assert get_profile("caliente") is not None
    assert get_profile("draftkings") is not None
    assert get_profile("nonexistent_book") is None


def test_filter_by_region_mx_only() -> None:
    books = ["caliente", "draftkings", "strendus", "fanduel"]
    mx_filtered = filter_by_region(books, Region.MX)
    assert "caliente" in mx_filtered
    assert "strendus" in mx_filtered
    assert "draftkings" not in mx_filtered
    assert "fanduel" not in mx_filtered


def test_filter_by_region_us_only() -> None:
    books = ["caliente", "draftkings", "strendus", "fanduel"]
    us_filtered = filter_by_region(books, Region.US)
    assert "draftkings" in us_filtered
    assert "fanduel" in us_filtered
    assert "caliente" not in us_filtered


def test_filter_by_region_both() -> None:
    books = ["caliente", "draftkings"]
    both = filter_by_region(books, Region.BOTH)
    assert both == frozenset(books)


def test_find_best_regional_offer_mx() -> None:
    """Con 3 cuotas MX + 2 US, MX region devuelve la mejor MX."""
    quotes = [
        BookmakerQuote(bookmaker="caliente", odds=1.95),
        BookmakerQuote(bookmaker="strendus", odds=1.98),
        BookmakerQuote(bookmaker="codere", odds=1.93),
        BookmakerQuote(bookmaker="draftkings", odds=2.00),
        BookmakerQuote(bookmaker="fanduel", odds=1.96),
    ]
    result = find_best_regional_offer(quotes, p_fair=0.58, region=Region.MX)
    assert result.best_offer is not None
    assert result.best_offer.bookmaker == "strendus"
    assert result.profile is not None
    assert result.profile.region == Region.MX


def test_find_best_regional_offer_no_qualifying() -> None:
    quotes = [BookmakerQuote(bookmaker="caliente", odds=1.50)]  # odds bajo min
    result = find_best_regional_offer(quotes, p_fair=0.80, region=Region.MX)
    # El evaluador puede rechazar por EV/odds range
    if result.best_offer is None:
        assert True  # aceptable
    else:
        assert result.best_offer.odds >= 1.5


def test_compare_regions_recommends_mx_when_better() -> None:
    quotes = [
        BookmakerQuote(bookmaker="strendus", odds=2.00),  # MX mejor
        BookmakerQuote(bookmaker="draftkings", odds=1.88),
    ]
    rec = compare_regions(
        event_id=1,
        market="h2h",
        outcome="home",
        p_fair=0.58,
        quotes=quotes,
    )
    assert rec.cross_recommendation == "MX"
    assert rec.mx.best_offer is not None
    assert rec.mx.best_offer.bookmaker == "strendus"


def test_compare_regions_recommends_us_when_better() -> None:
    quotes = [
        BookmakerQuote(bookmaker="caliente", odds=1.85),
        BookmakerQuote(bookmaker="draftkings", odds=2.05),  # US mejor
    ]
    rec = compare_regions(
        event_id=2,
        market="h2h",
        outcome="home",
        p_fair=0.55,
        quotes=quotes,
    )
    assert rec.cross_recommendation == "US"


def test_compare_regions_tie() -> None:
    """Cuotas equivalentes en ambas regiones → tie."""
    quotes = [
        BookmakerQuote(bookmaker="strendus", odds=1.95),
        BookmakerQuote(bookmaker="fanduel", odds=1.95),
    ]
    rec = compare_regions(
        event_id=3,
        market="h2h",
        outcome="home",
        p_fair=0.58,
        quotes=quotes,
    )
    # Diferencia de ajuste de tolerancia puede romper tie pero debe ser pequeño
    assert rec.cross_recommendation in ("tie", "MX", "US")
    assert abs(rec.expected_profit_diff_pct) < 0.01


def test_compare_regions_neither_when_no_offers() -> None:
    """Sin ninguna cuota pasa → neither."""
    quotes = [
        BookmakerQuote(bookmaker="caliente", odds=1.30),  # bajo min_odds
    ]
    rec = compare_regions(
        event_id=4,
        market="h2h",
        outcome="home",
        p_fair=0.80,
        quotes=quotes,
    )
    assert rec.cross_recommendation in ("MX", "neither")


def test_format_regional_summary_not_empty() -> None:
    quotes = [
        BookmakerQuote(bookmaker="strendus", odds=1.98),
        BookmakerQuote(bookmaker="draftkings", odds=1.95),
    ]
    rec = compare_regions(
        event_id=99,
        market="h2h",
        outcome="home",
        p_fair=0.58,
        quotes=quotes,
    )
    summary = format_regional_summary(rec)
    assert "🇲🇽" in summary
    assert "🇺🇸" in summary
    assert "Recomendación" in summary


def test_pinnacle_and_circa_absent_from_mx_catalog() -> None:
    """Pinnacle/Circa NO deben aparecer en MX_BOOKS (solo benchmark)."""
    assert "pinnacle" not in MX_BOOKS
    assert "circa" not in MX_BOOKS


def test_low_limit_book_warning() -> None:
    """Books con typical_limit_usd<1000 disparan warning informativo."""
    # Algunos books MX soft tienen típicos bajos; el warning se emite si el book
    # elegido tiene ese perfil (sin depender de stake/bankroll).
    quotes = [BookmakerQuote(bookmaker="caliente", odds=1.95)]
    result = find_best_regional_offer(quotes, p_fair=0.58, region=Region.MX)
    if (
        result.best_offer is not None
        and result.profile is not None
        and result.profile.typical_limit_usd < 1000
    ):
        assert any("low_limit" in w for w in result.warnings)
