"""Tests us_books_scraper: parser + camoufox fail-soft + fetch_all."""

from __future__ import annotations

import pytest

from apuestas.ingest.us_books_scraper import (
    DK_EVENTGROUP_IDS,
    USBookOdds,
    _american_to_decimal,
    parse_draftkings,
)


def test_dk_eventgroup_ids_deportes_principales() -> None:
    for s in ("nba", "mlb", "nfl", "nhl", "soccer_epl"):
        assert s in DK_EVENTGROUP_IDS


def test_american_to_decimal_basico() -> None:
    assert _american_to_decimal(-110) == 1.9091
    assert _american_to_decimal(200) == 3.0
    assert _american_to_decimal(-200) == 1.5


def test_parse_draftkings_estructura_vacia_devuelve_lista() -> None:
    """Parser no crashea con eventGroup vacío."""
    assert parse_draftkings({}, "nba") == []
    assert parse_draftkings({"eventGroup": {}}, "nba") == []
    assert parse_draftkings({"eventGroup": {"events": []}}, "nba") == []


def test_parse_draftkings_con_moneyline() -> None:
    """Fixture mínima: un event + moneyline con 2 outcomes."""
    payload = {
        "eventGroup": {
            "events": [
                {
                    "eventId": 12345,
                    "teamName1": "Lakers",
                    "teamName2": "Celtics",
                    "startDate": "2026-04-22T00:00:00Z",
                }
            ],
            "offerCategories": [
                {
                    "name": "Game Lines",
                    "offerSubcategoryDescriptors": [
                        {
                            "name": "Moneyline",
                            "offerSubcategory": {
                                "offers": [
                                    [
                                        {
                                            "eventId": 12345,
                                            "outcomes": [
                                                {
                                                    "label": "Lakers",
                                                    "oddsDecimal": 1.95,
                                                },
                                                {
                                                    "label": "Celtics",
                                                    "oddsDecimal": 1.90,
                                                },
                                            ],
                                        }
                                    ]
                                ]
                            },
                        }
                    ],
                }
            ],
        }
    }
    out = parse_draftkings(payload, "nba")
    assert len(out) == 2
    assert {o.outcome for o in out} == {"home", "away"}
    assert out[0].bookmaker == "draftkings"
    assert out[0].event_external_id == "12345"
    assert out[0].market == "h2h"
    assert abs(out[0].odds_decimal - 1.95) < 0.01


def test_usbookodds_es_frozen() -> None:
    odds = USBookOdds(
        bookmaker="draftkings",
        sport_code="nba",
        event_external_id="1",
        home="A",
        away="B",
        start_time=__import__("datetime").datetime(2026, 4, 20, tzinfo=__import__("datetime").UTC),
        market="h2h",
        outcome="home",
        odds_decimal=1.95,
    )
    with pytest.raises((AttributeError, TypeError)):
        odds.odds_decimal = 2.0  # type: ignore[misc]


@pytest.mark.asyncio
async def test_fetch_all_sin_camoufox_fail_soft() -> None:
    """Sin camoufox disponible, fetch_all no crashea — devuelve listas vacías."""
    from apuestas.ingest.us_books_scraper import fetch_all

    # Como no hay camoufox en el venv de test, internamente retorna [] para cada sport
    result = await fetch_all(["nba", "mlb"])
    assert "draftkings" in result
    assert "betmgm" in result
    # Puede ser [] si camoufox falla o tiene entradas; invariante: son listas
    assert isinstance(result["draftkings"], list)
    assert isinstance(result["betmgm"], list)
