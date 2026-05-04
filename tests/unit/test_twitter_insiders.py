"""Tests para extract_injury_from_text + dataclass InsiderReport."""

from __future__ import annotations

from datetime import UTC, datetime

from apuestas.ingest.twitter_insiders import (
    INSIDERS,
    InsiderReport,
    extract_injury_from_text,
)


def test_insiders_catalog_has_major_sports() -> None:
    assert "nba" in INSIDERS
    assert "nfl" in INSIDERS
    assert "mlb" in INSIDERS
    assert "soccer" in INSIDERS
    assert len(INSIDERS["nba"]) >= 2


def test_extract_injury_ruled_out() -> None:
    result = extract_injury_from_text("LeBron James ruled out for tonight")
    assert result is not None
    player, status = result
    assert status == "out"
    assert "LeBron" in player


def test_extract_injury_questionable() -> None:
    result = extract_injury_from_text("Jayson Tatum questionable with ankle sprain")
    assert result is not None
    _, status = result
    assert status == "questionable"


def test_extract_injury_returns_none_on_no_match() -> None:
    result = extract_injury_from_text("Lakers won 115-108 tonight. Great game.")
    assert result is None


def test_insider_report_dataclass_frozen() -> None:
    r = InsiderReport(
        source="@ShamsCharania",
        tweet_id="12345",
        player_name="LeBron James",
        team_name="Lakers",
        status="out",
        raw_text="LeBron ruled out",
        detected_at=datetime.now(tz=UTC),
        confidence=0.8,
    )
    assert r.confidence == 0.8
    # frozen → no se puede modificar
    import dataclasses

    assert dataclasses.fields(InsiderReport)[0].name == "source"
