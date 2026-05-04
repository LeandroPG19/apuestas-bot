"""Tests del parser tennis-data.co.uk CSV → match rows."""

from __future__ import annotations

from datetime import UTC, datetime

from apuestas.ingest.tennis_data_csv import (
    BOOK_MAP,
    match_to_historical_odds_rows,
    parse_csv_to_match_rows,
)

_CSV_SAMPLE = """ATP,Location,Tournament,Date,Series,Court,Surface,Round,Best of,Winner,Loser,WRank,LRank,Wsets,Lsets,PSW,PSL,B365W,B365L,MaxW,MaxL,AvgW,AvgL
ATP,Melbourne,Australian Open,16/01/2023,Grand Slam,Outdoor,Hard,1st Round,5,Djokovic N.,Carballes Baena R.,5,74,3,0,1.05,15.00,1.04,14.00,1.06,16.00,1.05,14.50
ATP,Melbourne,Australian Open,17/01/2023,Grand Slam,Outdoor,Hard,1st Round,5,Nadal R.,McDonald M.,2,65,1,3,1.25,4.20,1.22,4.33,1.28,4.50,1.25,4.28
"""


def test_book_map_includes_pinnacle() -> None:
    assert BOOK_MAP["PS"] == "pinnacle"


def test_parse_returns_two_matches() -> None:
    rows = parse_csv_to_match_rows(_CSV_SAMPLE, tour="atp")
    assert len(rows) == 2


def test_parse_match_fields() -> None:
    rows = parse_csv_to_match_rows(_CSV_SAMPLE, tour="atp")
    r = rows[0]
    assert r["tour"] == "atp"
    assert r["winner_name"] == "Djokovic N."
    assert r["loser_name"] == "Carballes Baena R."
    assert r["tournament"] == "Australian Open"
    assert r["surface"] == "hard"
    assert r["start_time"] == datetime(2023, 1, 16, 12, 0, tzinfo=UTC)
    assert r["winner_sets"] == 3
    assert r["loser_sets"] == 0
    assert r["best_of"] == 5


def test_parse_odds_by_book() -> None:
    rows = parse_csv_to_match_rows(_CSV_SAMPLE, tour="atp")
    odds = rows[0]["odds_by_book"]
    assert "pinnacle" in odds
    assert "bet365" in odds
    assert odds["pinnacle"]["winner"] == 1.05
    assert odds["pinnacle"]["loser"] == 15.00


def test_match_to_historical_odds_rows_tennis() -> None:
    parsed_list = parse_csv_to_match_rows(_CSV_SAMPLE, tour="atp")
    rows = match_to_historical_odds_rows(parsed_list[0], match_id=42)
    # 4 books (PS, B365, Max, Avg) → 4 rows
    assert len(rows) >= 4
    pinnacle = [r for r in rows if r.bookmaker == "pinnacle"][0]
    assert pinnacle.outcomes_odds["winner"] == 1.05
    assert pinnacle.outcomes_odds["loser"] == 15.00
    assert not pinnacle.is_closing


def test_parse_skips_rows_without_players() -> None:
    bad_csv = "ATP,Date,Winner,Loser,PSW,PSL\nATP,16/01/2023,,Nadal R.,1.50,2.50\n"
    rows = parse_csv_to_match_rows(bad_csv, tour="atp")
    assert rows == []


def test_parse_handles_invalid_odds() -> None:
    bad_csv = (
        "ATP,Date,Winner,Loser,Wsets,Lsets,Best of,PSW,PSL\n"
        "ATP,16/01/2023,Djokovic N.,Nadal R.,3,0,5,N/A,N/A\n"
    )
    rows = parse_csv_to_match_rows(bad_csv, tour="atp")
    # Sin books válidos → descarta
    assert rows == []
