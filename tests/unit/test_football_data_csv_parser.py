"""Tests del parser de football-data.co.uk CSV → HistoricalOddsRow."""

from __future__ import annotations

from datetime import UTC, datetime

from apuestas.ingest.football_data_csv import (
    BOOK_MAP,
    LEAGUE_CODES,
    match_to_historical_odds_rows,
    parse_csv_to_odds_rows,
)

_CSV_SAMPLE = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA,PSCH,PSCD,PSCA,WHH,WHD,WHA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA
E0,06/08/2022,Crystal Palace,Arsenal,0,2,A,3.50,3.30,2.15,3.56,3.33,2.16,3.45,3.40,2.18,3.50,3.20,2.15,3.60,3.40,2.20,3.52,3.31,2.16
E0,07/08/2022,Fulham,Liverpool,2,2,D,5.50,4.00,1.65,5.60,4.10,1.65,5.00,4.00,1.72,5.50,3.90,1.65,5.60,4.10,1.70,5.45,4.01,1.66
E0,07/08/2022,Tottenham,Southampton,4,1,H,1.40,5.00,8.00,1.40,4.90,8.00,1.39,4.90,8.50,1.40,4.50,7.00,1.43,5.10,8.50,1.41,4.91,8.01
"""


def test_league_codes_has_epl() -> None:
    assert LEAGUE_CODES["epl"] == "E0"


def test_book_map_includes_pinnacle_opening_and_closing() -> None:
    assert BOOK_MAP["PS"] == "pinnacle"
    assert BOOK_MAP["PSC"] == "pinnacle_close"


def test_parse_returns_3_matches() -> None:
    rows = parse_csv_to_odds_rows(_CSV_SAMPLE, league_slug="epl")
    assert len(rows) == 3


def test_parse_match_fields() -> None:
    rows = parse_csv_to_odds_rows(_CSV_SAMPLE, league_slug="epl")
    r0 = rows[0]
    assert r0["home_name"] == "Crystal Palace"
    assert r0["away_name"] == "Arsenal"
    assert r0["home_score"] == 0
    assert r0["away_score"] == 2
    assert r0["start_time"] == datetime(2022, 8, 6, 20, 0, tzinfo=UTC)
    assert r0["sport_code"] == "soccer"
    assert r0["league_slug"] == "epl"


def test_parse_odds_by_book() -> None:
    rows = parse_csv_to_odds_rows(_CSV_SAMPLE, league_slug="epl")
    odds = rows[0]["odds_by_book"]
    # Bet365 + Pinnacle opening + Pinnacle closing + WH + Max + Avg
    assert "bet365" in odds
    assert "pinnacle" in odds
    assert "pinnacle_close" in odds
    assert "william_hill" in odds
    assert odds["bet365"]["home"] == 3.50
    assert odds["bet365"]["draw"] == 3.30
    assert odds["bet365"]["away"] == 2.15
    assert odds["pinnacle_close"]["home"] == 3.45  # CLV reference


def test_match_to_historical_odds_rows_expands_books() -> None:
    parsed_list = parse_csv_to_odds_rows(_CSV_SAMPLE, league_slug="epl")
    parsed = parsed_list[0]
    rows = match_to_historical_odds_rows(parsed, match_id=999)
    # Al menos 6 books × 1 row c/u (opening) + 1 row closing = 6+ (cada book genera 1 row)
    assert len(rows) >= 6
    # Hay 1 row marcada closing (la de pinnacle_close)
    closing_rows = [r for r in rows if r.is_closing]
    assert len(closing_rows) == 1
    assert closing_rows[0].bookmaker == "pinnacle"
    # Hay 1 row opening de bet365
    b365_opening = [r for r in rows if r.bookmaker == "bet365" and not r.is_closing]
    assert len(b365_opening) == 1
    assert b365_opening[0].outcomes_odds["home"] == 3.50


def test_parse_skips_rows_without_teams() -> None:
    bad_csv = "Date,HomeTeam,AwayTeam,B365H,B365D,B365A\n06/08/2022,,Arsenal,2.0,3.0,3.5\n"
    rows = parse_csv_to_odds_rows(bad_csv, league_slug="epl")
    assert rows == []


def test_parse_skips_rows_without_odds() -> None:
    no_odds_csv = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n06/08/2022,Crystal Palace,Arsenal,0,2,A\n"
    rows = parse_csv_to_odds_rows(no_odds_csv, league_slug="epl")
    assert rows == []


def test_parse_handles_missing_score() -> None:
    # Partido futuro sin FTHG/FTAG
    future_csv = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        "06/08/2026,Team A,Team B,,,2.0,3.0,3.5\n"
    )
    rows = parse_csv_to_odds_rows(future_csv, league_slug="epl")
    assert len(rows) == 1
    assert rows[0]["home_score"] is None
    assert rows[0]["away_score"] is None
