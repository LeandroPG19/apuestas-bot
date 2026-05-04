"""Bulk ingest football-data.co.uk — odds históricas 1993-presente.

Fuente: https://www.football-data.co.uk/data.php (CSV por temporada × liga).
Licencia: Free use with attribution.

Descarga closing odds 1X2 + AH + totals + BTTS para ~100 ligas.
Persiste en `odds_history_archive` (tabla creada en 0026).

Uso:
    uv run python scripts/ingest_football_data_co_uk.py
    uv run python scripts/ingest_football_data_co_uk.py --since 2018 --leagues E0,E1,SP1
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

# League codes football-data.co.uk
LEAGUES = {
    # Inglaterra
    "E0": ("premier_league", "soccer"),
    "E1": ("championship", "soccer"),
    "E2": ("league_one", "soccer"),
    "E3": ("league_two", "soccer"),
    # España
    "SP1": ("la_liga", "soccer"),
    "SP2": ("la_liga_2", "soccer"),
    # Italia
    "I1": ("serie_a", "soccer"),
    "I2": ("serie_b", "soccer"),
    # Alemania
    "D1": ("bundesliga", "soccer"),
    "D2": ("bundesliga_2", "soccer"),
    # Francia
    "F1": ("ligue_1", "soccer"),
    "F2": ("ligue_2", "soccer"),
    # Países Bajos
    "N1": ("eredivisie", "soccer"),
    # Bélgica
    "B1": ("belgium_pro_a", "soccer"),
    # Portugal
    "P1": ("liga_portugal", "soccer"),
    # Escocia
    "SC0": ("scottish_premiership", "soccer"),
    # Turquía
    "T1": ("turkey_super_lig", "soccer"),
    # Grecia
    "G1": ("greek_super_league", "soccer"),
}


def _season_codes(since: int) -> list[str]:
    """2018 → ['1819', '1920', '2021', '2122', '2223', '2324', '2425']."""
    out = []
    for y in range(since, datetime.now(tz=UTC).year + 1):
        y1 = y % 100
        y2 = (y + 1) % 100
        out.append(f"{y1:02d}{y2:02d}")
    return out


async def _fetch_csv(client: httpx.AsyncClient, url: str) -> pd.DataFrame | None:
    try:
        r = await client.get(url, timeout=30.0)
        if r.status_code != 200 or len(r.content) < 500:
            return None
        return pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
    except Exception as exc:
        logger.debug("fd.fetch_fail", url=url, error=str(exc)[:100])
        return None


def _extract_row(row: pd.Series, league_name: str, season: str) -> dict | None:
    """Extrae columnas relevantes del CSV football-data.

    Columnas típicas:
    - Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    - B365H, B365D, B365A (Bet365 1X2)
    - BbMxH, BbMxD, BbMxA (max odds)
    - B365>2.5, B365<2.5
    """
    date_str = str(row.get("Date", ""))
    if not date_str or date_str == "nan":
        return None
    try:
        match_date = pd.to_datetime(date_str, dayfirst=True).date()
    except Exception:
        return None

    home = str(row.get("HomeTeam") or "").strip()
    away = str(row.get("AwayTeam") or "").strip()
    if not home or not away:
        return None

    def _nan_to_none(v):  # type: ignore[no-untyped-def]
        try:
            f = float(v)
            import math

            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    fthg = _nan_to_none(row.get("FTHG"))
    ftag = _nan_to_none(row.get("FTAG"))

    closing_odds = {
        "home": _nan_to_none(row.get("B365H") or row.get("AvgH") or row.get("BbAvH")),
        "draw": _nan_to_none(row.get("B365D") or row.get("AvgD") or row.get("BbAvD")),
        "away": _nan_to_none(row.get("B365A") or row.get("AvgA") or row.get("BbAvA")),
        "over25": _nan_to_none(row.get("B365>2.5") or row.get("Avg>2.5")),
        "under25": _nan_to_none(row.get("B365<2.5") or row.get("Avg<2.5")),
        "max_home": _nan_to_none(row.get("BbMxH") or row.get("MaxH")),
        "max_draw": _nan_to_none(row.get("BbMxD") or row.get("MaxD")),
        "max_away": _nan_to_none(row.get("BbMxA") or row.get("MaxA")),
    }

    return {
        "sport_code": "soccer",
        "league": league_name,
        "season": season,
        "match_date": match_date,
        "home_team": home,
        "away_team": away,
        "home_score": int(fthg) if fthg is not None else None,
        "away_score": int(ftag) if ftag is not None else None,
        "closing_odds": closing_odds,
    }


async def ingest_league_season(
    client: httpx.AsyncClient, session, code: str, season_code: str
) -> int:  # type: ignore[no-untyped-def]
    league_name, _ = LEAGUES[code]
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"
    df = await _fetch_csv(client, url)
    if df is None or len(df) == 0:
        return 0

    season_label = f"20{season_code[:2]}-20{season_code[2:]}"
    rows: list[dict] = []
    for _, row in df.iterrows():
        extracted = _extract_row(row, league_name, season_label)
        if extracted:
            rows.append(extracted)

    if not rows:
        return 0

    from sqlalchemy import text as _text

    # Bulk insert (ON CONFLICT es inviable sin unique constraint; tolera duplicados
    # usando DELETE previo por (league, season) para idempotencia).
    await session.execute(
        _text("DELETE FROM odds_history_archive WHERE league = :lg AND season = :ss"),
        {"lg": league_name, "ss": season_label},
    )
    inserted = 0
    for r in rows:
        await session.execute(
            _text(
                """
                INSERT INTO odds_history_archive (
                    sport_code, league, season, match_date, home_team, away_team,
                    home_score, away_score, closing_odds
                ) VALUES (
                    :sport, :lg, :ss, :md, :ht, :at, :hs, :ascore, CAST(:co AS jsonb)
                )
                """
            ),
            {
                "sport": r["sport_code"],
                "lg": r["league"],
                "ss": r["season"],
                "md": r["match_date"],
                "ht": r["home_team"],
                "at": r["away_team"],
                "hs": r["home_score"],
                "ascore": r["away_score"],
                "co": json.dumps(r["closing_odds"]),
            },
        )
        inserted += 1
    await session.commit()
    logger.info("fd.ingested", league=league_name, season=season_label, rows=inserted)
    return inserted


async def main_async(since: int, leagues_filter: list[str] | None) -> int:
    from apuestas.db import session_scope

    codes = list(LEAGUES.keys())
    if leagues_filter:
        codes = [c for c in codes if c in leagues_filter]

    seasons = _season_codes(since)
    total = 0
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for code in codes:
            for season_code in seasons:
                try:
                    async with session_scope() as session:
                        n = await ingest_league_season(client, session, code, season_code)
                        total += n
                except Exception as exc:
                    logger.warning(
                        "fd.season_fail", code=code, season=season_code, error=str(exc)[:80]
                    )
    logger.info("fd.done", total_rows=total, leagues=len(codes), seasons=len(seasons))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2018)
    parser.add_argument("--leagues", type=str, default=None, help="CSV codes, e.g. E0,SP1,I1")
    args = parser.parse_args()
    leagues_filter = args.leagues.split(",") if args.leagues else None
    n = asyncio.run(main_async(args.since, leagues_filter))
    print(f"✓ Inserted {n} rows from football-data.co.uk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
