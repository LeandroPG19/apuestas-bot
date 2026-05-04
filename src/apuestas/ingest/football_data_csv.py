"""Loader de odds históricas soccer desde football-data.co.uk.

Fuente: https://www.football-data.co.uk/data.php
Cobertura: 25+ ligas desde 2000 (Big-5 + otras). Columnas estándar incluyen:

| Columna  | Significado                                        |
|----------|----------------------------------------------------|
| Date     | Fecha del partido (DD/MM/YY o DD/MM/YYYY)          |
| HomeTeam | Equipo local                                        |
| AwayTeam | Equipo visitante                                    |
| FTHG     | Full-Time Home Goals                                |
| FTAG     | Full-Time Away Goals                                |
| FTR      | Full-Time Result (H/D/A)                           |
| B365H    | Bet365 home odds (opening)                          |
| B365D    | Bet365 draw odds                                    |
| B365A    | Bet365 away odds                                    |
| BWH/D/A  | bwin                                                |
| IWH/D/A  | Interwetten                                         |
| PSH/D/A  | Pinnacle opening                                    |
| WHH/D/A  | William Hill                                        |
| VCH/D/A  | VCbet                                               |
| PSCH/D/A | **Pinnacle CLOSING** (crítico para CLV histórico)   |
| MaxH/D/A | Max odds across market                              |
| AvgH/D/A | Average odds                                        |

Cache en `~/.cache/apuestas/historical/football-data/{season}/{league}.csv`.

Uso:
    from apuestas.ingest.football_data_csv import (
        fetch_league_season, parse_csv_to_odds_rows, LEAGUE_CODES,
    )
    rows = await fetch_league_season("E0", 2022)  # EPL 2022-23
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from apuestas.obs.logging import get_logger
from apuestas.validators.historical_odds_integrity import HistoricalOddsRow

logger = get_logger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Códigos de liga football-data.co.uk
LEAGUE_CODES: dict[str, str] = {
    "epl": "E0",  # Premier League
    "championship": "E1",  # Championship (2nd tier)
    "la_liga": "SP1",
    "la_liga_2": "SP2",
    "bundesliga": "D1",
    "bundesliga_2": "D2",
    "serie_a": "I1",
    "serie_b": "I2",
    "ligue_1": "F1",
    "ligue_2": "F2",
    "eredivisie": "N1",
    "liga_portugal": "P1",
    "belgium_a": "B1",
    "turkey_super": "T1",
    "greece_super": "G1",
    "scotland_premier": "SC0",
}

# Books disponibles en el CSV → nombre canónico para odds_history.bookmaker
BOOK_MAP: dict[str, str] = {
    "B365": "bet365",
    "BW": "bwin",
    "IW": "interwetten",
    "PS": "pinnacle",  # Pinnacle opening
    "PSC": "pinnacle_close",  # Pinnacle closing (CLV reference)
    "WH": "william_hill",
    "VC": "vcbet",
    "Max": "market_max",  # mejor oferta del mercado
    "Avg": "market_avg",  # consenso
}


_CACHE_DIR = Path.home() / ".cache" / "apuestas" / "historical" / "football-data"


def _season_to_url_suffix(season: int) -> str:
    """2022 → '2223' (football-data.co.uk usa formato 'YYYY' dos-dígitos concatenado)."""
    yy_start = str(season)[-2:]
    yy_end = str(season + 1)[-2:]
    return f"{yy_start}{yy_end}"


def _cache_path(season: int, league_code: str) -> Path:
    return _CACHE_DIR / _season_to_url_suffix(season) / f"{league_code}.csv"


async def fetch_csv_raw(
    league_code: str,
    season: int,
    *,
    force_refresh: bool = False,
    timeout: float = 30.0,
) -> str | None:
    """Descarga el CSV de football-data.co.uk con cache local.

    Si ya está cacheado y el archivo no está vacío, lo devuelve sin descargar.
    Retorna None si 404 / error red.
    """
    cache_file = _cache_path(season, league_code)
    if cache_file.exists() and cache_file.stat().st_size > 0 and not force_refresh:
        return cache_file.read_text(encoding="latin-1")

    url = f"{BASE_URL}/{_season_to_url_suffix(season)}/{league_code}.csv"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        logger.info("football_data.http_error", url=url[:80], error=str(exc)[:80])
        return None
    if resp.status_code >= 400:
        logger.info("football_data.not_found", url=url[:80], status=resp.status_code)
        return None
    # Football-data.co.uk usa encoding latin-1
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    text = resp.content.decode("latin-1", errors="replace")
    cache_file.write_text(text, encoding="latin-1")
    logger.info("football_data.fetched", league=league_code, season=season, bytes=len(text))
    return text


def _parse_date(raw: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            dt = datetime.strptime(raw.strip(), fmt).replace(tzinfo=UTC)
            # La fuente a veces usa años 2-dígitos ambiguos; si <30 asumimos 20xx
            if dt.year < 1990:
                dt = dt.replace(year=dt.year + 100)
            return dt
        except ValueError:
            continue
    return None


def _parse_float_safe(raw: str | None) -> float | None:
    if raw is None or raw.strip() in ("", "-"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_csv_to_odds_rows(
    csv_text: str,
    *,
    league_slug: str,
    sport_code: str = "soccer",
    assumed_kickoff_hour: int = 20,
) -> list[dict[str, Any]]:
    """Parsea CSV completo → lista de dicts intermedios (no aún HistoricalOddsRow).

    El dict tiene todo lo necesario para:
      1. Crear/resolver match (home_name, away_name, start_time, home_score, away_score).
      2. Crear múltiples odds_history rows (una por bookmaker + momento: opening/closing).

    football-data.co.uk no trae hora exacta del kickoff. Asumimos kickoff por defecto
    a las 20:00 UTC (configurable). Si hay columna `Time`, se usa.
    """
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for raw in reader:
        date_raw = raw.get("Date") or ""
        start_time = _parse_date(date_raw)
        if start_time is None:
            continue
        # Si existe columna Time (HH:MM), usarla
        time_raw = (raw.get("Time") or "").strip()
        if time_raw:
            try:
                hh, mm = time_raw.split(":")[:2]
                start_time = start_time.replace(hour=int(hh), minute=int(mm))
            except ValueError:
                start_time = start_time.replace(hour=assumed_kickoff_hour)
        else:
            start_time = start_time.replace(hour=assumed_kickoff_hour)

        home = (raw.get("HomeTeam") or "").strip()
        away = (raw.get("AwayTeam") or "").strip()
        if not home or not away:
            continue

        home_goals = _parse_float_safe(raw.get("FTHG"))
        away_goals = _parse_float_safe(raw.get("FTAG"))

        # Odds por book × (home/draw/away)
        odds_by_book: dict[str, dict[str, float]] = {}
        for prefix, book_name in BOOK_MAP.items():
            oh = _parse_float_safe(raw.get(f"{prefix}H"))
            od = _parse_float_safe(raw.get(f"{prefix}D"))
            oa = _parse_float_safe(raw.get(f"{prefix}A"))
            if oh is not None and oa is not None:
                odds_by_book[book_name] = {"home": oh, "draw": od or 0.0, "away": oa}

        if not odds_by_book:
            continue

        rows.append(
            {
                "sport_code": sport_code,
                "league_slug": league_slug,
                "home_name": home,
                "away_name": away,
                "start_time": start_time,
                "home_score": int(home_goals) if home_goals is not None else None,
                "away_score": int(away_goals) if away_goals is not None else None,
                "odds_by_book": odds_by_book,
            }
        )
    return rows


def match_to_historical_odds_rows(
    parsed: dict[str, Any],
    *,
    match_id: int,
    opening_offset_hours: int = 24,
    closing_offset_minutes: int = 5,
) -> list[HistoricalOddsRow]:
    """Expande un match parseado a múltiples HistoricalOddsRow (validables).

    Opening: ts = start_time - 24h.
    Closing: solo para bookmakers con prefix *C (PSC → pinnacle_close). ts = start - 5min.
    """
    out: list[HistoricalOddsRow] = []
    start_time: datetime = parsed["start_time"]
    opening_ts = start_time - timedelta(hours=opening_offset_hours)
    closing_ts = start_time - timedelta(minutes=closing_offset_minutes)

    for book_name, outcomes in parsed["odds_by_book"].items():
        is_closing = book_name.endswith("_close")
        ts = closing_ts if is_closing else opening_ts
        # Si draw=0 (faltante), lo quitamos del dict
        outcomes_clean = {k: v for k, v in outcomes.items() if v > 0}
        out.append(
            HistoricalOddsRow(
                match_id=match_id,
                bookmaker=book_name.replace("_close", ""),
                market="h2h",
                outcomes_odds=outcomes_clean,
                ts=ts,
                start_time=start_time,
                is_closing=is_closing,
            )
        )
    return out


async def fetch_league_season(
    league_slug: str,
    season: int,
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Shortcut: descarga + parsea. Retorna lista de dicts match-level."""
    code = LEAGUE_CODES.get(league_slug)
    if code is None:
        msg = f"league_slug desconocido: {league_slug!r}"
        raise ValueError(msg)
    csv_text = await fetch_csv_raw(code, season, force_refresh=force_refresh)
    if csv_text is None:
        return []
    return parse_csv_to_odds_rows(csv_text, league_slug=league_slug)
