"""Loader de odds históricas NBA/NFL/NHL desde SportsBook Review archive.

SBR no expone API pública. Estrategia pragmática: usar el dataset público
comunitario **sportsref_odds** publicado por la comunidad quant en GitHub
(mirror estable de SBR + Vegas Insider + betting-data-nba) como CSVs.

Fuentes primarias utilizadas (en orden de preferencia):
  1. `kaggle.com/datasets/erichqiu/nba-odds-and-scores` — NBA 2008-2024
  2. `github.com/bttmly/nba` — archivo histórico con odds
  3. `github.com/ThomasMinetti/NHL-Betting-Data` — NHL 2010-2024
  4. `github.com/hvpkod/NFL-Data` — NFL 2007-2024 con closing lines

Como estos datasets no se pueden descargar vía API simple y cambian de ubicación,
este loader soporta dos modos:

- **local**: apunta a un CSV local (usuario descarga manualmente desde Kaggle)
- **github_raw**: descarga CSV desde GitHub raw content si la URL es estable

Columnas esperadas tras normalización:
| Col          | Meaning                             |
|--------------|-------------------------------------|
| date         | YYYY-MM-DD                          |
| home_team    | nombre canónico                     |
| away_team    | nombre canónico                     |
| home_score   | int o null                          |
| away_score   | int o null                          |
| home_ml      | moneyline home (decimal o americano)|
| away_ml      | moneyline away                      |
| home_spread  | spread home (ej. -3.5)              |
| away_spread  | spread away                         |
| total        | total points line                   |
| spread_home_odds  | odds decimal side                  |
| total_over_odds   | odds decimal over                  |
| book         | sportsbook origen (pinnacle, ws, etc.)|

Si el CSV de origen usa odds americanas (+150 / -110), se convierten a decimales.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from apuestas.obs.logging import get_logger
from apuestas.validators.historical_odds_integrity import HistoricalOddsRow

logger = get_logger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "apuestas" / "historical" / "sbr"

# Datasets comunitarios estables (raw GitHub links, validados con HEAD → 200 OK)
# flancast90/sportsbookreview-scraper publica JSON con scores + open/close ML +
# open/close spread + open/close total. Cobertura 2011-2021, ~7 MB cada sport.
GITHUB_DATASETS: dict[str, dict[str, str]] = {
    "nba": {
        "url": "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/nba_archive_10Y.json",
        "format": "sbr_json",
    },
    "nfl": {
        "url": "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/nfl_archive_10Y.json",
        "format": "sbr_json",
    },
    "nhl": {
        "url": "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/nhl_archive_10Y.json",
        "format": "sbr_json",
    },
}


def american_to_decimal(price: float | int | None) -> float | None:
    """Convierte odds americanas (-110, +150) a decimal. None si inválido."""
    if price is None:
        return None
    p = float(price)
    if p == 0 or (abs(p) < 100 and abs(p) > 10):
        # Valores raros tipo -50 no son odds americanas válidas
        return None
    if p > 0:
        return round(p / 100 + 1, 4)
    return round(100 / abs(p) + 1, 4)


def _cache_path(sport: str) -> Path:
    return _CACHE_DIR / f"{sport}.json"


async def fetch_sport_csv(
    sport: str,
    *,
    force_refresh: bool = False,
    local_path: Path | None = None,
    timeout: float = 120.0,
) -> str | None:
    """Descarga CSV del sport (o lee de local_path si se provee).

    Si el dataset GitHub no está disponible (404 / moved), retorna None.
    El caller debería fallback a otro método (ingesta live Pinnacle).
    """
    if local_path is not None and local_path.exists():
        return local_path.read_text(encoding="utf-8", errors="replace")

    cache_file = _cache_path(sport)
    if cache_file.exists() and cache_file.stat().st_size > 1000 and not force_refresh:
        return cache_file.read_text(encoding="utf-8", errors="replace")

    ds = GITHUB_DATASETS.get(sport)
    if ds is None:
        logger.info("sbr.sport_unsupported", sport=sport)
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(ds["url"])
    except httpx.HTTPError as exc:
        logger.info("sbr.http_error", url=ds["url"][:80], error=str(exc)[:80])
        return None
    if resp.status_code >= 400:
        logger.info("sbr.not_found", sport=sport, status=resp.status_code)
        return None

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text, encoding="utf-8")
    logger.info("sbr.fetched", sport=sport, bytes=len(resp.text))
    return resp.text


def _parse_date(raw: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw.strip()[:10], fmt).replace(tzinfo=UTC, hour=20)
        except ValueError:
            continue
    return None


def _parse_float_safe(raw: str | None) -> float | None:
    if raw is None or raw.strip() in ("", "-", "NA", "N/A", "null"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


_TEAM_NICK_TO_FULL_NBA: dict[str, str] = {
    "Hawks": "Atlanta Hawks",
    "Celtics": "Boston Celtics",
    "Nets": "Brooklyn Nets",
    "Hornets": "Charlotte Hornets",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Mavericks": "Dallas Mavericks",
    "Nuggets": "Denver Nuggets",
    "Pistons": "Detroit Pistons",
    "Warriors": "Golden State Warriors",
    "Rockets": "Houston Rockets",
    "Pacers": "Indiana Pacers",
    "Clippers": "Los Angeles Clippers",
    "Lakers": "Los Angeles Lakers",
    "Grizzlies": "Memphis Grizzlies",
    "Heat": "Miami Heat",
    "Bucks": "Milwaukee Bucks",
    "Timberwolves": "Minnesota Timberwolves",
    "Pelicans": "New Orleans Pelicans",
    "Knicks": "New York Knicks",
    "Thunder": "Oklahoma City Thunder",
    "Magic": "Orlando Magic",
    "76ers": "Philadelphia 76ers",
    "Suns": "Phoenix Suns",
    "Trail Blazers": "Portland Trail Blazers",
    "TrailBlazers": "Portland Trail Blazers",
    "Blazers": "Portland Trail Blazers",
    "Kings": "Sacramento Kings",
    "Spurs": "San Antonio Spurs",
    "Raptors": "Toronto Raptors",
    "Jazz": "Utah Jazz",
    "Wizards": "Washington Wizards",
}


def _resolve_full_name(nick: str, sport: str) -> str:
    """Map short nickname ('Lakers') → full ('Los Angeles Lakers')."""
    nick = nick.strip()
    if sport == "nba" and nick in _TEAM_NICK_TO_FULL_NBA:
        return _TEAM_NICK_TO_FULL_NBA[nick]
    return nick


def _parse_sbr_date_yyyymmdd(date_raw: Any) -> datetime | None:
    """flancast90 usa date como float 20210115.0 o str '20210115'."""
    if date_raw is None:
        return None
    try:
        s = str(int(float(date_raw))) if isinstance(date_raw, float | int | str) else ""
    except (TypeError, ValueError):
        return None
    if len(s) != 8:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=UTC, hour=20)
    except ValueError:
        return None


def parse_sbr_json(json_text: str, sport: str) -> list[dict[str, Any]]:
    """Parser del dataset flancast90/sportsbookreview-scraper (JSON).

    Esquema típico por item:
      season, date (float YYYYMMDD), home_team, away_team, home_final, away_final,
      home_close_ml, away_close_ml, home_open_spread, home_close_spread,
      open_over_under, close_over_under
    """
    rows: list[dict[str, Any]] = []
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("sbr_json.parse_fail", error=str(exc)[:120])
        return rows

    if not isinstance(data, list):
        return rows

    for item in data:
        if not isinstance(item, dict):
            continue
        start_time = _parse_sbr_date_yyyymmdd(item.get("date"))
        if start_time is None:
            continue

        home_raw = str(item.get("home_team") or "").strip()
        away_raw = str(item.get("away_team") or "").strip()
        if not home_raw or not away_raw:
            continue
        home = _resolve_full_name(home_raw, sport)
        away = _resolve_full_name(away_raw, sport)

        def _score_int(raw: Any) -> int | None:
            if raw in (None, "", "null"):
                return None
            try:
                return int(float(raw))
            except (TypeError, ValueError):
                return None

        home_score = _score_int(item.get("home_final"))
        away_score = _score_int(item.get("away_final"))

        odds_by_book: dict[str, dict[str, float]] = {}

        h_close_ml = american_to_decimal(_parse_float_safe(str(item.get("home_close_ml"))))
        a_close_ml = american_to_decimal(_parse_float_safe(str(item.get("away_close_ml"))))
        if h_close_ml and a_close_ml:
            odds_by_book["pinnacle_close"] = {"home": h_close_ml, "away": a_close_ml}

        h_open_ml = american_to_decimal(_parse_float_safe(str(item.get("home_open_ml"))))
        a_open_ml = american_to_decimal(_parse_float_safe(str(item.get("away_open_ml"))))
        if h_open_ml and a_open_ml:
            odds_by_book["pinnacle"] = {"home": h_open_ml, "away": a_open_ml}

        if not odds_by_book:
            continue

        rows.append(
            {
                "sport_code": sport,
                "home_name": home,
                "away_name": away,
                "start_time": start_time,
                "home_score": home_score,
                "away_score": away_score,
                "odds_by_book": odds_by_book,
            }
        )
    return rows


def parse_sport_csv(csv_text: str, sport: str) -> list[dict[str, Any]]:
    """Compat: parser dispatcher. Nuevo format 'sbr_json' usa parse_sbr_json."""
    ds = GITHUB_DATASETS.get(sport)
    if ds is None:
        return []
    if ds["format"] == "sbr_json":
        return parse_sbr_json(csv_text, sport)
    return []


def match_to_historical_odds_rows(
    parsed: dict[str, Any],
    *,
    match_id: int,
    opening_offset_hours: int = 12,
    closing_offset_minutes: int = 10,
) -> list[HistoricalOddsRow]:
    out: list[HistoricalOddsRow] = []
    start_time: datetime = parsed["start_time"]
    for book_name, outcomes in parsed["odds_by_book"].items():
        is_closing = book_name.endswith("_close")
        ts = (
            start_time - timedelta(minutes=closing_offset_minutes)
            if is_closing
            else start_time - timedelta(hours=opening_offset_hours)
        )
        out.append(
            HistoricalOddsRow(
                match_id=match_id,
                bookmaker=book_name.replace("_close", ""),
                market="h2h",
                outcomes_odds={k: v for k, v in outcomes.items() if v and v > 0},
                ts=ts,
                start_time=start_time,
                is_closing=is_closing,
            )
        )
    return out


async def fetch_sport(sport: str, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    csv_text = await fetch_sport_csv(sport, force_refresh=force_refresh)
    if csv_text is None:
        return []
    return parse_sport_csv(csv_text, sport)
