"""Loader de odds históricas tennis desde tennis-data.co.uk.

Fuente: http://tennis-data.co.uk/alldata.php
Cobertura: ATP desde 2000, WTA desde 2007. Un Excel/CSV por tour × temporada.

Formato disponible: xls (xlsx) por defecto en la fuente. Para este loader usamos
el mirror CSV: `<tour>/<year>/<year>.csv` publicado por mirrors académicos.
Alternativa: descargar xlsx con openpyxl si llegara a ser necesario.

Columnas estándar (tras normalización):
| Columna   | Significado                                       |
|-----------|---------------------------------------------------|
| ATP/WTA   | Tour                                              |
| Location  | Ciudad torneo                                      |
| Tournament| Nombre                                             |
| Date      | Fecha partido (DD/MM/YYYY)                        |
| Series    | Grand Slam, Masters 1000, International, etc.      |
| Court     | Outdoor/Indoor                                     |
| Surface   | Hard/Clay/Grass/Carpet                             |
| Round     | Ronda                                              |
| Best of   | 3 o 5 sets                                         |
| Winner    | Nombre jugador ganador                             |
| Loser     | Nombre jugador perdedor                            |
| WRank     | Ranking ganador                                    |
| LRank     | Ranking perdedor                                   |
| Wsets     | Sets ganados por Winner                            |
| Lsets     | Sets ganados por Loser                             |
| PSW       | Pinnacle odds Winner                               |
| PSL       | Pinnacle odds Loser                                |
| B365W     | Bet365 odds Winner                                 |
| B365L     | Bet365 odds Loser                                  |
| MaxW/MaxL | Max odds del mercado                               |
| AvgW/AvgL | Average odds                                       |

En tennis NO hay "home/away" — siempre es h2h player1 vs player2 con dos odds
(Winner/Loser determinado post-match).
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

# tennis-data.co.uk usa xlsx; descargamos el fichero y lo parseamos
# Sin embargo xls puede requerir openpyxl. Usamos el endpoint que sí expone CSV:
BASE_URL = "http://www.tennis-data.co.uk"
TOURS = ("atp", "wta")

BOOK_MAP: dict[str, str] = {
    "PS": "pinnacle",
    "B365": "bet365",
    "EX": "betfair_exchange",
    "LB": "ladbrokes",
    "SJ": "stan_james",
    "Max": "market_max",
    "Avg": "market_avg",
}

_CACHE_DIR = Path.home() / ".cache" / "apuestas" / "historical" / "tennis-data"


def _cache_path(tour: str, season: int) -> Path:
    return _CACHE_DIR / tour / f"{season}.xlsx"


async def fetch_season_xlsx(
    tour: str,
    season: int,
    *,
    force_refresh: bool = False,
    timeout: float = 60.0,
) -> bytes | None:
    """Descarga xlsx de tenis con cache local. Retorna bytes o None si 404."""
    if tour not in TOURS:
        msg = f"tour debe ser 'atp' o 'wta', got {tour!r}"
        raise ValueError(msg)

    cache_file = _cache_path(tour, season)
    if cache_file.exists() and cache_file.stat().st_size > 1000 and not force_refresh:
        return cache_file.read_bytes()

    # tennis-data.co.uk publica: /{year}/{year}.xlsx (ATP) o /{year}w/{year}.xlsx (WTA).
    # Sufijo `w` en directorio WTA (verificado vía alldata.php curl HEAD → 200 OK).
    suffix = "" if tour == "atp" else "w"
    url = f"{BASE_URL}/{season}{suffix}/{season}.xlsx"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        logger.info("tennis_data.http_error", url=url[:80], error=str(exc)[:80])
        return None
    if resp.status_code >= 400:
        logger.info("tennis_data.not_found", url=url[:80], status=resp.status_code)
        return None

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(resp.content)
    logger.info("tennis_data.fetched", tour=tour, season=season, bytes=len(resp.content))
    return resp.content


def xlsx_to_csv_text(xlsx_bytes: bytes) -> str:
    """Convierte xlsx a CSV texto usando openpyxl (ya en dependencias)."""
    try:
        import openpyxl  # type: ignore[import-untyped]
    except ImportError:
        msg = "openpyxl no instalado. Añadir a pyproject.toml"
        raise RuntimeError(msg) from None

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in ws.iter_rows(values_only=True):
        writer.writerow(["" if v is None else str(v) for v in row])
    return buf.getvalue()


def _parse_date(raw: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw.strip()[:10], fmt).replace(tzinfo=UTC, hour=12)
        except ValueError:
            continue
    return None


def _parse_float_safe(raw: str | None) -> float | None:
    if raw is None or raw.strip() in ("", "-", "N/A"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_csv_to_match_rows(
    csv_text: str,
    *,
    tour: str,
) -> list[dict[str, Any]]:
    """Convierte CSV a match-level dicts para luego insertar."""
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for raw in reader:
        start_time = _parse_date(raw.get("Date", ""))
        if start_time is None:
            continue
        winner = (raw.get("Winner") or "").strip()
        loser = (raw.get("Loser") or "").strip()
        if not winner or not loser:
            continue

        w_sets = _parse_float_safe(raw.get("Wsets"))
        l_sets = _parse_float_safe(raw.get("Lsets"))

        odds_by_book: dict[str, dict[str, float]] = {}
        for prefix, book_name in BOOK_MAP.items():
            ow = _parse_float_safe(raw.get(f"{prefix}W"))
            ol = _parse_float_safe(raw.get(f"{prefix}L"))
            if ow is not None and ol is not None:
                # Usamos "winner"/"loser" como outcomes para h2h (post-match)
                # Downstream convertirá a home/away según convención del bot.
                odds_by_book[book_name] = {"winner": ow, "loser": ol}

        if not odds_by_book:
            continue

        rows.append(
            {
                "sport_code": "tennis",
                "tour": tour,
                "tournament": (raw.get("Tournament") or "").strip(),
                "surface": (raw.get("Surface") or "").strip().lower(),
                "series": (raw.get("Series") or "").strip(),
                "round": (raw.get("Round") or "").strip(),
                "best_of": int(raw.get("Best of", 3) or 3),
                "winner_name": winner,
                "loser_name": loser,
                "winner_rank": _parse_float_safe(raw.get("WRank")),
                "loser_rank": _parse_float_safe(raw.get("LRank")),
                "start_time": start_time,
                "winner_sets": int(w_sets) if w_sets is not None else None,
                "loser_sets": int(l_sets) if l_sets is not None else None,
                "odds_by_book": odds_by_book,
            }
        )
    return rows


def match_to_historical_odds_rows(
    parsed: dict[str, Any],
    *,
    match_id: int,
    opening_offset_hours: int = 6,
) -> list[HistoricalOddsRow]:
    """Tennis las odds se publican horas (no días) antes — opening = 6h antes."""
    out: list[HistoricalOddsRow] = []
    start_time: datetime = parsed["start_time"]
    opening_ts = start_time - timedelta(hours=opening_offset_hours)
    for book_name, outcomes in parsed["odds_by_book"].items():
        out.append(
            HistoricalOddsRow(
                match_id=match_id,
                bookmaker=book_name,
                market="h2h",
                outcomes_odds={k: v for k, v in outcomes.items() if v > 0},
                ts=opening_ts,
                start_time=start_time,
                is_closing=False,
            )
        )
    return out


async def fetch_tour_season(
    tour: str,
    season: int,
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    xlsx = await fetch_season_xlsx(tour, season, force_refresh=force_refresh)
    if xlsx is None:
        return []
    csv_text = xlsx_to_csv_text(xlsx)
    if not csv_text.strip():
        return []
    return parse_csv_to_match_rows(csv_text, tour=tour)
