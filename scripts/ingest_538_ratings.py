"""Bulk ingest FiveThirtyEight archives — Elo/SPI/RAPTOR ratings.

Fuente: https://github.com/fivethirtyeight/data
Licencia: CC BY 4.0.

Datasets:
- NBA RAPTOR (1976-2022, actualizado hasta cese de 538)
- NFL Elo (1920-2022)
- Soccer SPI (2016-2022)

Inserta en `power_rankings_external` con source='538_spi'|'538_elo_nfl'|'538_raptor'.

Uso:
    uv run python scripts/ingest_538_ratings.py
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

URLS = {
    "538_spi_soccer": "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv",
    "538_nfl_elo": "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv",
    "538_raptor_nba": "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/modern_RAPTOR_by_team.csv",
}


async def _fetch_csv(client: httpx.AsyncClient, url: str) -> pd.DataFrame | None:
    try:
        r = await client.get(url, timeout=60.0, follow_redirects=True)
        if r.status_code != 200:
            logger.warning("538.fetch_fail", url=url[-60:], status=r.status_code)
            return None
        return pd.read_csv(io.BytesIO(r.content))
    except Exception as exc:
        logger.warning("538.fetch_exc", url=url[-60:], error=str(exc)[:100])
        return None


async def _resolve_team_id_soccer(session, team_name: str, league: str) -> int | None:  # type: ignore[no-untyped-def]
    from sqlalchemy import text as _text

    try:
        r = await session.execute(
            _text(
                """
                SELECT DISTINCT t.id FROM teams t
                JOIN matches m ON (m.home_team_id=t.id OR m.away_team_id=t.id)
                WHERE m.sport_code='soccer' AND LOWER(t.name) LIKE :p
                LIMIT 1
                """
            ),
            {"p": f"%{team_name.lower()[:12]}%"},
        )
        row = r.first()
        return int(row[0]) if row else None
    except Exception:
        return None


async def _resolve_team_id_nfl(session, team_abbr: str) -> int | None:  # type: ignore[no-untyped-def]
    from sqlalchemy import text as _text

    try:
        r = await session.execute(
            _text(
                "SELECT t.id FROM teams t "
                "JOIN matches m ON (m.home_team_id=t.id OR m.away_team_id=t.id) "
                "WHERE m.sport_code='nfl' AND (UPPER(t.short_name)=:abbr OR UPPER(t.name) LIKE :pattern) "
                "LIMIT 1"
            ),
            {"abbr": team_abbr.upper(), "pattern": f"%{team_abbr.upper()}%"},
        )
        row = r.first()
        return int(row[0]) if row else None
    except Exception:
        return None


async def ingest_spi_soccer() -> int:
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    async with httpx.AsyncClient() as client:
        df = await _fetch_csv(client, URLS["538_spi_soccer"])
    if df is None or len(df) == 0:
        return 0
    from datetime import UTC, datetime

    today = datetime.now(tz=UTC).date()
    total = 0
    async with session_scope() as session:
        for _, row in df.iterrows():
            name = str(row.get("name") or "").strip()
            league = str(row.get("league") or "").strip()
            rank = row.get("rank")
            spi = row.get("spi")
            if not name or spi is None or pd.isna(spi):
                continue
            team_id = await _resolve_team_id_soccer(session, name, league)
            if team_id is None:
                continue
            try:
                await session.execute(
                    _text(
                        """
                        INSERT INTO power_rankings_external (
                            team_id, sport_code, rating_date, source, rating, rank
                        ) VALUES (:tid, 'soccer', :d, '538_spi', :r, :rk)
                        ON CONFLICT (team_id, source, rating_date)
                        DO UPDATE SET rating = EXCLUDED.rating, rank = EXCLUDED.rank
                        """
                    ),
                    {
                        "tid": team_id,
                        "d": today,
                        "r": float(spi),
                        "rk": int(rank) if rank is not None else None,
                    },
                )
                total += 1
            except Exception as exc:
                logger.debug("538.spi_insert_fail", error=str(exc)[:80])
        await session.commit()
    logger.info("538.spi_done", rows=total)
    return total


async def ingest_nfl_elo() -> int:
    """NFL Elo: cada row es un game con elo_home/away pre-game. Extraemos
    ratings por team×season promediando últimos 16 juegos."""
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    async with httpx.AsyncClient() as client:
        df = await _fetch_csv(client, URLS["538_nfl_elo"])
    if df is None or len(df) == 0:
        return 0

    # Expandir por game: row home y row away
    rows: list[dict] = []
    for _, row in df.iterrows():
        date = row.get("date")
        if pd.isna(date):
            continue
        for side in ("team1", "team2"):
            elo_col = "elo1_pre" if side == "team1" else "elo2_pre"
            team = row.get(side)
            elo = row.get(elo_col)
            if pd.isna(team) or pd.isna(elo):
                continue
            rows.append(
                {"team_abbr": str(team), "date": pd.to_datetime(date).date(), "elo": float(elo)}
            )

    if not rows:
        return 0

    total = 0
    async with session_scope() as session:
        seen: set[tuple] = set()
        for r in rows:
            key = (r["team_abbr"], r["date"])
            if key in seen:
                continue
            seen.add(key)
            team_id = await _resolve_team_id_nfl(session, r["team_abbr"])
            if team_id is None:
                continue
            try:
                await session.execute(
                    _text(
                        """
                        INSERT INTO power_rankings_external (
                            team_id, sport_code, rating_date, source, rating
                        ) VALUES (:tid, 'nfl', :d, '538_elo_nfl', :r)
                        ON CONFLICT (team_id, source, rating_date)
                        DO UPDATE SET rating = EXCLUDED.rating
                        """
                    ),
                    {"tid": team_id, "d": r["date"], "r": r["elo"]},
                )
                total += 1
            except Exception:
                pass
        await session.commit()
    logger.info("538.nfl_elo_done", rows=total)
    return total


async def ingest_raptor_nba() -> int:
    """NBA RAPTOR per team-season promedio."""
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    async with httpx.AsyncClient() as client:
        df = await _fetch_csv(client, URLS["538_raptor_nba"])
    if df is None or len(df) == 0:
        return 0

    # Agregar a nivel team-season: team_id → promedio raptor_total por season
    total = 0
    async with session_scope() as session:
        # Por team, season: promedio raptor_total de todos los jugadores
        grouped = df.groupby(["team", "season"])[["raptor_total", "war_total"]].mean().reset_index()
        for _, row in grouped.iterrows():
            team_abbr = str(row.get("team"))
            season = int(row.get("season"))
            raptor = row.get("raptor_total")
            if pd.isna(raptor):
                continue
            # Fecha: último día de la season (ej 2022 → 2022-04-10)
            import datetime as _dt

            rating_date = _dt.date(season, 4, 10)
            try:
                r = await session.execute(
                    _text(
                        "SELECT t.id FROM teams t "
                        "JOIN matches m ON (m.home_team_id=t.id OR m.away_team_id=t.id) "
                        "WHERE m.sport_code='nba' AND UPPER(t.short_name)=:a LIMIT 1"
                    ),
                    {"a": team_abbr.upper()},
                )
                row2 = r.first()
                if row2 is None:
                    continue
                team_id = int(row2[0])
                await session.execute(
                    _text(
                        """
                        INSERT INTO power_rankings_external (
                            team_id, sport_code, rating_date, source, rating
                        ) VALUES (:tid, 'nba', :d, '538_raptor', :r)
                        ON CONFLICT (team_id, source, rating_date)
                        DO UPDATE SET rating = EXCLUDED.rating
                        """
                    ),
                    {"tid": team_id, "d": rating_date, "r": float(raptor)},
                )
                total += 1
            except Exception:
                pass
        await session.commit()
    logger.info("538.raptor_done", rows=total)
    return total


async def main_async() -> int:
    n1 = await ingest_spi_soccer()
    n2 = await ingest_nfl_elo()
    n3 = await ingest_raptor_nba()
    print(f"✓ 538 ratings: spi={n1} nfl_elo={n2} raptor={n3}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
