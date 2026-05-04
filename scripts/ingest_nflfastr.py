"""Bulk ingest nflfastR — play-by-play NFL 1999-presente.

Fuente: https://github.com/nflverse/nflverse-data
Licencia: MIT.

Descarga parquet directo de releases. Agrega a `nfl_epa_plays` con
EPA/CPOE/success_rate por play.

Uso:
    uv run python scripts/ingest_nflfastr.py --since 2018
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

BASE = "https://github.com/nflverse/nflverse-data/releases/download/pbp"


async def _fetch_parquet(client: httpx.AsyncClient, season: int):  # type: ignore[no-untyped-def]
    url = f"{BASE}/play_by_play_{season}.parquet"
    try:
        r = await client.get(url, timeout=120.0, follow_redirects=True)
        if r.status_code != 200:
            logger.warning("nflfastr.fetch_fail", season=season, status=r.status_code)
            return None
        import pandas as pd

        return pd.read_parquet(io.BytesIO(r.content))
    except Exception as exc:
        logger.warning("nflfastr.fetch_exc", season=season, error=str(exc)[:100])
        return None


async def ingest_season(season: int) -> int:
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    async with httpx.AsyncClient() as client:
        df = await _fetch_parquet(client, season)
    if df is None or len(df) == 0:
        return 0

    # Filtrar cols relevantes
    cols = [
        "game_id",
        "play_id",
        "posteam",
        "defteam",
        "down",
        "ydstogo",
        "yardline_100",
        "play_type",
        "epa",
        "cpoe",
        "success",
    ]
    existing = [c for c in cols if c in df.columns]
    df = df[existing].copy()

    # Bulk insert con COPY via psycopg2 sería 10x más rápido, pero
    # mantenemos asyncpg con chunks de 5000.
    total = 0
    async with session_scope() as session:
        await session.execute(
            _text("DELETE FROM nfl_epa_plays WHERE game_id_nflverse LIKE :prefix"),
            {"prefix": f"{season}%"},
        )
        chunk_size = 5000
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            rows = []
            for _, r in chunk.iterrows():
                rows.append(
                    {
                        "game_id": str(r.get("game_id") or ""),
                        "play_id": int(r.get("play_id") or 0),
                        "offense": str(r.get("posteam") or ""),
                        "defense": str(r.get("defteam") or ""),
                        "down": int(r["down"])
                        if r.get("down") is not None and not _isnan(r.get("down"))
                        else None,
                        "ydstogo": int(r["ydstogo"])
                        if r.get("ydstogo") is not None and not _isnan(r.get("ydstogo"))
                        else None,
                        "yl100": int(r["yardline_100"])
                        if r.get("yardline_100") is not None and not _isnan(r.get("yardline_100"))
                        else None,
                        "pt": str(r.get("play_type") or "")[:32],
                        "epa": float(r["epa"])
                        if r.get("epa") is not None and not _isnan(r.get("epa"))
                        else None,
                        "cpoe": float(r["cpoe"])
                        if r.get("cpoe") is not None and not _isnan(r.get("cpoe"))
                        else None,
                        "suc": int(r["success"])
                        if r.get("success") is not None and not _isnan(r.get("success"))
                        else None,
                    }
                )
            if rows:
                await session.execute(
                    _text(
                        """
                        INSERT INTO nfl_epa_plays (
                            game_id_nflverse, play_id, offense_team, defense_team,
                            down, ydstogo, yardline_100, play_type, epa, cpoe, success
                        ) VALUES (:game_id, :play_id, :offense, :defense,
                                  :down, :ydstogo, :yl100, :pt, :epa, :cpoe, :suc)
                        ON CONFLICT (game_id_nflverse, play_id) DO NOTHING
                        """
                    ),
                    rows,
                )
                total += len(rows)
        await session.commit()
    logger.info("nflfastr.season_done", season=season, plays=total)
    return total


def _isnan(v) -> bool:  # type: ignore[no-untyped-def]
    try:
        import math

        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


async def main_async(since: int) -> int:
    total = 0
    end = datetime.now(tz=UTC).year
    for season in range(since, end + 1):
        try:
            n = await ingest_season(season)
            total += n
        except Exception as exc:
            logger.warning("nflfastr.season_fail", season=season, error=str(exc)[:100])
    logger.info("nflfastr.done", total_plays=total)
    print(f"✓ Inserted {total} NFL plays")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2018)
    args = parser.parse_args()
    return asyncio.run(main_async(args.since))


if __name__ == "__main__":
    raise SystemExit(main())
