"""Bulk ingest Jeff Sackmann tennis data — ATP + WTA 1968-presente.

Fuente: https://github.com/JeffSackmann/tennis_atp + tennis_wta
Licencia: CC BY-NC-SA 4.0.

Descarga atp_matches_YYYY.csv y wta_matches_YYYY.csv con stats por partido:
- serve/return stats (ace, df, 1st_in, 1st_won, 2nd_won, bp_saved, bp_faced)
- ranking pre-match
- tourney_level + surface
- score

Volumen: ~200k partidos ATP + ~150k WTA.

Uso:
    uv run python scripts/ingest_sackmann_tennis.py --since 2018
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

BASE_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
BASE_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"


async def _fetch_year(client: httpx.AsyncClient, base: str, year: int) -> pd.DataFrame | None:
    url = f"{base}/{_year_file_name(base, year)}"
    try:
        r = await client.get(url, timeout=60.0, follow_redirects=True)
        if r.status_code != 200:
            return None
        return pd.read_csv(io.BytesIO(r.content))
    except Exception as exc:
        logger.debug("sackmann.fetch_fail", year=year, error=str(exc)[:100])
        return None


def _year_file_name(base: str, year: int) -> str:
    if "tennis_atp" in base:
        return f"atp_matches_{year}.csv"
    return f"wta_matches_{year}.csv"


async def _ensure_table() -> None:
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    async with session_scope() as s:
        await s.execute(
            _text(
                """
                CREATE TABLE IF NOT EXISTS tennis_matches_sackmann (
                    id bigserial PRIMARY KEY,
                    tour text NOT NULL,
                    tourney_id text,
                    tourney_name text,
                    tourney_date date,
                    surface text,
                    draw_size integer,
                    tourney_level text,
                    match_num integer,
                    winner_id integer,
                    winner_name text,
                    winner_rank integer,
                    winner_age numeric(4,1),
                    loser_id integer,
                    loser_name text,
                    loser_rank integer,
                    loser_age numeric(4,1),
                    score text,
                    best_of integer,
                    round text,
                    minutes integer,
                    w_ace integer, w_df integer, w_svpt integer,
                    w_1stIn integer, w_1stWon integer, w_2ndWon integer,
                    w_SvGms integer, w_bpSaved integer, w_bpFaced integer,
                    l_ace integer, l_df integer, l_svpt integer,
                    l_1stIn integer, l_1stWon integer, l_2ndWon integer,
                    l_SvGms integer, l_bpSaved integer, l_bpFaced integer,
                    ingested_at timestamptz DEFAULT now(),
                    UNIQUE (tour, tourney_id, match_num)
                )
                """
            )
        )
        await s.execute(
            _text(
                "CREATE INDEX IF NOT EXISTS idx_tennis_sackmann_date "
                "ON tennis_matches_sackmann (tourney_date DESC)"
            )
        )
        await s.commit()


async def ingest_year(tour: str, year: int) -> int:
    from sqlalchemy import text as _text

    from apuestas.db import session_scope

    base = BASE_ATP if tour == "atp" else BASE_WTA
    async with httpx.AsyncClient() as client:
        df = await _fetch_year(client, base, year)
    if df is None or len(df) == 0:
        return 0

    # Normalizar tourney_date
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(
            df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
        )

    columns_to_insert = [
        "tourney_id",
        "tourney_name",
        "tourney_date",
        "surface",
        "draw_size",
        "tourney_level",
        "match_num",
        "winner_id",
        "winner_name",
        "winner_rank",
        "winner_age",
        "loser_id",
        "loser_name",
        "loser_rank",
        "loser_age",
        "score",
        "best_of",
        "round",
        "minutes",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]
    existing_cols = [c for c in columns_to_insert if c in df.columns]

    inserted = 0
    async with session_scope() as session:
        # Truncar año para idempotencia
        await session.execute(
            _text(
                "DELETE FROM tennis_matches_sackmann "
                "WHERE tour = :t AND EXTRACT(YEAR FROM tourney_date) = :y"
            ),
            {"t": tour, "y": year},
        )
        for _, row in df.iterrows():
            values = {"tour": tour}
            for c in existing_cols:
                v = row.get(c)
                if pd.isna(v):
                    values[c] = None
                elif c == "tourney_date":
                    values[c] = v.date() if hasattr(v, "date") else None
                else:
                    values[c] = v
            placeholders = ", ".join(f":{c}" for c in ["tour"] + existing_cols)
            cols_str = ", ".join(["tour"] + existing_cols)
            try:
                await session.execute(
                    _text(
                        f"INSERT INTO tennis_matches_sackmann ({cols_str}) "
                        f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
                    ),
                    values,
                )
                inserted += 1
            except Exception:
                pass
        await session.commit()
    logger.info("sackmann.year_done", tour=tour, year=year, matches=inserted)
    return inserted


async def main_async(since: int) -> int:
    await _ensure_table()
    total = 0
    end = datetime.now(tz=UTC).year
    for tour in ("atp", "wta"):
        for year in range(since, end + 1):
            try:
                n = await ingest_year(tour, year)
                total += n
            except Exception as exc:
                logger.warning("sackmann.year_fail", tour=tour, year=year, error=str(exc)[:100])
    logger.info("sackmann.done", total=total)
    print(f"✓ Inserted {total} Sackmann tennis matches")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=2018)
    args = parser.parse_args()
    return asyncio.run(main_async(args.since))


if __name__ == "__main__":
    raise SystemExit(main())
