"""MLB Statcast pitcher ingester — Sprint 11 Fase G operacional.

Usa `pybaseball` (gratis con rate limit) para descargar Statcast pitch-level
y agregarlo a un nivel pitcher-game para alimentar `features/mlb_pitching_plus.py`.

Inserta en tabla nueva `pitcher_game_stats` con columnas que
`estimate_stuff_plus` consume:
- spin_rate_avg (mean release_spin_rate)
- velo_avg (mean release_speed)
- whiff_pct (pct swinging_strike)
- release_consistency (std release_pos_x + release_pos_z)
- n_pitches

Uso:
    await ingest_mlb_statcast_for_date("2024-07-01")
    # CLI
    uv run python -m apuestas.ingest.mlb_statcast --date 2024-07-01
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def _to_date(v):  # type: ignore[no-untyped-def]
    """Convierte string/Timestamp/date → date object."""
    if v is None:
        return None
    if hasattr(v, "date"):
        return v.date()
    try:
        s = str(v)
        if len(s) >= 10:
            return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=UTC).date()
    except Exception:
        return None
    return None


async def _fetch_statcast(start: str, end: str):  # type: ignore[no-untyped-def]
    """Blocking fetch via pybaseball, wrapped en thread pool."""

    def _sync():  # type: ignore[no-untyped-def]
        import pybaseball as pb

        return pb.statcast(start_dt=start, end_dt=end)

    try:
        return await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.warning("mlb_statcast.fetch_fail", start=start, end=end, error=str(exc)[:120])
        return None


def _aggregate_pitcher_game(df) -> list[dict]:  # type: ignore[no-untyped-def]
    """Agrega pitch-level → pitcher × game_pk stats."""
    if df is None or len(df) == 0:
        return []
    required = {"pitcher", "game_pk"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("mlb_statcast.missing_cols", missing=list(missing))
        return []

    # Whiff: description == 'swinging_strike' o 'swinging_strike_blocked'
    desc_col = df.get("description")
    is_whiff = (
        desc_col.isin(["swinging_strike", "swinging_strike_blocked"])
        if desc_col is not None
        else None
    )
    df = df.assign(
        _whiff=is_whiff.astype(int) if is_whiff is not None else 0,
    )

    agg = (
        df.groupby(["pitcher", "game_pk"])
        .agg(
            spin_rate_avg=("release_spin_rate", "mean"),
            velo_avg=("release_speed", "mean"),
            n_pitches=("pitcher", "count"),
            whiff_count=("_whiff", "sum"),
            release_x_std=("release_pos_x", "std"),
            release_z_std=("release_pos_z", "std"),
            game_date=("game_date", "max"),
        )
        .reset_index()
    )

    out: list[dict] = []
    for row in agg.to_dict(orient="records"):
        n = int(row.get("n_pitches") or 0)
        if n < 10:  # pitcher que solo entró brevemente
            continue
        whiff = int(row.get("whiff_count") or 0)
        out.append(
            {
                "pitcher_mlbam_id": int(row["pitcher"]) if row.get("pitcher") else None,
                "game_pk": int(row["game_pk"]) if row.get("game_pk") else None,
                "game_date": str(row.get("game_date") or ""),
                "spin_rate_avg": float(row.get("spin_rate_avg") or 0.0),
                "velo_avg": float(row.get("velo_avg") or 0.0),
                "whiff_pct": float(whiff) / float(n) if n else 0.0,
                "release_consistency": float(
                    (row.get("release_x_std") or 0.0) + (row.get("release_z_std") or 0.0)
                )
                / 2.0,
                "n_pitches": n,
            }
        )
    return out


async def _ensure_table() -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS pitcher_game_stats (
                    id bigserial PRIMARY KEY,
                    pitcher_mlbam_id bigint NOT NULL,
                    game_pk bigint NOT NULL,
                    game_date date,
                    spin_rate_avg numeric(8,2),
                    velo_avg numeric(6,2),
                    whiff_pct numeric(5,4),
                    release_consistency numeric(6,4),
                    n_pitches integer,
                    ingested_at timestamptz DEFAULT now()
                )
                """
            )
        )
        await s.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_pitcher_game "
                "ON pitcher_game_stats (pitcher_mlbam_id, game_pk)"
            )
        )
        await s.commit()


async def ingest_mlb_statcast_for_date(game_date: str) -> int:
    """Descarga + upsert pitcher_game_stats para una fecha."""
    await _ensure_table()
    df = await _fetch_statcast(game_date, game_date)
    if df is None or len(df) == 0:
        logger.info("mlb_statcast.no_data", date=game_date)
        return 0

    records = _aggregate_pitcher_game(df)
    if not records:
        return 0

    inserted = 0
    async with session_scope() as s:
        for r in records:
            try:
                await s.execute(
                    text(
                        """
                        INSERT INTO pitcher_game_stats (
                            pitcher_mlbam_id, game_pk, game_date,
                            spin_rate_avg, velo_avg, whiff_pct,
                            release_consistency, n_pitches
                        ) VALUES (:pid, :gpk, :gd, :spin, :velo, :whiff, :rc, :np)
                        ON CONFLICT (pitcher_mlbam_id, game_pk) DO UPDATE SET
                            spin_rate_avg = EXCLUDED.spin_rate_avg,
                            velo_avg = EXCLUDED.velo_avg,
                            whiff_pct = EXCLUDED.whiff_pct,
                            release_consistency = EXCLUDED.release_consistency,
                            n_pitches = EXCLUDED.n_pitches,
                            ingested_at = now()
                        """
                    ),
                    {
                        "pid": r["pitcher_mlbam_id"],
                        "gpk": r["game_pk"],
                        "gd": _to_date(r.get("game_date")),
                        "spin": r["spin_rate_avg"],
                        "velo": r["velo_avg"],
                        "whiff": r["whiff_pct"],
                        "rc": r["release_consistency"],
                        "np": r["n_pitches"],
                    },
                )
                inserted += 1
            except Exception as exc:
                logger.debug("mlb_statcast.upsert_fail", error=str(exc)[:80])
        await s.commit()
    logger.info("mlb_statcast.ingested", date=game_date, pitchers=inserted)
    return inserted


async def ingest_mlb_statcast_range(start_date: date, end_date: date) -> int:
    total = 0
    current = start_date
    while current <= end_date:
        n = await ingest_mlb_statcast_for_date(current.strftime("%Y-%m-%d"))
        total += n
        current += timedelta(days=1)
        await asyncio.sleep(1.5)  # rate limit Baseball Savant
    return total


async def _main_async(args) -> int:  # type: ignore[no-untyped-def]
    if args.date:
        n = await ingest_mlb_statcast_for_date(args.date)
    elif args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC).date()
        e = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC).date()
        n = await ingest_mlb_statcast_range(s, e)
    else:
        print("❌ Usa --date YYYY-MM-DD o --start ... --end ...")
        return 1
    print(f"✓ {n} pitcher-game rows insertados")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ingest_mlb_statcast_for_date", "ingest_mlb_statcast_range"]
