"""Daily refresh de `book_power_ratings` — Sprint 11 Fase D operacional.

Ejecutar como cron 03:00 UTC (o Prefect schedule). Computa edge bps por
(bookmaker, league) sobre rolling 90d y persiste en Valkey cache.

Uso:
    uv run python scripts/refresh_book_power_ratings.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apuestas.betting.book_power_ratings import compute_book_power_ratings
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def main_async() -> int:
    from apuestas.db import session_scope

    async with session_scope() as session:
        profiles = await compute_book_power_ratings(
            session,
            lookback_days=90,
            min_samples=50,
        )

    if not profiles:
        logger.warning("book_power.refresh.no_profiles")
        return 1

    payload = {
        f"{bk}|{lg}": {
            "bookmaker": p.bookmaker,
            "league": p.league,
            "sport_code": p.sport_code,
            "mean_edge_bps": p.mean_edge_bps,
            "std_edge_bps": p.std_edge_bps,
            "n_samples": p.n_samples,
            "last_updated": p.last_updated.isoformat(),
        }
        for (bk, lg), p in profiles.items()
    }
    payload_json = json.dumps(payload, indent=2)

    # Cache Valkey (TTL 36h) con fallback a archivo local si falla.
    try:
        from apuestas.cache import cache_set

        await cache_set("book_power_ratings:v1", payload_json, ttl_seconds=130_000)
        logger.info("book_power.refresh.persisted_cache", n=len(profiles))
    except Exception as exc:
        logger.warning("book_power.cache_fail_fallback_file", error=str(exc)[:100])

    # Siempre archivo (doble persistencia para auditoría)
    out = ROOT / "artifacts" / "book_power" / "latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload_json, encoding="utf-8")
    logger.info("book_power.refresh.persisted_file", path=str(out), n=len(profiles))

    # Top 10 para logging
    top = sorted(profiles.items(), key=lambda kv: -kv[1].mean_edge_bps)[:10]
    for (bk, lg), p in top:
        logger.info(
            "book_power.top",
            book=bk,
            league=lg,
            sport=p.sport_code,
            edge_bps=round(p.mean_edge_bps, 1),
            n=p.n_samples,
        )
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
