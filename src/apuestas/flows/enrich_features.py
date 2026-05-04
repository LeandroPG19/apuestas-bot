"""Prefect flow: enrichment de features Sprint 11 — daily job.

Orquesta:
1. `refresh_book_power_ratings` (diario 03:00 UTC) — 90d rolling edge bps.
2. `ingest_nba_pbp` (si API disponible) — llena `play_by_play` NBA.
3. `ingest_mlb_statcast` (si API disponible) — spin rate, velo, whiff%.
4. `ingest_soccer_shots` (si API disponible) — possession_pct, shots_on_target.

Los ingesters específicos viven en `apuestas.ingest.*`; este flow es el
scheduler. Si alguna fase falla (rate limit, 404, etc.), continúa con las
demás sin abortar — fail-silent por diseño.

Uso:
    uv run python -m apuestas.flows.enrich_features
"""

from __future__ import annotations

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def refresh_book_power() -> bool:
    """Wrapper del script refresh_book_power_ratings para uso en flow."""
    try:
        from apuestas.betting.book_power_ratings import compute_book_power_ratings
        from apuestas.db import session_scope

        async with session_scope() as session:
            profiles = await compute_book_power_ratings(
                session,
                lookback_days=90,
                min_samples=50,
            )
        if not profiles:
            logger.warning("enrich.book_power.no_profiles")
            return False

        # Persistir doble: cache + archivo
        import json
        from pathlib import Path

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

        try:
            from apuestas.cache import cache_set

            await cache_set("book_power_ratings:v1", payload_json, ttl_seconds=130_000)
        except Exception as exc:
            logger.warning("enrich.book_power.cache_fail", error=str(exc)[:80])

        out = Path(__file__).resolve().parents[3] / "artifacts" / "book_power" / "latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload_json, encoding="utf-8")
        logger.info("enrich.book_power.refreshed", n=len(profiles))
        return True
    except Exception as exc:
        logger.warning("enrich.book_power.fail", error=str(exc)[:100])
        return False


async def enrich_nba_clutch(*, days_back: int = 1) -> bool:
    """Descarga PBP NBA de los últimos N días y puebla `play_by_play`."""
    try:
        from datetime import UTC, datetime, timedelta

        from apuestas.ingest.nba_pbp import ingest_nba_pbp_range

        end = datetime.now(UTC).date()
        start = end - timedelta(days=days_back)
        total = await ingest_nba_pbp_range(start, end)
        logger.info("enrich.nba_clutch.done", events=total, days=days_back)
        return total > 0
    except Exception as exc:
        logger.warning("enrich.nba_clutch.fail", error=str(exc)[:120])
        return False


async def enrich_mlb_statcast(*, days_back: int = 1) -> bool:
    """Descarga Statcast pitcher-game stats últimos N días."""
    try:
        from datetime import UTC, datetime, timedelta

        from apuestas.ingest.mlb_statcast import ingest_mlb_statcast_range

        end = datetime.now(UTC).date()
        start = end - timedelta(days=days_back)
        total = await ingest_mlb_statcast_range(start, end)
        logger.info("enrich.mlb_statcast.done", rows=total, days=days_back)
        return total > 0
    except Exception as exc:
        logger.warning("enrich.mlb_statcast.fail", error=str(exc)[:120])
        return False


async def enrich_soccer_shots() -> bool:
    """Attempt FBref match shots/possession.

    Sólo ejecuta si env `APUESTAS_ENABLE_SOCCER_SHOTS=true` porque
    requiere scrape FBref con riesgo de rate-limit Cloudflare.
    """
    import os as _os

    if _os.environ.get("APUESTAS_ENABLE_SOCCER_SHOTS", "false").lower() != "true":
        logger.info("enrich.soccer_shots.disabled_by_env")
        return False
    logger.info(
        "enrich.soccer_shots.opt_in_only",
        note="usar ingest_soccer_shots_for_match_fbref con match_fbref_id específico",
    )
    return False


async def enrich_features_flow() -> dict[str, bool]:
    """Ejecuta toda la cadena. No aborta si alguna falla."""
    results = {
        "book_power": await refresh_book_power(),
        "nba_clutch": await enrich_nba_clutch(),
        "mlb_statcast": await enrich_mlb_statcast(),
        "soccer_shots": await enrich_soccer_shots(),
    }
    logger.info("enrich.flow.done", **dict(results))
    return results


def main() -> int:
    import asyncio

    results = asyncio.run(enrich_features_flow())
    # Exit 0 si al menos 1 tuvo éxito (book_power es el crítico)
    return 0 if results.get("book_power") else 1


if __name__ == "__main__":
    raise SystemExit(main())
