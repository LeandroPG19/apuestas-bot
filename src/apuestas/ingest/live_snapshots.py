"""Live match snapshots ingester — Fase 1 wire #154 support.

Captura snapshot per-minute durante matches soccer in_progress.
Populate `match_live_snapshots` consumida por `flows/soccer_live_2h.py`.

Tabla match_live_snapshots (auto-create):
  match_id bigint
  snapshot_minute smallint
  home_score_1h int, away_score_1h int
  home_xg_1h numeric, away_xg_1h numeric
  home_shots_1h int, away_shots_1h int
  captured_at timestamptz

Fuente primaria: The Odds API /sports/soccer_*/events/{id} con live scores.
Fuente xG: Understat si disponible (scrape con CF bypass).

Trigger: timer 5min durante ventana ±2h kickoff.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from apuestas.db import session_scope
from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


async def ensure_snapshot_table() -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS match_live_snapshots (
                  id bigserial PRIMARY KEY,
                  match_id bigint NOT NULL,
                  snapshot_minute smallint NOT NULL,
                  home_score_1h int,
                  away_score_1h int,
                  home_xg_1h numeric(6,3),
                  away_xg_1h numeric(6,3),
                  home_shots_1h int,
                  away_shots_1h int,
                  captured_at timestamptz DEFAULT now()
                )
                """
            )
        )
        await s.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_mls_match_minute "
                "ON match_live_snapshots (match_id, snapshot_minute)"
            )
        )


async def capture_live_soccer_snapshots() -> int:
    """Scan matches in kickoff window ±2h, capture score + shots proxy.

    Sin xG data: usa shots_on_target como proxy (cada shot ~0.1 xG).
    """
    await ensure_snapshot_table()
    now = datetime.now(tz=UTC)
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT m.id, m.start_time, m.external_id
                    FROM matches m
                    WHERE m.sport_code='soccer'
                      AND m.start_time BETWEEN :start AND :end
                      AND (m.status IS NULL OR m.status IN ('scheduled','in_progress','live','ht'))
                    """
                ),
                {
                    "start": now - timedelta(hours=2),
                    "end": now + timedelta(hours=1),
                },
            )
        ).fetchall()

    logger.info("live_snapshots.scan", n_matches=len(rows))
    n_captured = 0
    # Placeholder: real implementation requires Sofascore API or livescore scraper
    # For now, this is a skeleton that would call:
    #   score, shots = await _fetch_live_score(match_external_id)
    # For each match, inserts snapshot at current minute.
    return n_captured


async def main():
    n = await capture_live_soccer_snapshots()
    print(f"Live snapshots captured: {n}")


if __name__ == "__main__":
    asyncio.run(main())
