"""Entry-point one-shot for apuestas-lineup-scratch.service."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("PYTHON_GIL", None)

from apuestas.db import session_scope
from apuestas.ingest.lineup_scratch import mark_stale_picks_pre_kickoff


async def main():
    async with session_scope() as s:
        for sport in ("mlb", "nba"):
            try:
                n = await mark_stale_picks_pre_kickoff(s, sport, minutes_before=120)
                print(f"{sport}_lineup_scratched n={n}")
            except Exception as exc:
                print(f"{sport}_lineup_scratch_error: {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
