#!/usr/bin/env bash
# NBA PBP backfill extendido 2021-22 a 2023-24 — Sprint 14 #151.
#
# Ejecuta ingest_nba_pbp_range para temporadas completas.
# Rate limit nba_api ~1.2s/game = ~8min per 400-game season.
# ETA total: ~25 min para 3 seasons.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

unset PYTHON_GIL
set -a; source .env; set +a
unset PYTHON_GIL
export POSTGRES_HOST=localhost POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export PYTHONPATH=src

LOG=/tmp/retrain_logs/nba_pbp_extended.log
mkdir -p /tmp/retrain_logs

echo "[$(date +%H:%M:%S)] NBA PBP backfill 2021-22..." | tee -a "$LOG"
env -u PYTHON_GIL .venv/bin/python -c "
import asyncio
from apuestas.ingest.nba_pbp import ingest_nba_pbp_range
asyncio.run(ingest_nba_pbp_range(start_date='2021-10-19', end_date='2022-06-16'))
" 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] NBA PBP backfill 2022-23..." | tee -a "$LOG"
env -u PYTHON_GIL .venv/bin/python -c "
import asyncio
from apuestas.ingest.nba_pbp import ingest_nba_pbp_range
asyncio.run(ingest_nba_pbp_range(start_date='2022-10-18', end_date='2023-06-12'))
" 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] NBA PBP backfill 2023-24..." | tee -a "$LOG"
env -u PYTHON_GIL .venv/bin/python -c "
import asyncio
from apuestas.ingest.nba_pbp import ingest_nba_pbp_range
asyncio.run(ingest_nba_pbp_range(start_date='2023-10-24', end_date='2024-06-17'))
" 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] === NBA PBP EXTENDED DONE ===" | tee -a "$LOG"
touch /tmp/retrain_logs/nba_pbp_extended_done.flag
