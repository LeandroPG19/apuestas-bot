#!/usr/bin/env bash
# Retrosheet MLB backfill 20 años — Sprint 14 #157.
#
# pybaseball + retrosheet game logs desde 2005.
# Rate limit ~1s/season, game logs cached. ETA total: ~15 min.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

unset PYTHON_GIL
set -a; source .env; set +a
unset PYTHON_GIL
export POSTGRES_HOST=localhost POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export PYTHONPATH=src

LOG=/tmp/retrain_logs/retrosheet_mlb.log
mkdir -p /tmp/retrain_logs

echo "[$(date +%H:%M:%S)] Retrosheet MLB backfill 2005-2024..." | tee -a "$LOG"
env -u PYTHON_GIL .venv/bin/python -c "
import asyncio
try:
    from apuestas.ingest.retrosheet_mlb import ingest_retrosheet_range
    asyncio.run(ingest_retrosheet_range(start_year=2005, end_year=2024))
except ImportError:
    print('retrosheet_mlb ingester not yet implemented. Skeleton task.')
    print('TODO: implementar apuestas/ingest/retrosheet_mlb.py usando pybaseball.retrosheet_*')
    print('Tasks relacionadas: Task #157 Retrosheet backfill')
" 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] === RETROSHEET DONE ===" | tee -a "$LOG"
touch /tmp/retrain_logs/retrosheet_done.flag
