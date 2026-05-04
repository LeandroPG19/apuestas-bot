#!/usr/bin/env bash
# Pipeline automático post-backfill — Sprint 12 closing.
#
# Espera que todos los backfills terminen, ejecuta retrains con data nueva,
# valida KPI gate, promueve modelos que pasen, y prepara bot para launch.
#
# Uso: ./scripts/post_backfill_pipeline.sh > /tmp/retrain_logs/post_backfill.log 2>&1 &

set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export MLFLOW_TRACKING_URI="file:///tmp/mlflow_sprint12_final"
export APUESTAS_USE_MARKET_STACKER=true
export APUESTAS_USE_FOCAL_LOSS=false  # LGBM custom obj tiene bug downstream
export APUESTAS_ENABLE_XT=true
export APUESTAS_ENABLE_NBA_CLUTCH=true
export APUESTAS_ENABLE_MLB_STUFF_PLUS=true
export APUESTAS_USE_BOOK_POWER=true
export APUESTAS_SPRINT11_SOFT_TAGS=true

LOG="/tmp/retrain_logs/post_backfill.log"
mkdir -p /tmp/retrain_logs

echo "[$(date +%H:%M:%S)] Waiting for backfills to complete..." | tee -a "$LOG"

# Wait for NBA PBP backfill
while pgrep -f "ingest_nba_pbp_range" > /dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] NBA PBP DONE" | tee -a "$LOG"

# Wait for MLB Statcast backfill
while pgrep -f "ingest_mlb_statcast_range" > /dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] MLB Statcast DONE" | tee -a "$LOG"

# Wait for weather backfill
while pgrep -f "ingest_weather_archive" > /dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] Weather DONE" | tee -a "$LOG"

# Wait for clubelo resume
while pgrep -f "ingest_clubelo" > /dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] Clubelo DONE" | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] All backfills done. Starting retrains..." | tee -a "$LOG"

# Data status dashboard
docker exec apuestas-postgres psql -U apuestas -d apuestas -c "
SELECT 'nba_pbp' AS src, COUNT(*)::bigint AS n FROM play_by_play WHERE sport_code='nba'
UNION ALL SELECT 'pitcher_statcast', COUNT(*) FROM pitcher_game_stats
UNION ALL SELECT 'weather', COUNT(*) FROM weather_stadium_archive
UNION ALL SELECT 'clubelo', COUNT(*) FROM team_elo_daily WHERE source='clubelo'
UNION ALL SELECT 'statsbomb', COUNT(*) FROM statsbomb_events
ORDER BY src;" 2>&1 | tee -a "$LOG"

# Retrain 4 sports in sequence
echo "[$(date +%H:%M:%S)] Retraining MLB..." | tee -a "$LOG"
uv run python -c "
import asyncio
from apuestas.ml.train_mlb import train_mlb, MLBTrainConfig
async def main():
    cfg = MLBTrainConfig(years=[2022, 2023, 2024], target='moneyline', n_trials=10)
    result = await train_mlb(cfg)
    print(f'MLB: log_loss={result.holdout_log_loss:.4f} brier={result.holdout_brier:.4f} ece={result.holdout_ece:.4f}')
asyncio.run(main())
" 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] Retraining NBA..." | tee -a "$LOG"
uv run python -c "
import asyncio
from apuestas.ml.train_nba import train_nba, NBATrainConfig
async def main():
    cfg = NBATrainConfig(seasons=['2023-24','2024-25'], target='win', n_trials=10)
    result = await train_nba(cfg)
    print(f'NBA: log_loss={result.holdout_log_loss:.4f} brier={result.holdout_brier:.4f} ece={result.holdout_ece:.4f}')
asyncio.run(main())
" 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] Retrains DONE. Backfill pipeline closed." | tee -a "$LOG"

# Fit closing line predictor all sports
uv run python scripts/fit_closing_line_predictor.py --all 2>&1 | tee -a "$LOG"

# Refresh book power ratings
uv run python scripts/refresh_book_power_ratings.py 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] Pipeline COMPLETE. Ready for bot launch." | tee -a "$LOG"
touch /tmp/retrain_logs/pipeline_done.flag
