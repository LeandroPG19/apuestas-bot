#!/usr/bin/env bash
# Full retrain con stack Sprint 11+12+13 completo — máxima precisión.
#
# Re-entrena los 6 deportes + 10 ligas soccer con:
# - market_stacker (LGBM monotonic)
# - Elo features
# - Venn-Abers calibration (si n_per_class>=100)
# - StatsBomb features (soccer)
# - book_power wire
# - Historical features wire
#
# Uso:
#   bash scripts/full_retrain_sprint13.sh > /tmp/retrain_logs/full_retrain.log 2>&1 &

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

unset PYTHON_GIL
set -a
source .env
set +a
unset PYTHON_GIL

export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export PREFECT_API_URL="http://localhost:4200/api"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minio-admin"
export AWS_SECRET_ACCESS_KEY="change-me-minio-password"
export PYTHONPATH="src"

# Sprint 11+12+13 flags
export APUESTAS_USE_MARKET_STACKER=true
export APUESTAS_USE_FOCAL_LOSS=false  # LGBM custom obj tiene bug downstream
export APUESTAS_ENABLE_XT=true
export APUESTAS_ENABLE_NBA_CLUTCH=true
export APUESTAS_ENABLE_MLB_STUFF_PLUS=true
export APUESTAS_USE_BOOK_POWER=true
export APUESTAS_SPRINT11_SOFT_TAGS=true
export APUESTAS_USE_MODEL_HIERARCHY=true
export APUESTAS_FEATURE_MIN_COVERAGE=0.30

PY=.venv/bin/python
LOG="/tmp/retrain_logs/full_retrain.log"
mkdir -p /tmp/retrain_logs

echo "[$(date +%H:%M:%S)] === FULL RETRAIN Sprint 13 ===" | tee -a "$LOG"

# ─── NBA ─────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] NBA moneyline..." | tee -a "$LOG"
env -u PYTHON_GIL "$PY" -c "
import asyncio
from apuestas.ml.train_nba import train_nba, NBATrainConfig
async def main():
    cfg = NBATrainConfig(seasons=['2022-23','2023-24','2024-25'], target='win', n_trials=15)
    r = await train_nba(cfg)
    print(f'NBA: log_loss={r.holdout_log_loss:.4f} brier={r.holdout_brier:.4f} ece={r.holdout_ece:.4f}')
asyncio.run(main())
" 2>&1 | tee -a "$LOG"

# ─── MLB ─────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] MLB moneyline..." | tee -a "$LOG"
env -u PYTHON_GIL "$PY" -c "
import asyncio
from apuestas.ml.train_mlb import train_mlb, MLBTrainConfig
async def main():
    cfg = MLBTrainConfig(years=[2022,2023,2024], target='moneyline', n_trials=15)
    r = await train_mlb(cfg)
    print(f'MLB: log_loss={r.holdout_log_loss:.4f} brier={r.holdout_brier:.4f} ece={r.holdout_ece:.4f}')
asyncio.run(main())
" 2>&1 | tee -a "$LOG"

# ─── NFL ─────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] NFL ats..." | tee -a "$LOG"
env -u PYTHON_GIL "$PY" -c "
import asyncio
from apuestas.ml.train_nfl import train_nfl, NFLTrainConfig
async def main():
    cfg = NFLTrainConfig(seasons=['2019-20','2020-21','2021-22','2022-23'], n_trials=15)
    r = await train_nfl(cfg)
    print(f'NFL: log_loss={r.holdout_log_loss:.4f} brier={r.holdout_brier:.4f} ece={r.holdout_ece:.4f}')
asyncio.run(main())
" 2>&1 | tee -a "$LOG"

# ─── Soccer leagues (one per league, using existing train_soccer) ───
for LEAGUE_ID in 4 5 6 7 8 10 11 12 14 16; do
    echo "[$(date +%H:%M:%S)] Soccer league_${LEAGUE_ID}..." | tee -a "$LOG"
    env -u PYTHON_GIL "$PY" -c "
import asyncio
from apuestas.ml.train_soccer import train_soccer, SoccerTrainConfig
async def main():
    try:
        cfg = SoccerTrainConfig(league_id=${LEAGUE_ID}, seasons=['2022-2023','2023-2024','2024-2025'], n_trials=10)
        r = await train_soccer(cfg)
        print(f'Soccer L${LEAGUE_ID}: done')
    except Exception as e:
        print(f'Soccer L${LEAGUE_ID} SKIP: {str(e)[:120]}')
asyncio.run(main())
" 2>&1 | tee -a "$LOG"
done

# ─── Tennis (Sackmann) ──────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Tennis Sackmann..." | tee -a "$LOG"
env -u PYTHON_GIL "$PY" scripts/train_tennis_sackmann.py --tours atp,wta --since 2018 --n-trials 15 2>&1 | tee -a "$LOG"

# ─── Fit closing_line_predictor all sports ──────────────────────────
echo "[$(date +%H:%M:%S)] Fit closing_line_predictor all sports..." | tee -a "$LOG"
env -u PYTHON_GIL "$PY" scripts/fit_closing_line_predictor.py --all 2>&1 | tee -a "$LOG"

# ─── Refresh book_power_ratings ─────────────────────────────────────
env -u PYTHON_GIL "$PY" scripts/refresh_book_power_ratings.py 2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] === RETRAIN DONE ===" | tee -a "$LOG"
touch /tmp/retrain_logs/full_retrain_done.flag
