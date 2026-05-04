#!/usr/bin/env bash
# autopilot_post_training.sh — ejecuta pipeline completo post-trainings.
#
# Waits hasta que no queden procesos retrain_sport activos, luego:
#   1. Seed player_game_logs (NBA + NFL + MLB + NHL paralelo)
#   2. Compute player_stat_std desde logs
#   3. Bulk fit player_prop_distributions
#   4. Scrape referees recientes (Sofascore)
#   5. Archive injuries (Wayback Machine, 12 weeks)
#   6. Drift monitor + auto-retrain si drift detectado
#   7. Backtest NBA + Soccer con modelos nuevos
#
# Uso:
#   nohup bash scripts/autopilot_post_training.sh > logs/autopilot_post_training.log 2>&1 &

set -eu
cd "$(dirname "$0")/.."
mkdir -p logs/training logs

PY="${APUESTAS_VENV:-.venv}/bin/python"
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5434}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export PREFECT_API_URL="${PREFECT_API_URL:-http://localhost:4200/api}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minio-admin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-change-me-minio-password}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://localhost:9000}"
export PYTHONUNBUFFERED=1

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a logs/autopilot_post_training.log; }

telegram_notify() {
    msg="$1"
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] && \
        curl -sS -X POST \
          "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
          -d "chat_id=${TELEGRAM_CHAT_ID}" \
          -d "text=${msg}" >/dev/null 2>&1 || true
}

log "Waiting for all retrain_sport processes to complete..."
while pgrep -f "retrain_sport" >/dev/null 2>&1; do
    N=$(pgrep -f "retrain_sport" 2>/dev/null | wc -l || echo 0)
    log "  ${N} retrain procs still running..."
    sleep 60
done
log "✅ All trainings completed."
telegram_notify "✅ Trainings completados. Iniciando post-training pipeline."

# ─── Seed player_game_logs paralelo ───────────────────────────────────────
log "Step 1/7: Seed player_game_logs (NBA + NFL + MLB + NHL en paralelo)..."
$PY -m apuestas.scripts.seed_player_game_logs_nba --seasons "2023-24,2024-25" \
    > logs/seed_player_nba.log 2>&1 &
PID_PLA_NBA=$!
$PY -m apuestas.scripts.seed_player_game_logs_nfl --seasons "2023,2024" \
    > logs/seed_player_nfl.log 2>&1 &
PID_PLA_NFL=$!
$PY -m apuestas.scripts.seed_player_game_logs_mlb --seasons "2023,2024" \
    > logs/seed_player_mlb.log 2>&1 &
PID_PLA_MLB=$!
$PY -m apuestas.scripts.seed_player_game_logs_nhl --seasons "20232024,20242025" \
    > logs/seed_player_nhl.log 2>&1 &
PID_PLA_NHL=$!
wait $PID_PLA_NBA $PID_PLA_NFL $PID_PLA_MLB $PID_PLA_NHL 2>/dev/null || true
log "✅ Seed player_game_logs completado."

# ─── Compute player_stat_std ──────────────────────────────────────────────
log "Step 2/7: Compute player_stat_std desde logs..."
$PY -m apuestas.scripts.compute_player_stat_std --all \
    >> logs/compute_player_stat_std.log 2>&1 || \
    log "⚠  compute_player_stat_std fallo"

# ─── Bulk fit player_prop_distributions ───────────────────────────────────
log "Step 3/7: Bulk fit player_prop_distributions..."
$PY -m apuestas.scripts.bulk_fit_player_props --all \
    >> logs/bulk_fit_props.log 2>&1 || \
    log "⚠  bulk_fit_player_props fallo"

# ─── Scrape referees (Sofascore) ──────────────────────────────────────────
log "Step 4/7: Scrape referees (Sofascore)..."
$PY -m apuestas.ingest.referee_scraper --sport soccer --days 7 \
    >> logs/referee_scraper.log 2>&1 || \
    log "⚠  referee_scraper fallo"

# ─── Archive injuries (Wayback Machine) ───────────────────────────────────
log "Step 5/7: Archive injuries (Wayback, 12 weeks NBA/NFL/MLB/NHL)..."
$PY -m apuestas.ingest.injury_archive --sport nba,nfl,mlb,nhl --weeks 12 \
    >> logs/injury_archive.log 2>&1 || \
    log "⚠  injury_archive fallo"

# ─── Drift monitor + auto-retrain ─────────────────────────────────────────
log "Step 6/7: Drift monitor + auto-retrain..."
$PY -m apuestas.monitors.drift_monitor \
    >> logs/drift_monitor.log 2>&1 || \
    log "⚠  drift_monitor fallo"

# ─── Backtest con modelos nuevos ──────────────────────────────────────────
CURRENT_YEAR=$(date +%Y)
BACKTEST_SEASONS="$((CURRENT_YEAR-3)),$((CURRENT_YEAR-2)),$((CURRENT_YEAR-1))"
log "Step 7/7: Backtest NBA + Soccer seasons=$BACKTEST_SEASONS..."
mkdir -p reports
$PY -m apuestas.scripts.run_backtest --sport nba --seasons "$BACKTEST_SEASONS" \
    --output reports/backtest_nba.json --min-sharpe 1.0 > logs/backtest_nba.log 2>&1 &
$PY -m apuestas.scripts.run_backtest --sport soccer --seasons "$BACKTEST_SEASONS" \
    --output reports/backtest_soccer.json --min-sharpe 1.0 > logs/backtest_soccer.log 2>&1 &
wait

log "✅ autopilot_post_training DONE"
telegram_notify "✅ [AUTOPILOT] Post-training pipeline completa. Verifica logs."
