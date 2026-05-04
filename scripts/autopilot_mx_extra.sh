#!/usr/bin/env bash
# Bloques extra MX + MLB que faltaban en el autopilot original.
# 100% ESCALABLE: todas las temporadas se calculan dinámicamente del año del sistema.
# Al correr en 2027 sembrará hasta 2027; en 2030 hasta 2030. Cero hardcoded.

set -uo pipefail
cd "$(dirname "$0")/.."

export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export POSTGRES_PASSWORD=change-me-to-long-random-string
export POSTGRES_USER=apuestas
export POSTGRES_DB=apuestas
export APUESTAS_ENV=local
export APUESTAS_LOG_LEVEL=INFO
export PYTHONPATH=src
unset PYTHON_GIL

PY=.venv/bin/python
LOG=logs/autopilot_mx_extra.log
mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ─── Cálculo dinámico de temporadas (ZERO hardcoded) ───────────────
CURRENT_YEAR=$(date '+%Y')
LOOKBACK_YEARS=${LOOKBACK_YEARS:-7}  # default: 7 temporadas atrás
CURRENT_SEASON_NBA=$((CURRENT_YEAR - 1))  # NBA 2025-26 se expresa como season=2025

# Construye CSV de temporadas dinámicamente
build_seasons_csv() {
    local count="$1"
    local end_year="$2"
    local start=$((end_year - count + 1))
    local csv=""
    for ((y=start; y<=end_year; y++)); do
        csv="${csv}${y},"
    done
    echo "${csv%,}"
}

MLB_SEASONS=$(build_seasons_csv "$LOOKBACK_YEARS" "$CURRENT_YEAR")
LIGA_MX_SEASONS=$(build_seasons_csv 6 "$CURRENT_YEAR")
NBA_SEASONS=$(build_seasons_csv 2 "$CURRENT_SEASON_NBA")
SOCCER_CURRENT_SEASONS=$(build_seasons_csv 2 "$((CURRENT_YEAR - 1))")
TENNIS_CURRENT=$(build_seasons_csv 2 "$CURRENT_YEAR")

# Token/chat desde .env (no hardcoded)
source .env 2>/dev/null || true
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

telegram_notify() {
    local msg="$1"
    [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ] && return
    $PY -c "
import httpx, os
t = os.environ.get('TELEGRAM_BOT_TOKEN', '')
c = os.environ.get('TELEGRAM_CHAT_ID', '')
if t and c:
    try:
        httpx.post(f'https://api.telegram.org/bot{t}/sendMessage',
                   json={'chat_id': int(c), 'text': '''$msg'''}, timeout=10)
    except Exception as e: print(f'tg fail: {e}')
" 2>&1 | tail -1
}

echo "[$(ts)] autopilot_mx_extra START (year=$CURRENT_YEAR, lookback=$LOOKBACK_YEARS)" | tee -a "$LOG"
echo "[$(ts)] seasons: MLB=$MLB_SEASONS | LigaMX=$LIGA_MX_SEASONS | NBA=$NBA_SEASONS | soccer_current=$SOCCER_CURRENT_SEASONS | tennis=$TENNIS_CURRENT" | tee -a "$LOG"

telegram_notify "🇲🇽 Sembrando MX extra + MLB hasta $CURRENT_YEAR (paralelo al principal)..."

# MLB (US, hasta temporada actual dinámica)
echo "[$(ts)] seeding MLB $MLB_SEASONS..." | tee -a "$LOG"
$PY -m apuestas.scripts.seed_historical --sport mlb --seasons "$MLB_SEASONS" >> "$LOG" 2>&1 && \
    echo "[$(ts)] ✅ MLB done" | tee -a "$LOG" || \
    echo "[$(ts)] ⚠  MLB fallo" | tee -a "$LOG"

# Liga MX hasta temporada actual
echo "[$(ts)] seeding Liga MX $LIGA_MX_SEASONS..." | tee -a "$LOG"
$PY -m apuestas.scripts.seed_historical --sport soccer --league liga_mx --seasons "$LIGA_MX_SEASONS" >> "$LOG" 2>&1 && \
    echo "[$(ts)] ✅ Liga MX done" | tee -a "$LOG" || \
    echo "[$(ts)] ⚠  Liga MX fallo" | tee -a "$LOG"

# NBA temporadas actuales dinámicas
echo "[$(ts)] seeding NBA $NBA_SEASONS (temporadas actuales)..." | tee -a "$LOG"
$PY -m apuestas.scripts.seed_historical --sport nba --seasons "$NBA_SEASONS" >> "$LOG" 2>&1 && \
    echo "[$(ts)] ✅ NBA actual done" | tee -a "$LOG" || \
    echo "[$(ts)] ⚠  NBA actual fallo" | tee -a "$LOG"

# Soccer Big-5 temporadas actuales
echo "[$(ts)] seeding soccer Big-5 $SOCCER_CURRENT_SEASONS (actual)..." | tee -a "$LOG"
$PY -m apuestas.scripts.seed_historical \
    --sport soccer-odds --seasons "$SOCCER_CURRENT_SEASONS" \
    --league epl,la_liga,bundesliga,serie_a,ligue_1 >> "$LOG" 2>&1 && \
    echo "[$(ts)] ✅ Soccer Big-5 actual done" | tee -a "$LOG" || \
    echo "[$(ts)] ⚠  Soccer actual fallo" | tee -a "$LOG"

# Tennis ATP/WTA actuales
echo "[$(ts)] seeding tennis $TENNIS_CURRENT..." | tee -a "$LOG"
$PY -m apuestas.scripts.seed_historical --sport tennis --seasons "$TENNIS_CURRENT" --league atp,wta >> "$LOG" 2>&1 && \
    echo "[$(ts)] ✅ tennis actual done" | tee -a "$LOG" || \
    echo "[$(ts)] ⚠  tennis actual fallo" | tee -a "$LOG"

# Dispara historical_backfill (trae hoy mismo)
echo "[$(ts)] triggering historical_backfill..." | tee -a "$LOG"
systemctl --user start apuestas-historical-backfill.service 2>&1 | tee -a "$LOG"

# Catchup final
echo "[$(ts)] catchup_flow refresh final..." | tee -a "$LOG"
$PY -m apuestas.flows.catchup >> "$LOG" 2>&1 || echo "[$(ts)] ⚠  catchup fallo" | tee -a "$LOG"

echo "[$(ts)] autopilot_mx_extra DONE" | tee -a "$LOG"
telegram_notify "✅ MX + MLB extra completados. Año actual ${CURRENT_YEAR} cubierto. Check logs/autopilot_mx_extra.log"
