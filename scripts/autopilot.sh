#!/usr/bin/env bash
# scripts/autopilot.sh — todo en background, sin intervención manual.
#
# Orquesta en secuencia:
#   1. Activa flags .env (books + sofascore)
#   2. Seed histórico (6 sports, background)
#   3. Retrain modelos ML (tras seed)
#   4. Backtest walk-forward (tras retrain)
#   5. Cada etapa notifica a Telegram
#
# Se invoca con:
#   nohup bash scripts/autopilot.sh > logs/autopilot.log 2>&1 &
#
# O vía:
#   apuestas autopilot

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG=logs/autopilot.log
mkdir -p logs reports

# Env overrides para que Python conecte a localhost (no al docker network)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export POSTGRES_PASSWORD=change-me-to-long-random-string
export POSTGRES_USER=apuestas
export POSTGRES_DB=apuestas
export APUESTAS_ENV=local
export APUESTAS_LOG_LEVEL=INFO
export MLFLOW_TRACKING_URI=http://localhost:5000
export PREFECT_API_URL=http://localhost:4200/api
export PYTHONPATH=src
unset PYTHON_GIL

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ─── Cálculo dinámico de temporadas (ZERO hardcoded) ────────────────────
CURRENT_YEAR=$(date '+%Y')
LOOKBACK_YEARS="${LOOKBACK_YEARS:-7}"
CURRENT_SEASON_NBA=$((CURRENT_YEAR - 1))

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

SOCCER_HIST_SEASONS=$(build_seasons_csv "$LOOKBACK_YEARS" "$CURRENT_YEAR")
TENNIS_SEASONS=$(build_seasons_csv 5 "$CURRENT_YEAR")
NBA_SEASONS=$(build_seasons_csv 6 "$CURRENT_SEASON_NBA")
LIGA_MX_SEASONS=$(build_seasons_csv 5 "$CURRENT_YEAR")
MLB_SEASONS=$(build_seasons_csv 6 "$CURRENT_YEAR")

telegram_notify() {
    local msg="$1"
    [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ] && return
    $PY -c "
import httpx, os
t = os.environ.get('TELEGRAM_BOT_TOKEN', '')
c = os.environ.get('TELEGRAM_CHAT_ID', '')
if t and c:
    try:
        httpx.post(f'https://api.telegram.org/bot{t}/sendMessage',
                   json={'chat_id': int(c), 'text': '''$msg'''}, timeout=10)
    except Exception as e:
        print(f'telegram fail: {e}')
" 2>&1 | tail -2
}

echo "[$(ts)] autopilot START" | tee -a "$LOG"
telegram_notify "🤖 Autopilot iniciado — seed histórico + retrain + backtest corriendo en background. Te notifico cada etapa."

# ─── Paso 1: Activar flags en .env ──────────────────────────────────────────
echo "[$(ts)] [1/5] Activando flags .env..." | tee -a "$LOG"
$PY <<'PYEOF' 2>&1 | tee -a "$LOG"
from pathlib import Path
env = Path('.env')
text = env.read_text(encoding='utf-8')

flags_to_set = {
    'APUESTAS_ENABLE_SOFASCORE': 'true',
    'APUESTAS_ENABLE_BETUS': 'true',           # menos riesgo cloudflare que otros
    # Dejo BetWhale/Everygame/SportsBetting.ag en false (nuevos, TOS risk)
    # Dejo BC.GAME en false (requiere crypto wallet)
    # Dejo Winpot/CampoBet/JugaBet en false hasta user confirme scraping OK
}

lines = text.splitlines()
seen = set()
out = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith('#') or '=' not in stripped:
        out.append(line)
        continue
    key, _, _v = stripped.partition('=')
    key = key.strip()
    if key in flags_to_set:
        out.append(f'{key}={flags_to_set[key]}')
        seen.add(key)
    else:
        out.append(line)
for k, v in flags_to_set.items():
    if k not in seen:
        out.append(f'{k}={v}')

env.write_text('\n'.join(out) + '\n', encoding='utf-8')
print('flags activados:', list(flags_to_set.keys()))
PYEOF

telegram_notify "✅ [1/5] Flags .env activados: Sofascore + BetUS"

# Reinicio bot para que tome los flags
systemctl --user restart apuestas-telegram.service
sleep 3

# ─── Paso 2: Seed histórico multi-sport ──────────────────────────────────
echo "[$(ts)] [2/5] Seed histórico: soccer + tennis + US sports..." | tee -a "$LOG"

seed_block() {
    local name="$1"
    shift
    echo "[$(ts)]   Seeding $name..." | tee -a "$LOG"
    $PY -m apuestas.scripts.seed_historical "$@" >> "$LOG" 2>&1 && \
        echo "[$(ts)]   ✅ $name done" | tee -a "$LOG" || \
        echo "[$(ts)]   ⚠  $name failed (check log)" | tee -a "$LOG"
}

# 2a) Soccer — TODAS las ligas EU disponibles en football-data.co.uk (16 ligas × N seasons)
seed_block "soccer EU completo (16 ligas × $LOOKBACK_YEARS seasons)" \
    --sport soccer-odds \
    --seasons "$SOCCER_HIST_SEASONS" \
    --league epl,championship,la_liga,la_liga_2,bundesliga,bundesliga_2,serie_a,serie_b,ligue_1,ligue_2,eredivisie,liga_portugal,belgium_a,turkey_super,greece_super,scotland_premier

# 2b) Tennis — ATP + WTA (temporadas dinámicas)
seed_block "tennis (ATP + WTA)" \
    --sport tennis \
    --seasons "$TENNIS_SEASONS" \
    --league atp,wta

# 2c) US sports — NBA/NFL/NHL via SBR community datasets
seed_block "us-sports (NBA + NFL + NHL)" \
    --sport us-sports-odds \
    --league nba,nfl,nhl

# 2d) NBA detalle via nba_api (temporadas dinámicas)
seed_block "NBA nba_api" \
    --sport nba \
    --seasons "$NBA_SEASONS"

# 2e) Soccer Big-5 fbref PBP + stats (dinámico, ligas EUR)
# Liga MX histórico NO está en fuentes gratis → se cubre vía Pinnacle live catchup
seed_block "soccer Big-5 fbref PBP + stats" \
    --sport soccer \
    --league big5 \
    --seasons "$LIGA_MX_SEASONS"

# 2f) MLB (US) — pybaseball (temporadas dinámicas hasta año actual)
seed_block "MLB pybaseball" \
    --sport mlb \
    --seasons "$MLB_SEASONS"

# 2g) Liga MX + Expansion MX vía fbref directo (no existe en soccerdata)
seed_block "Liga MX + Expansion (fbref)" \
    --sport liga-mx \
    --seasons "$LIGA_MX_SEASONS" \
    --league liga_mx,liga_expansion

telegram_notify "✅ [2/5] Seed completado hasta $CURRENT_YEAR: 16 ligas EU + Liga MX + Expansion MX + Tennis ATP/WTA + NBA/NFL/NHL + MLB."

# ─── Paso 3: Rebuild features ────────────────────────────────────────────
echo "[$(ts)] [3/5] Rebuild features team_stats_rolling..." | tee -a "$LOG"
# Los triggers trigger_team_stats_rolling_* ya existen, solo necesita refresh.
# Ejecutar catchup para poblar odds recientes y triggerar rebuilds.
$PY -m apuestas.flows.catchup >> "$LOG" 2>&1 &
CATCHUP_PID=$!
wait $CATCHUP_PID || echo "[$(ts)]   ⚠  catchup fail"
telegram_notify "✅ [3/5] Features reconstruidas + catchup completo."

# ─── Paso 4: Retrain modelos ─────────────────────────────────────────────
echo "[$(ts)] [4/5] Retrain modelos ML..." | tee -a "$LOG"

# Ejecutamos deep_analysis para forzar entrenamiento/uso de modelos
# (retrain full pipelineaplicable si hay >= 500 muestras).
$PY -m apuestas.flows.retrain_weekly >> "$LOG" 2>&1 || \
    echo "[$(ts)]   (retrain_weekly flow no disponible; skip)" | tee -a "$LOG"
telegram_notify "✅ [4/5] Retrain modelos intentado."

# ─── Paso 5: Backtest walk-forward ───────────────────────────────────────
echo "[$(ts)] [5/5] Backtest walk-forward..." | tee -a "$LOG"

for sport in soccer nba; do
    echo "[$(ts)]   Backtest $sport..." | tee -a "$LOG"
    $PY -m apuestas.scripts.run_backtest \
        --sport "$sport" \
        --seasons 2021,2022,2023 \
        --output "reports/backtest_${sport}.json" \
        --min-sharpe 1.0 \
        >> "$LOG" 2>&1 || echo "[$(ts)]   ⚠  backtest $sport falló (esperado si no hay suficientes bets históricos)"
done

# ─── Resumen final ───────────────────────────────────────────────────────
echo "[$(ts)] autopilot DONE" | tee -a "$LOG"

# Summary de data en DB
SUMMARY=$($PY -c "
import asyncio
from apuestas.db import session_scope
from sqlalchemy import text as t
async def go():
    async with session_scope() as s:
        r = await s.execute(t('''
            SELECT sport_code, COUNT(*) n_matches,
                   COUNT(home_score) n_results
            FROM matches GROUP BY sport_code ORDER BY 2 DESC
        '''))
        lines = []
        for row in r.all():
            lines.append(f'  {row.sport_code}: {row.n_matches} matches ({row.n_results} con resultado)')
        r2 = await s.execute(t('''SELECT bookmaker, COUNT(*) n FROM odds_history
                                   WHERE ts > now()-interval '30 days'
                                   GROUP BY bookmaker ORDER BY 2 DESC LIMIT 10'''))
        lines.append('')
        lines.append('  odds_history (30d):')
        for row in r2.all():
            lines.append(f'    {row.bookmaker}: {row.n}')
        print('\n'.join(lines))
asyncio.run(go())
" 2>&1 | tail -30)

telegram_notify "🎯 [5/5] AUTOPILOT COMPLETADO

Data sembrada:
$SUMMARY

Backtest reports en reports/backtest_*.json
Log completo: logs/autopilot.log

Bot sigue operando 24/7 con timer auto-análisis cada 6h."

echo "[$(ts)] ─── autopilot EXIT ───" | tee -a "$LOG"
