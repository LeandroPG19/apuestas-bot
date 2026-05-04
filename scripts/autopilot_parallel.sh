#!/usr/bin/env bash
# autopilot_parallel.sh — versión paralela con 6 seeds simultáneos.
# Reemplaza scripts/autopilot.sh. Acelera ~4-6x el tiempo total.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
mkdir -p logs reports

export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export POSTGRES_PASSWORD=change-me-to-long-random-string
export POSTGRES_USER=apuestas
export POSTGRES_DB=apuestas
export APUESTAS_ENV=local
export APUESTAS_LOG_LEVEL=WARNING  # reduce verbosity → más velocidad I/O logs
export MLFLOW_TRACKING_URI=http://localhost:5000
export PREFECT_API_URL=http://localhost:4200/api
export PYTHONPATH=src
unset PYTHON_GIL

source .env 2>/dev/null || true

# Override post-source: .env define POSTGRES_HOST=postgres (docker hostname)
# pero corremos desde host → localhost + port expuesto.
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:${POSTGRES_PASSWORD:-change-me-to-long-random-string}@localhost:5434/apuestas"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

CURRENT_YEAR=$(date '+%Y')
LOOKBACK_YEARS="${LOOKBACK_YEARS:-7}"
CURRENT_SEASON_NBA=$((CURRENT_YEAR - 1))

build_seasons_csv() {
    local count="$1" end_year="$2"
    local start=$((end_year - count + 1))
    local csv=""
    for ((y=start; y<=end_year; y++)); do csv="${csv}${y},"; done
    echo "${csv%,}"
}

SOCCER_SEASONS=$(build_seasons_csv "$LOOKBACK_YEARS" "$CURRENT_YEAR")
TENNIS_SEASONS=$(build_seasons_csv 5 "$CURRENT_YEAR")
NBA_SEASONS=$(build_seasons_csv 6 "$CURRENT_SEASON_NBA")
LIGA_MX_SEASONS=$(build_seasons_csv 5 "$CURRENT_YEAR")
MLB_SEASONS=$(build_seasons_csv 6 "$CURRENT_YEAR")

telegram_notify() {
    local msg="$1"
    [ -z "${TELEGRAM_BOT_TOKEN:-}" ] && return
    $PY -c "
import httpx, os
t = os.environ.get('TELEGRAM_BOT_TOKEN', '')
c = os.environ.get('TELEGRAM_CHAT_ID', '')
if t and c:
    try:
        httpx.post(f'https://api.telegram.org/bot{t}/sendMessage',
                   json={'chat_id': int(c), 'text': '''$msg'''}, timeout=10)
    except Exception: pass
" 2>&1 | tail -1
}

echo "[$(ts)] autopilot_parallel START (year=$CURRENT_YEAR, 6 seeds en paralelo)" | tee logs/autopilot_parallel.log
telegram_notify "🚀 Autopilot PARALELO iniciado — 6 seeds simultáneos. Acelerado ~5x. Te aviso cuando termine."

# ─── Activar flags (rápido) ──────────────────────────────────────────────
$PY <<'PYEOF' 2>&1 | tee -a logs/autopilot_parallel.log
from pathlib import Path
env = Path('.env')
text = env.read_text(encoding='utf-8')
flags = {'APUESTAS_ENABLE_SOFASCORE': 'true', 'APUESTAS_ENABLE_BETUS': 'true'}
lines = text.splitlines()
seen = set()
out = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith('#') or '=' not in stripped:
        out.append(line); continue
    key, _, _v = stripped.partition('=')
    key = key.strip()
    if key in flags:
        out.append(f'{key}={flags[key]}'); seen.add(key)
    else: out.append(line)
for k, v in flags.items():
    if k not in seen: out.append(f'{k}={v}')
env.write_text('\n'.join(out) + '\n', encoding='utf-8')
print('flags set')
PYEOF

systemctl --user restart apuestas-telegram.service

# ─── Lanzar 6 seeds en paralelo ───────────────────────────────────────────
echo "[$(ts)] lanzando 6 seeds paralelos..." | tee -a logs/autopilot_parallel.log

run_seed() {
    local name="$1"
    local log="$2"
    shift 2
    $PY -m apuestas.scripts.seed_historical "$@" > "logs/$log" 2>&1 && \
        echo "[$(ts)] ✅ $name done" | tee -a logs/autopilot_parallel.log || \
        echo "[$(ts)] ⚠  $name fail" | tee -a logs/autopilot_parallel.log
}

# 1) Soccer EU 16 ligas (httpx async interno, rápido)
run_seed "soccer EU 16 ligas" "seed_soccer_eu.log" \
    --sport soccer-odds --seasons "$SOCCER_SEASONS" \
    --league epl,championship,la_liga,la_liga_2,bundesliga,bundesliga_2,serie_a,serie_b,ligue_1,ligue_2,eredivisie,liga_portugal,belgium_a,turkey_super,greece_super,scotland_premier &
PID_SOCCER=$!

# 2) Tennis ATP+WTA (XLSX download)
run_seed "tennis ATP+WTA" "seed_tennis.log" \
    --sport tennis --seasons "$TENNIS_SEASONS" --league atp,wta &
PID_TENNIS=$!

# 3) US sports odds (NBA+NFL+NHL, CSVs GitHub)
run_seed "US sports" "seed_us_sports.log" \
    --sport us-sports-odds --league nba,nfl,nhl &
PID_US=$!

# 4) NBA detalle (nba_api)
run_seed "NBA nba_api" "seed_nba.log" \
    --sport nba --seasons "$NBA_SEASONS" &
PID_NBA=$!

# 5) MLB (pybaseball)
run_seed "MLB pybaseball" "seed_mlb.log" \
    --sport mlb --seasons "$MLB_SEASONS" &
PID_MLB=$!

# 6) Liga MX + Expansion MX (fbref)
run_seed "Liga MX + Expansion" "seed_liga_mx.log" \
    --sport liga-mx --seasons "$LIGA_MX_SEASONS" --league liga_mx,liga_expansion &
PID_MX=$!

echo "[$(ts)] PIDs paralelos: soccer=$PID_SOCCER tennis=$PID_TENNIS us=$PID_US nba=$PID_NBA mlb=$PID_MLB mx=$PID_MX" | tee -a logs/autopilot_parallel.log

# Esperar a todos
wait $PID_SOCCER $PID_TENNIS $PID_US $PID_NBA $PID_MLB $PID_MX 2>/dev/null

echo "[$(ts)] TODOS los seeds completados" | tee -a logs/autopilot_parallel.log
telegram_notify "✅ [SEED] 6 fuentes completadas en paralelo. Ahora player logs + retrain + backtest."

# ─── Seed player_game_logs (NBA + NFL en paralelo) ────────────────────────
echo "[$(ts)] seed_player_game_logs..." | tee -a logs/autopilot_parallel.log
$PY -m apuestas.scripts.seed_player_game_logs_nba --seasons "2023-24,2024-25" \
    >> logs/seed_player_logs_nba.log 2>&1 &
PID_PLA_NBA=$!
$PY -m apuestas.scripts.seed_player_game_logs_nfl --seasons "2023,2024" \
    >> logs/seed_player_logs_nfl.log 2>&1 &
PID_PLA_NFL=$!
wait $PID_PLA_NBA $PID_PLA_NFL 2>/dev/null
telegram_notify "✅ [PLAYER LOGS] NBA + NFL player_game_logs seedeados."

# ─── Compute player_stat_std desde logs ───────────────────────────────────
echo "[$(ts)] compute_player_stat_std..." | tee -a logs/autopilot_parallel.log
$PY -m apuestas.scripts.compute_player_stat_std --all \
    >> logs/compute_player_stat_std.log 2>&1 || \
    echo "[$(ts)] ⚠  compute_player_stat_std fallo" | tee -a logs/autopilot_parallel.log

# ─── Retrain (cada sport puede paralelo si memoria lo permite) ────────────
echo "[$(ts)] retrain_weekly_flow..." | tee -a logs/autopilot_parallel.log
$PY -m apuestas.flows.retrain_weekly >> logs/autopilot_parallel.log 2>&1 || \
    echo "[$(ts)] ⚠  retrain fallo" | tee -a logs/autopilot_parallel.log

telegram_notify "✅ [RETRAIN] Modelos intentados."

# ─── Drift monitor post-retrain ───────────────────────────────────────────
echo "[$(ts)] drift_monitor..." | tee -a logs/autopilot_parallel.log
$PY -m apuestas.monitors.drift_monitor >> logs/drift_monitor.log 2>&1 || \
    echo "[$(ts)] ⚠  drift_monitor fallo" | tee -a logs/autopilot_parallel.log

# ─── Backtest en paralelo (NBA + Soccer) ──────────────────────────────────
BACKTEST_SEASONS=$(build_seasons_csv 3 "$((CURRENT_YEAR - 1))")
$PY -m apuestas.scripts.run_backtest --sport nba --seasons "$BACKTEST_SEASONS" --output reports/backtest_nba.json --min-sharpe 1.0 > logs/backtest_nba.log 2>&1 &
$PY -m apuestas.scripts.run_backtest --sport soccer --seasons "$BACKTEST_SEASONS" --output reports/backtest_soccer.json --min-sharpe 1.0 > logs/backtest_soccer.log 2>&1 &
wait

echo "[$(ts)] autopilot_parallel DONE" | tee -a logs/autopilot_parallel.log

# Summary final
SUMMARY=$($PY -c "
import asyncio
from apuestas.db import session_scope
from sqlalchemy import text as t
async def go():
    async with session_scope() as s:
        r = await s.execute(t('''SELECT sport_code, COUNT(*) n, COUNT(home_score) r
                                   FROM matches GROUP BY sport_code ORDER BY 2 DESC'''))
        lines = [f'{row.sport_code}: {row.n} matches ({row.r} w/score)' for row in r.all()]
        r2 = await s.execute(t('''SELECT bookmaker, COUNT(*) n FROM odds_history
                                   WHERE ts > now()-interval '30 days'
                                   GROUP BY bookmaker ORDER BY 2 DESC LIMIT 10'''))
        lines.append('─── odds 30d ───')
        lines += [f'{row.bookmaker}: {row.n}' for row in r2.all()]
        r3 = await s.execute(t('SELECT model_name, stage, promoted_at FROM model_registry_meta ORDER BY promoted_at DESC'))
        lines.append('─── modelos ───')
        for row in r3.all():
            lines.append(f'{row.model_name} [{row.stage}]')
        if not lines[-1].startswith('─'):
            pass
        else:
            lines.append('(sin modelos)')
        print('\n'.join(lines))
asyncio.run(go())
" 2>&1 | tail -25)

telegram_notify "🎯 AUTOPILOT PARALELO COMPLETADO

$SUMMARY

Logs por sport en logs/seed_*.log
Backtests en reports/backtest_*.json"
