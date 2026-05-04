#!/usr/bin/env bash
# scripts/operate.sh — operar el bot end-to-end desde host (sin container api).
#
# Post-pivote 2026-04-23: modo detector puro. El bot ya no maneja banca,
# stake ni PnL; solo emite alertas de valor que tú decides apostar manualmente.
#
# Uso:
#   bash scripts/operate.sh catchup          # solo ingesta (Pinnacle + Caliente + DK si VPN)
#   bash scripts/operate.sh analyze          # solo deep_analysis (requiere catchup previo)
#   bash scripts/operate.sh full             # catchup + analyze (todo-en-uno)
#   bash scripts/operate.sh status           # estado stack + alertas + precisión
#   bash scripts/operate.sh picks            # últimas alertas emitidas (24h)

set -euo pipefail
cd "$(dirname "$0")/.."

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
# Fuerza output sin buffering para que logs aparezcan en tiempo real
export PYTHONUNBUFFERED=1
unset PYTHON_GIL

# Carga token/chat_id del .env sin inyectar GIL
while IFS='=' read -r k v; do
  case "$k" in
    TELEGRAM_BOT_TOKEN|TELEGRAM_CHAT_ID|APUESTAS_US_VPN_ACTIVE|APUESTAS_ENABLE_DK|APUESTAS_ENABLE_FANDUEL|APUESTAS_ENABLE_BETMGM|APUESTAS_ENABLE_PROPS|APUESTAS_MIRROR_MIN_COMPLETENESS|APUESTAS_PAPER_TRADING)
      export "$k=$v" ;;
  esac
done < <(grep -E '^[A-Z_]+=' .env | sed 's/#.*//' | sed 's/[[:space:]]*$//')

PY=.venv/bin/python

case "${1:-help}" in

catchup)
  echo "═══ CATCHUP (odds últimas 30 min) ═══"
  $PY -c "
import asyncio
from apuestas.flows.catchup import catchup_flow
r = asyncio.run(catchup_flow.fn())
for k, v in r.items():
    print(f'  {k}: {v}')
"
  ;;

analyze)
  echo "═══ DEEP ANALYSIS (emite picks con EV≥1%) ═══"
  echo "🔎 Iniciando búsqueda... (puede tardar 2-6 min)"
  echo ""
  $PY -u -c "
import asyncio, sys
from apuestas.flows.deep_analysis import deep_analysis_flow
print('📡 Analizando próximos 48h (máx 25 eventos)...', flush=True)
r = asyncio.run(deep_analysis_flow(hours_ahead=48, max_events=25))
print('', flush=True)
print('✅ ANÁLISIS COMPLETADO', flush=True)
print('─' * 50, flush=True)
for k, v in r.items():
    print(f'  {k}: {v}', flush=True)
print('─' * 50, flush=True)
if r.get('picks_emitted', 0) == 0:
    print('ℹ️  0 picks emitidos: no hay EV+ en los eventos actuales.', flush=True)
    print('   Esto es normal cuando los mercados están eficientes.', flush=True)
else:
    print(f'🎯 {r.get(\"picks_emitted\", 0)} picks nuevos enviados a Telegram + grupo.', flush=True)
"
  ;;

full)
  bash "$0" catchup
  echo ""
  bash "$0" analyze
  echo ""
  bash "$0" picks
  ;;

status)
  echo "═══ SERVICIOS ═══"
  systemctl --user is-active apuestas-telegram.service | xargs -I{} echo "  telegram-bot: {}"
  docker ps --format '  {{.Names}}: {{.Status}}' | grep apuestas | sort
  echo ""
  echo "═══ DATA ═══"
  $PY -c "
import asyncio
from apuestas.db import session_scope
from sqlalchemy import text as t
async def go():
    async with session_scope() as s:
        row = (await s.execute(t('''
          SELECT COUNT(*) FILTER (WHERE ts > now()-interval '1 hour') AS odds_1h,
                 COUNT(*) FILTER (WHERE ts > now()-interval '24 hours') AS odds_24h,
                 COUNT(DISTINCT bookmaker) FILTER (WHERE ts > now()-interval '24 hours') AS books
          FROM odds_history'''))).first()
        print(f'  odds 1h: {row.odds_1h}  ·  24h: {row.odds_24h}  ·  books distintos 24h: {row.books}')
        row = (await s.execute(t('''
          SELECT COUNT(*) AS n_active,
                 COUNT(*) FILTER (WHERE outcome_result IN ('won','lost')) AS resolved
          FROM pick_alerts
        '''))).first()
        print(f'  alertas vivas: {row.n_active - (row.resolved or 0)}  ·  resueltas: {row.resolved or 0}')
        row = (await s.execute(t('''
          SELECT COUNT(*) AS n,
                 COUNT(*) FILTER (WHERE outcome_result = 'won') AS wins
          FROM pick_alerts
          WHERE outcome_result IN ('won','lost')
            AND result_settled_at >= now()-interval '30 days'
        '''))).first()
        hit = (row.wins / row.n) if row.n else 0.0
        print(f'  precisión 30d: n={row.n or 0}  wins={row.wins or 0}  hit_rate={hit:.1%}')
asyncio.run(go())
"
  ;;

picks)
  echo "═══ ÚLTIMOS PICKS 24h ═══"
  $PY -c "
import asyncio
from apuestas.db import session_scope
from sqlalchemy import text as t
async def go():
    async with session_scope() as s:
        rows = (await s.execute(t('''
          SELECT pa.id, pa.placed_at, m.sport_code, ht.name AS home, at.name AS away,
                 pa.market, pa.outcome, pa.line, pa.bookmaker, pa.odds_placed,
                 pa.best_odds_seen, pa.upgrade_count, pa.outcome_result,
                 p.ev,
                 pa.notification_sent_at IS NOT NULL AS notified
          FROM pick_alerts pa
          JOIN matches m ON m.id = pa.match_id
          JOIN teams ht ON ht.id = m.home_team_id
          JOIN teams at ON at.id = m.away_team_id
          LEFT JOIN predictions p ON p.id = pa.prediction_id
          WHERE pa.placed_at > now()-interval '24 hours'
          ORDER BY pa.placed_at DESC
          LIMIT 20'''))).all()
        if not rows:
            print('  📭 Sin alertas en 24h. Corre: bash scripts/operate.sh analyze')
            return
        for r in rows:
            bell = '🔔' if r.notified else '  '
            ev = float(r.ev or 0)*100
            status = r.outcome_result or 'abierta'
            up = f' ↑{r.upgrade_count}x' if r.upgrade_count else ''
            print(f\"  #{r.id} {bell}  {r.sport_code}  {r.home[:15]:>15} vs {r.away[:15]:<15} | {r.market}/{r.outcome}  @ {r.bookmaker} {float(r.odds_placed):.2f}{up}  EV={ev:+.2f}%  [{status}]\")
asyncio.run(go())
"
  ;;

*)
  cat <<EOF
═══════════════════════════════════════════════════════════
  Apuestas Bot — operar desde CLI (modo detector puro)
═══════════════════════════════════════════════════════════
  bash scripts/operate.sh catchup    — ingesta odds últimas
  bash scripts/operate.sh analyze    — correr deep_analysis
  bash scripts/operate.sh full       — catchup + analyze + picks
  bash scripts/operate.sh status     — estado servicios + alertas
  bash scripts/operate.sh picks      — últimas alertas 24h
═══════════════════════════════════════════════════════════

Alternativas:
  • Desde Telegram: escribe /analyze en tu bot configurado
  • TUI visual: bash scripts/tui.sh
  • Logs bot: tail -f logs/telegram.log
EOF
  ;;
esac
