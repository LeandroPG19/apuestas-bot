#!/usr/bin/env bash
# TUI como centro de mando único:
#   - Arranca todo si no está arriba (Docker + bot + timer auto-análisis)
#   - Abre la TUI Textual
#   - Al salir de la TUI: pregunta si apagar todo o dejar corriendo
#
# Uso:
#   bash scripts/tui.sh           → arranca todo + TUI (pregunta al salir)
#   bash scripts/tui.sh --stop-on-exit  → apaga automático al cerrar TUI
#   bash scripts/tui.sh --no-start      → solo abre TUI (no toca servicios)
#
# Alias:  make tui

set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-}  # "", --stop-on-exit, --no-start

# ─── 1. Sanity: .env válido ─────────────────────────────────────────────
if [ ! -f .env ] || grep -q "your-telegram-bot-token\|your-numeric-chat-id" .env 2>/dev/null; then
  echo "❌ .env aún tiene placeholders. Corre primero:"
  echo "   make telegram-setup"
  exit 1
fi

# ─── 2. Bootstrap (idempotente) — solo si no está corriendo ──────────────
if [ "$MODE" != "--no-start" ]; then
  echo "═══════════════════════════════════════════════════════════"
  echo "  🚀 Iniciando stack Apuestas"
  echo "═══════════════════════════════════════════════════════════"

  # Docker base (idempotente: ya corriendo → no-op)
  echo "📦 [1/3] Docker base services..."
  docker compose up -d postgres valkey minio mlflow prefect >/dev/null 2>&1 || true
  printf "    waiting postgres"
  until [ "$(docker inspect apuestas-postgres --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ]; do
    printf "."
    sleep 2
  done
  echo " ✅"

  # Bot Telegram (systemd). Modo on-demand puro: SIN timers en background.
  # El usuario usa el bot ~1h por sesión, 2× al día. Cualquier timer entre
  # sesiones quema cuota de The Odds API ($/mes) sin que nadie lo vea.
  echo "🤖 [2/3] Bot Telegram (on-demand, sin timers)..."
  mkdir -p logs
  systemctl --user daemon-reload >/dev/null 2>&1 || true

  if ! systemctl --user is-active apuestas-telegram.service >/dev/null 2>&1; then
    systemctl --user start apuestas-telegram.service >/dev/null 2>&1 || true
    sleep 2
  fi
  echo "    bot:     $(systemctl --user is-active apuestas-telegram.service)"
  echo "    modo:    on-demand (apuestas analyze para ciclos extras)"

  # Primer catchup si DB está vacía de odds recientes (no bloquea)
  echo "🔄 [3/3] Verificando data fresca..."
  RECENT_ODDS=$(docker exec apuestas-postgres psql -U apuestas -d apuestas -tAc \
    "SELECT COUNT(*) FROM odds_history WHERE ts > now()-interval '2 hours'" 2>/dev/null || echo 0)
  if [ "${RECENT_ODDS:-0}" -lt 100 ]; then
    echo "    Lanzando catchup en background (logs: logs/analyze.log)..."
    (bash scripts/operate.sh catchup >> logs/analyze.log 2>&1 &)
  else
    echo "    ✅ ${RECENT_ODDS} odds frescas en últimas 2h"
  fi
  echo ""
fi

# ─── 3. Launch TUI ───────────────────────────────────────────────────────
# Entorno para ejecución fuera de Docker (conecta a puerto expuesto)
set -a
# shellcheck disable=SC1091
source .env 2>/dev/null || true
set +a
unset PYTHON_GIL  # el venv no es free-threaded
export POSTGRES_HOST="${POSTGRES_HOST_OVERRIDE:-localhost}"
export POSTGRES_PORT="${POSTGRES_HOST_PORT:-5434}"
export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export PREFECT_API_URL="${PREFECT_API_URL:-http://localhost:4200/api}"
export PYTHONPATH="src"
export APUESTAS_TUI_ACTIVE=1

# Busca venv (prefiere .venv local, luego /tmp/test-venv)
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x "/tmp/test-venv/bin/python" ]; then
  PY="/tmp/test-venv/bin/python"
else
  echo "❌ No hay venv. Corre: uv sync"
  exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "  📊 Abriendo TUI · DB: localhost:${POSTGRES_PORT}"
echo "  Tip: 'q' para salir, tabs/flechas para navegar"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Corre la TUI (no exec — necesitamos recuperar control al salir)
set +e
"$PY" -m apuestas.tui
TUI_EXIT=$?
set -e

# ─── 4. Al cerrar la TUI: preguntar qué hacer con servicios ─────────────
echo ""
if [ "$MODE" = "--stop-on-exit" ]; then
  DECISION="s"
elif [ "$MODE" = "--no-start" ]; then
  DECISION="n"
else
  echo "═══════════════════════════════════════════════════════════"
  echo "  TUI cerrada. ¿Qué hacer con los servicios?"
  echo "═══════════════════════════════════════════════════════════"
  echo ""
  echo "  [s] Apagar todo (bot + timer + Docker)"
  echo "  [m] Apagar bot y timer, mantener Docker (default 10s)"
  echo "  [n] NO apagar nada — seguir recibiendo picks 24/7"
  echo ""
  read -t 10 -p "  Opción [s/m/N]: " DECISION || DECISION="m"
  DECISION=$(echo "${DECISION:-m}" | tr '[:upper:]' '[:lower:]')
fi

case "$DECISION" in
  s)
    echo ""
    echo "🛑 Apagando TODO (bot + timer + Docker)..."
    bash scripts/stop.sh --full
    ;;
  n)
    echo ""
    echo "✅ Servicios siguen UP. El bot recibe picks 24/7."
    echo "   Timer siguiente: $(systemctl --user list-timers apuestas-analyze.timer --no-pager 2>/dev/null | awk 'NR==2 {print $2, $3}' || echo 'activo')"
    echo "   Para apagar después: bash scripts/stop.sh"
    ;;
  m|*)
    echo ""
    echo "🛑 Apagando bot y timer, Docker sigue UP..."
    bash scripts/stop.sh
    ;;
esac

exit $TUI_EXIT
