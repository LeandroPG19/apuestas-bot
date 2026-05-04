#!/usr/bin/env bash
# Un solo comando: deja todo operando 24/7.
#
#   - Levanta Docker services (postgres + valkey + minio + mlflow + prefect)
#   - Arranca bot Telegram (systemd user, auto-restart)
#   - Programa análisis automático cada 6h (systemd user timer)
#   - Corre el primer ciclo completo (catchup + deep_analysis)
#   - Muestra el estado al terminar
#
# Uso: bash scripts/start.sh   (o: make go)

set -euo pipefail
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════════"
echo "  🚀 Apuestas Bot — start 24/7"
echo "═══════════════════════════════════════════════════════════"

# ─── 1. Sanity: .env con token válido ───────────────────────────────────
if grep -q "your-telegram-bot-token\|your-numeric-chat-id" .env 2>/dev/null; then
  echo "❌ .env aún tiene placeholders. Corre primero:"
  echo "     .venv/bin/python scripts/setup_telegram.py"
  exit 1
fi

# ─── 2. Docker services (base infra) ─────────────────────────────────────
echo ""
echo "📦 [1/4] Docker services..."
docker compose up -d postgres valkey minio mlflow prefect >/dev/null 2>&1 || true

# Esperar que postgres esté healthy (crítico para bot + analyze)
printf "    waiting postgres"
until [ "$(docker inspect apuestas-postgres --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ]; do
  printf "."
  sleep 2
done
echo " ✅"

# ─── 3. Bot Telegram (systemd user) ──────────────────────────────────────
echo ""
echo "🤖 [2/4] Bot Telegram (systemd)..."
mkdir -p logs
systemctl --user daemon-reload >/dev/null
if ! systemctl --user is-enabled apuestas-telegram.service >/dev/null 2>&1; then
  systemctl --user enable apuestas-telegram.service >/dev/null 2>&1 || true
fi
# Si quedó en failed por SIGKILL previo, resetear antes de restart
systemctl --user reset-failed apuestas-telegram.service >/dev/null 2>&1 || true
systemctl --user restart apuestas-telegram.service
# Verificar que efectivamente arrancó
sleep 2
if ! systemctl --user is-active apuestas-telegram.service >/dev/null 2>&1; then
  echo "    ❌ Bot NO arrancó — ver logs:"
  echo "       journalctl --user -u apuestas-telegram.service -n 30 --no-pager"
  exit 1
fi

# Modo on-demand puro: NO instalar timers en background.
# El usuario arranca el bot 1h, recibe picks, lo apaga. Cualquier timer corriendo
# durante las ~22h restantes quema quota de The Odds API ($/mes).
# Si en una sesión quiere otro ciclo, lo dispara con `apuestas analyze` o desde
# Telegram con /analyze. Pero NO hay nada automático.
echo "    bot:     $(systemctl --user is-active apuestas-telegram.service)"
echo "    modo:    on-demand (sin timers en background)"

# ─── 4. Primer ciclo manual ─────────────────────────────────────────────
echo ""
echo "🔄 [3/4] Catchup real (Pinnacle + Caliente + Codere + DK si VPN)..."
bash scripts/operate.sh catchup 2>&1 | tail -12 | sed 's/^/    /'

echo ""
echo "🎯 [4/4] Deep analysis..."
bash scripts/operate.sh analyze 2>&1 | tail -8 | sed 's/^/    /'

# ─── 5. Resumen final ───────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Bot ARRANCADO (modo on-demand)"
echo "═══════════════════════════════════════════════════════════"
bash scripts/operate.sh status 2>&1 | sed 's/^/  /'
echo ""
echo "  📱 Telegram: abre tu bot configurado, envía /start"
echo "  🔁 Otro ciclo: 'apuestas analyze' (manual; no hay timer en background)"
echo "  📊 TUI visual: bash scripts/tui.sh"
echo "  🛑 Parar todo: apuestas stop  (apaga bot + procesos)"
echo "  📋 Logs:      tail -f logs/telegram.log  |  tail -f logs/analyze.log"
echo "═══════════════════════════════════════════════════════════"
