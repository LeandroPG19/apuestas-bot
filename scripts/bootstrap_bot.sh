#!/usr/bin/env bash
# Bootstrap completo tras setup_telegram.py:
# - Arranca stack Docker (postgres + valkey + api + telegram).
# - Verifica que Telegram container se conecte.
# - Corre catchup real (Pinnacle + Caliente + Codere + DK si VPN).
# - Lanza deep_analysis para que emita picks y los auto-envíe al chat.
#
# Uso:
#   bash scripts/bootstrap_bot.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════════"
echo "  🚀 Bootstrap Apuestas Bot — post setup_telegram.py"
echo "═══════════════════════════════════════════════════════════"

# 1. Sanity check — .env debe tener token/chat ya configurados
if grep -q "your-telegram-bot-token\|your-numeric-chat-id" .env; then
  echo "❌ .env aún tiene placeholders. Corre primero:"
  echo "     .venv/bin/python scripts/setup_telegram.py"
  exit 1
fi

if ! grep -q "^APUESTAS_US_VPN_ACTIVE=true" .env; then
  echo "⚠  APUESTAS_US_VPN_ACTIVE no está en true. Activando..."
  echo "APUESTAS_US_VPN_ACTIVE=true" >>.env
fi

# 2. Levantar stack completo (compose) — api depende de mlflow+prefect+valkey+postgres
echo ""
echo "📦 Levantando stack Docker..."
docker compose up -d postgres valkey minio mlflow prefect api telegram 2>&1 | tail -30

# 3. Esperar healthy
echo ""
echo "⏳ Esperando que api + telegram estén healthy..."
for i in {1..30}; do
  api_state=$(docker inspect apuestas-api --format '{{.State.Health.Status}}' 2>/dev/null || echo "missing")
  tg_state=$(docker inspect apuestas-telegram --format '{{.State.Status}}' 2>/dev/null || echo "missing")
  if [ "$api_state" = "healthy" ] && [ "$tg_state" = "running" ]; then
    echo "✅ api=$api_state · telegram=$tg_state"
    break
  fi
  sleep 3
done

# 4. Verificar logs del bot telegram
echo ""
echo "📋 Últimas 10 líneas del container telegram:"
docker logs apuestas-telegram --tail 10 2>&1 || echo "⚠  container no existe todavía"

# 5. Correr catchup real
echo ""
echo "🔄 Corriendo catchup_flow real (puede tardar 2-3 min)..."
docker compose exec -T api python -c "
import asyncio
from apuestas.flows.catchup import catchup_flow
r = asyncio.run(catchup_flow.fn())
print('Catchup summary:', {k: (len(v) if hasattr(v, '__len__') else v) for k, v in r.items()})
" || echo "⚠  catchup falló, revisa logs"

# 6. Deep analysis → emite picks → auto-notifica Telegram
echo ""
echo "🎯 Deep analysis — emite picks al Telegram chat..."
docker compose exec -T api python -c "
import asyncio
from apuestas.flows.deep_analysis import deep_analysis_flow
r = asyncio.run(deep_analysis_flow(hours_ahead=48, max_events=15))
print('Deep analysis:', r)
" || echo "⚠  deep_analysis falló"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Bootstrap completo."
echo ""
echo "  • Los picks emitidos llegan automáticamente al chat Telegram."
echo "  • Desde el bot: /analyze /today /bankroll /clv"
echo "  • TUI visual:   make tui   o   .venv/bin/python -m apuestas.tui"
echo "  • Logs bot:     docker logs -f apuestas-telegram"
echo "  • Apagar:       make down"
echo "═══════════════════════════════════════════════════════════"
