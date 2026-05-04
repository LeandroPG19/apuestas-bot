#!/usr/bin/env bash
# Apagado graceful: todos los datos quedan persistidos.
#
#   - Stop timer auto-análisis (deja el bot corriendo hasta que ACK updates)
#   - Stop bot con SIGTERM → PicklePersistence flushea offset + state
#   - Detiene containers SIN borrar volúmenes (Postgres/MinIO/MLflow/Prefect)
#   - Reporta dónde quedan persistidos los datos
#
# Uso:
#   bash scripts/stop.sh            (graceful, deja Docker base up)
#   bash scripts/stop.sh --full     (apaga TAMBIÉN Docker, volúmenes intactos)

set -euo pipefail
cd "$(dirname "$0")/.."

FULL=${1:-}

echo "🛑 Apagando Apuestas Bot..."
echo ""

# 1) Detener TODOS los timers + services apuestas-* (excepto telegram, que va en paso 2)
#    Discovery dinámico: cualquier unit apuestas-* con prefijo conocido se detiene.
echo "[1/3] Deteniendo todos los timers + services apuestas-*..."

# 1a) Timers — primero, para que no disparen nuevos services mientras apagamos
ALL_TIMERS=$(systemctl --user list-unit-files --type=timer --no-legend 2>/dev/null \
  | awk '/^apuestas-/ {print $1}')
for t in $ALL_TIMERS; do
  systemctl --user stop "$t" 2>/dev/null || true
done

# 1b) Services oneshot/simple (excepto telegram, settle-worker; settle-worker puede no
#     existir post-Sprint 1 demolición bankroll, lo deshabilitamos defensivo).
ALL_SERVICES=$(systemctl --user list-unit-files --type=service --no-legend 2>/dev/null \
  | awk '/^apuestas-/ {print $1}' \
  | grep -v "apuestas-telegram\.service")
for s in $ALL_SERVICES; do
  systemctl --user stop "$s" 2>/dev/null || true
done

# 1c) Settle-worker zombie (módulo borrado en Sprint 1, en loop auto-restart).
#     Disable persistente para que no resucite tras reboot.
if systemctl --user list-unit-files apuestas-settle-worker.service --no-legend 2>/dev/null | grep -q .; then
  systemctl --user stop apuestas-settle-worker.service 2>/dev/null || true
  systemctl --user disable apuestas-settle-worker.service 2>/dev/null || true
fi

# 1d) Kill orphan procesos lanzados por ejecuciones manuales (apuestas analyze, etc.)
pkill -f "apuestas.flows.catchup" 2>/dev/null || true
pkill -f "apuestas.flows.deep_analysis" 2>/dev/null || true
pkill -f "apuestas.flows.historical_backfill" 2>/dev/null || true
pkill -f "apuestas.flows.live_scores" 2>/dev/null || true
pkill -f "apuestas.flows.pre_match" 2>/dev/null || true
pkill -f "apuestas.flows.in_play" 2>/dev/null || true
pkill -f "apuestas.flows.capture_closing_lines" 2>/dev/null || true
pkill -f "apuestas.flows.enrich_features" 2>/dev/null || true
pkill -f "apuestas.flows.steam_watcher" 2>/dev/null || true
pkill -f "apuestas.flows.sofascore_sync" 2>/dev/null || true
pkill -f "apuestas.flows.alert_cleanup" 2>/dev/null || true
pkill -f "apuestas.flows.retrain_weekly" 2>/dev/null || true
pkill -f "apuestas.flows.retrain_on_drift" 2>/dev/null || true
pkill -f "apuestas.scripts.seed_historical" 2>/dev/null || true
pkill -f "apuestas.scripts.operate" 2>/dev/null || true
pkill -f "apuestas.ingest.injury_nlp_ingest" 2>/dev/null || true
pkill -f "apuestas.ingest.lineups_mlb" 2>/dev/null || true
pkill -f "apuestas.ingest.news_pipeline" 2>/dev/null || true
pkill -f "apuestas.ingest.referee_scraper" 2>/dev/null || true
pkill -f "apuestas.ingest.injury_archive" 2>/dev/null || true
pkill -f "apuestas.ingest.twitter_insiders" 2>/dev/null || true
pkill -f "apuestas.ml.train_" 2>/dev/null || true
pkill -f "apuestas.monitors.drift_monitor" 2>/dev/null || true
pkill -f "scripts/run_lineup_scratch.py" 2>/dev/null || true
pkill -f "scripts/autopilot" 2>/dev/null || true
pkill -f "scripts/full_retrain_sprint13.sh" 2>/dev/null || true

n_stopped=$(echo "$ALL_TIMERS $ALL_SERVICES" | wc -w)
echo "    ✅ $n_stopped units systemd detenidas + procesos huérfanos kill"

# 2) Bot Telegram con SIGTERM → graceful shutdown
#    PTB escucha SIGTERM, flushea PicklePersistence (logs/telegram_state.pickle),
#    ACK el último update, cierra conexión long-polling.
echo "[2/3] Deteniendo bot Telegram (graceful)..."
# Stop en background para no bloquear si se cuelga
systemctl --user stop apuestas-telegram.service 2>/dev/null &
STOP_PID=$!
# Espera hasta 15s al graceful stop; si no, forzamos kill.
for i in {1..15}; do
  if ! systemctl --user is-active apuestas-telegram.service >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
# Si todavía está, kill forzado
if systemctl --user is-active apuestas-telegram.service >/dev/null 2>&1; then
  echo "    ⚠️  graceful stop timeout tras 15s — forzando kill..."
  systemctl --user kill -s SIGKILL apuestas-telegram.service 2>/dev/null || true
  sleep 1
fi
wait $STOP_PID 2>/dev/null || true
# Limpiar estado failed para que el próximo 'apuestas go' no falle
systemctl --user reset-failed apuestas-telegram.service 2>/dev/null || true

# Kill any leftover python procs en venv/apuestas (TUI, granian local, prefect ephemeral,
# uvicorn lanzado por flows ephemeral, etc.) — NO toca docker, NO toca claude/MCPs.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APUESTAS_VENV_PY="$PROJECT_ROOT/.venv/bin/python"
APUESTAS_VENV_PY3="$PROJECT_ROOT/.venv/bin/python3"
for pat in "$APUESTAS_VENV_PY" "$APUESTAS_VENV_PY3"; do
  pkill -f "^$pat" 2>/dev/null || true
done
# Prefect ephemeral uvicorn lanzado por flows fuera de Docker
pkill -f "prefect.server.api.server:create_app" 2>/dev/null || true
# TUI Textual
pkill -f "apuestas.tui" 2>/dev/null || true
# Bot scripts wrapper
pkill -f "scripts/run_telegram_bot.sh" 2>/dev/null || true

if [ -f logs/telegram_state.pickle ]; then
  size=$(du -h logs/telegram_state.pickle | cut -f1)
  echo "    ✅ estado persistido en logs/telegram_state.pickle ($size)"
else
  echo "    ✅ bot detenido"
fi

# 2.5) Sanity: ¿quedó algo del proyecto vivo? (pgrep retorna 1 si no hay match → || true)
remaining=$(pgrep -fa "${APUESTAS_VENV_PY}|apuestas\.flows|apuestas\.ingest|apuestas\.ml\.train_" 2>/dev/null | wc -l || true)
if [ "${remaining:-0}" -gt 0 ]; then
  echo "    ⚠️  $remaining proceso(s) Python apuestas todavía vivos — kill -9..."
  pgrep -f "${APUESTAS_VENV_PY}" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  pgrep -f "apuestas\.flows|apuestas\.ingest|apuestas\.ml\.train_" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
fi
true  # asegurar exit-code 0 antes de continuar bajo set -e

# 3) Docker (opcional)
if [ "$FULL" = "--full" ]; then
  echo "[3/3] Deteniendo containers Docker..."
  docker compose stop 2>&1 | sed 's/^/    /'
  echo "    ✅ containers apagados (volúmenes intactos)"
else
  echo "[3/3] Docker base services: dejados UP."
  echo "    → Para apagar también: bash scripts/stop.sh --full"
fi

# 4) Reporte de persistencia
cat <<EOF

═══════════════════════════════════════════════════════════
  ✅ Apagado completo · NINGÚN DATO PERDIDO
═══════════════════════════════════════════════════════════

Datos persistidos automáticamente:

  🗄  PostgreSQL       → volumen docker 'postgres_data'
      (matches, odds_history, bets, predictions, post_mortems,
       bankroll_history, model_registry_meta, audit_log)

  📦 MinIO             → volumen docker 'minio_data'
      (artefactos MLflow, datasets, backups)

  📈 MLflow            → volumen docker 'mlflow_data'
      (sqlite local con experiments + runs)

  🔄 Prefect           → volumen docker 'prefect_data'
      (historial de flows, task states)

  💾 Valkey            → volumen docker 'valkey_data'
      (cache TaskIQ; regenerable, no es crítico)

  🤖 Telegram state    → logs/telegram_state.pickle
      (offset updates, conversation states, user_data)

  📋 Logs              → logs/telegram.log + logs/analyze.log

  ⚙️  Configuración    → .env (token, chat_id, flags)

  🎛  Systemd units    → ~/.config/systemd/user/apuestas-*.service/.timer
      (re-activables con: make go)

═══════════════════════════════════════════════════════════
  Para reactivar: make go    (un solo comando)
═══════════════════════════════════════════════════════════
EOF
