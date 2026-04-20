#!/usr/bin/env bash
# chaos.sh — Chaos engineering lite (§19.17).
# Mata un container aleatorio worker/service y valida auto-recovery.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

TARGETS=(
  "apuestas-worker-ingest"
  "apuestas-worker-ml"
  "apuestas-worker-scrape"
  "apuestas-llm"
  "apuestas-embed"
)

# Selector aleatorio
VICTIM="${TARGETS[$RANDOM % ${#TARGETS[@]}]}"

echo "── Chaos drill — $(date +%F_%T) ──"
echo "Víctima elegida: $VICTIM"

# Verificar que está running
if ! docker ps --format '{{.Names}}' | grep -q "^${VICTIM}$"; then
  warn "$VICTIM no está running. Skip chaos."
  exit 0
fi

# 1. Snapshot pre-kill
echo "▶ Pre-kill healthcheck..."
docker inspect --format='{{.State.Health.Status}}' "$VICTIM" 2>/dev/null || echo "no healthcheck"

# 2. Kill
echo "▶ Kill $VICTIM..."
docker kill "$VICTIM"

# 3. Esperar restart
echo "▶ Esperando restart (30s)..."
for i in {1..30}; do
  if docker ps --format '{{.Names}}' | grep -q "^${VICTIM}$"; then
    ok "$VICTIM reiniciado en ${i}s"
    break
  fi
  sleep 1
done

# 4. Health check tras restart
echo "▶ Post-restart healthcheck (60s grace)..."
for i in {1..60}; do
  status=$(docker inspect --format='{{.State.Health.Status}}' "$VICTIM" 2>/dev/null || echo "no_hc")
  if [[ "$status" == "healthy" ]]; then
    ok "$VICTIM healthy tras ${i}s"
    exit 0
  fi
  if [[ "$status" == "no_hc" ]]; then
    if docker inspect --format='{{.State.Running}}' "$VICTIM" | grep -q true; then
      ok "$VICTIM running (sin healthcheck)"
      exit 0
    fi
  fi
  sleep 1
done

warn "$VICTIM no alcanzó healthy en 60s — revisar logs"
docker logs --tail 30 "$VICTIM"
exit 1
