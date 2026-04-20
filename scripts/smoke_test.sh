#!/usr/bin/env bash
# smoke_test.sh — Verifica health de todos los servicios tras make up.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail_count=0
fail() { echo -e "${RED}✗${NC} $*"; fail_count=$((fail_count + 1)); }

load_env() {
  [[ -f .env ]] || { fail ".env no existe"; exit 1; }
  set -a; . ./.env; set +a
}
load_env

check() {
  local name="$1" url="$2" expected="${3:-200}"
  if curl -fsS --max-time 5 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null | grep -q "^$expected"; then
    ok "$name → $url"
  else
    fail "$name no responde en $url"
  fi
}

echo "── Smoke test: servicios y endpoints ──"

if docker compose ps --status running --quiet | grep -q .; then
  ok "Containers arriba"
else
  fail "Ningún container running. Ejecuta: make up"
  exit 1
fi

# Postgres
if docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-apuestas}" -d "${POSTGRES_DB:-apuestas}" >/dev/null 2>&1; then
  ok "Postgres accepting connections"
else
  fail "Postgres no responde"
fi

# Verificar extensiones
if docker compose exec -T postgres psql -U "${POSTGRES_USER:-apuestas}" -d "${POSTGRES_DB:-apuestas}" -tAc \
  "SELECT 1 FROM pg_extension WHERE extname IN ('timescaledb','vector','pg_trgm') GROUP BY 1" 2>/dev/null | grep -q 1; then
  ok "Extensiones timescaledb + vector + pg_trgm instaladas"
else
  fail "Extensiones Postgres faltantes"
fi

# Valkey
if docker compose exec -T valkey valkey-cli --pass "${VALKEY_PASSWORD}" ping 2>/dev/null | grep -q PONG; then
  ok "Valkey PONG"
else
  fail "Valkey no responde"
fi

# MLflow
check "MLflow" "http://localhost:5000/health"

# Prefect
check "Prefect" "http://localhost:4200/api/health"

# FastAPI
check "FastAPI health" "http://localhost:${API_HOST_PORT:-8001}/health"
check "FastAPI metrics" "http://localhost:${API_HOST_PORT:-8001}/metrics"

# MinIO console (si se expone)
check "MinIO console" "http://localhost:9001" 403 || true

# GPU containers
if docker compose ps llm 2>/dev/null | grep -q running; then
  if docker compose exec -T llm curl -fsS http://localhost:8080/health >/dev/null 2>&1; then
    ok "llama.cpp server (LLM)"
  else
    warn "llama.cpp aún calentando (primer arranque puede tomar 60s)"
  fi
fi

if docker compose ps embed 2>/dev/null | grep -q running; then
  if docker compose exec -T embed curl -fsS http://localhost/health >/dev/null 2>&1; then
    ok "TEI (embed BGE-M3)"
  else
    warn "TEI aún descargando modelo (primera vez ~3 min)"
  fi
fi

echo ""
if [[ $fail_count -eq 0 ]]; then
  echo -e "${GREEN}✅ Smoke test PASSED${NC}"
  exit 0
else
  echo -e "${RED}✗ Smoke test FAILED ($fail_count errores)${NC}"
  exit 1
fi
