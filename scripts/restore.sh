#!/usr/bin/env bash
# restore.sh — Restaura desde pg_dump file.
# Uso: make restore FILE=backups/pg_2026-04-19_0300.dump

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

DUMP_FILE="${1:-}"
[[ -n "$DUMP_FILE" ]] || fail "Uso: $0 <dump_file>"
[[ -f "$DUMP_FILE" ]] || fail "Archivo no existe: $DUMP_FILE"

set -a; . ./.env; set +a

warn "ADVERTENCIA: esto DROP y restaura DB '${POSTGRES_DB}' completa."
read -r -p "¿Continuar? (escribe 'restore'): " confirm
[[ "$confirm" == "restore" ]] || fail "Cancelado"

echo "▶ Deteniendo servicios que dependen de PG..."
docker compose stop api worker-ingest worker-ml worker-scrape prefect mlflow telegram 2>/dev/null || true

echo "▶ Drop + recreate database..."
docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d postgres <<SQL
DROP DATABASE IF EXISTS ${POSTGRES_DB};
CREATE DATABASE ${POSTGRES_DB} OWNER ${POSTGRES_USER};
SQL

echo "▶ Restoring pg_dump ($(du -h "$DUMP_FILE" | cut -f1))..."
docker compose exec -T postgres pg_restore \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  --no-owner --no-privileges \
  --verbose < "$DUMP_FILE" 2>&1 | tail -30

echo "▶ Re-crear extensions..."
docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" <<SQL
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS unaccent;
SQL

echo "▶ Relanzando servicios..."
docker compose up -d

ok "Restore completo. Verifica: make smoke-test"
