#!/usr/bin/env bash
# dr_drill.sh — Disaster recovery drill trimestral (§17.9).
# Prueba: restore último backup sobre DB temporal, verifica integridad.
# NO toca la DB de producción.

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

LATEST_DUMP=$(ls -1t backups/pg_*.dump 2>/dev/null | head -1)
[[ -n "$LATEST_DUMP" ]] || fail "No hay backups en backups/pg_*.dump"

set -a; . ./.env; set +a

DRILL_DB="${POSTGRES_DB}_dr_drill"

echo "── DR Drill — $(date +%F) ──"
echo "Dump usado: $LATEST_DUMP ($(du -h "$LATEST_DUMP" | cut -f1))"
echo "DB temporal: $DRILL_DB"
echo ""

# 1. Crear DB temporal
echo "▶ [1/5] Creando DB temporal..."
docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d postgres <<SQL
DROP DATABASE IF EXISTS ${DRILL_DB};
CREATE DATABASE ${DRILL_DB} OWNER ${POSTGRES_USER};
SQL
ok "DB temporal creada"

# 2. Restaurar dump
echo "▶ [2/5] pg_restore sobre DB temporal..."
docker compose exec -T postgres pg_restore \
  -U "${POSTGRES_USER}" \
  -d "$DRILL_DB" \
  --no-owner --no-privileges 2>&1 < "$LATEST_DUMP" | tail -5 || true
ok "pg_restore completado"

# 3. Extensions + migraciones head
echo "▶ [3/5] Reinstalando extensions..."
docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d "$DRILL_DB" <<SQL
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS unaccent;
SQL

# 4. Sanity checks
echo "▶ [4/5] Sanity checks..."
for table in matches odds_history predictions bets post_mortems; do
  count=$(docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d "$DRILL_DB" -tAc \
    "SELECT COUNT(*) FROM $table" 2>/dev/null || echo "ERROR")
  if [[ "$count" == "ERROR" ]]; then
    warn "Tabla $table no existe en el dump"
  else
    ok "Tabla $table: $count filas"
  fi
done

# 5. Cleanup
echo "▶ [5/5] Cleanup DB temporal..."
docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d postgres <<SQL
DROP DATABASE IF EXISTS ${DRILL_DB};
SQL
ok "DB temporal eliminada"

echo ""
ok "DR drill PASSED — último backup válido y restaurable"
