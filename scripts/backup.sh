#!/usr/bin/env bash
# backup.sh — pg_dump + snapshot MinIO.

set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-./backups}"
TS=$(date +%F_%H%M)
mkdir -p "$BACKUP_DIR"

set -a; . ./.env; set +a

# Postgres dump (custom format, compresado)
echo "▶ pg_dump → $BACKUP_DIR/pg_$TS.dump"
docker compose exec -T postgres pg_dump \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  -F c -Z 6 --no-owner --no-privileges \
  > "$BACKUP_DIR/pg_$TS.dump"

size=$(du -h "$BACKUP_DIR/pg_$TS.dump" | cut -f1)
echo "✓ Postgres dump ($size)"

# Retención: keep últimos 14 días, compresión adicional para viejos
find "$BACKUP_DIR" -name "pg_*.dump" -mtime +14 -delete 2>/dev/null || true

# Manifest
cat > "$BACKUP_DIR/pg_$TS.manifest.json" <<EOF
{
  "timestamp": "$TS",
  "dump_file": "pg_$TS.dump",
  "size_bytes": $(stat -c%s "$BACKUP_DIR/pg_$TS.dump" 2>/dev/null || stat -f%z "$BACKUP_DIR/pg_$TS.dump"),
  "db_name": "${POSTGRES_DB}",
  "postgres_version": "$(docker compose exec -T postgres psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc 'SHOW server_version' | tr -d '\r')"
}
EOF

echo "✅ Backup completo en $BACKUP_DIR/pg_$TS.dump"
