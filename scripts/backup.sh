#!/usr/bin/env bash
# backup.sh — full backup: Postgres + MinIO artifacts + config.
#
# Output: ./backups/YYYYMMDD_HHMM/
#   pg_<ts>.dump             Postgres custom format compressed
#   minio_<ts>.tar.gz        MinIO artifacts (MLflow models + parquet)
#   env_<ts>.txt             Config snapshot (redacted)
#   models_<ts>.txt          Production models list
#   manifest_<ts>.json       Metadata
#
# Retention: 14 días local.
# Offsite opcional via `make backup-offsite` → Backblaze B2.

set -euo pipefail

cd "$(dirname "$0")/.."
BACKUP_DIR="${BACKUP_DIR:-./backups}"
TS=$(date +%Y%m%d_%H%M)
DEST="$BACKUP_DIR/$TS"
mkdir -p "$DEST"

# Cargar .env (parser robusto: ignora comments + líneas con espacios sin quotes)
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        case "$key" in
            \#*|"") continue ;;
            *)
                # Solo export keys válidos (identifier format)
                if [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
                    export "$key=$value"
                fi
                ;;
        esac
    done < .env
fi

echo "▶ Backup iniciado: $TS → $DEST"

# ─── 1. Postgres dump ────────────────────────────────────────
echo "  [1/4] pg_dump Postgres..."
docker compose exec -T postgres pg_dump \
    -U "${POSTGRES_USER:-apuestas}" \
    -d "${POSTGRES_DB:-apuestas}" \
    -F c -Z 6 --no-owner --no-privileges \
    > "$DEST/pg_$TS.dump" 2>/dev/null
pg_size=$(du -h "$DEST/pg_$TS.dump" | cut -f1)
echo "    ✓ Postgres ($pg_size)"

# ─── 2. MinIO artifacts via docker cp (minio no tiene tar) ───
echo "  [2/4] MinIO artifacts..."
TMP_MINIO=$(mktemp -d)
trap "rm -rf $TMP_MINIO" EXIT
docker cp apuestas-minio:/data/mlflow-artifacts "$TMP_MINIO/" 2>/dev/null || true
docker cp apuestas-minio:/data/parquet-cold "$TMP_MINIO/" 2>/dev/null || true
docker cp apuestas-minio:/data/scrapes-raw "$TMP_MINIO/" 2>/dev/null || true
if [ -d "$TMP_MINIO/mlflow-artifacts" ] || [ -d "$TMP_MINIO/parquet-cold" ]; then
    tar czf "$DEST/minio_$TS.tar.gz" -C "$TMP_MINIO" . 2>/dev/null
    minio_size=$(du -h "$DEST/minio_$TS.tar.gz" | cut -f1)
    echo "    ✓ MinIO ($minio_size)"
else
    echo "    ⚠ MinIO backup fail (no data)"
fi
rm -rf "$TMP_MINIO"
trap - EXIT

# ─── 3. Config snapshot ──────────────────────────────────────
echo "  [3/4] Config snapshot..."
grep -v -E "SECRET|TOKEN|PASSWORD|KEY" .env 2>/dev/null \
    > "$DEST/env_$TS.txt" || true
cp docker-compose.yml "$DEST/docker-compose_$TS.yml" 2>/dev/null || true
docker compose exec -T postgres psql \
    -U "${POSTGRES_USER:-apuestas}" \
    -d "${POSTGRES_DB:-apuestas}" \
    -c "SELECT model_name, stage, sport_code, model_version, promoted_at
        FROM model_registry_meta ORDER BY promoted_at DESC" \
    > "$DEST/models_$TS.txt" 2>/dev/null || true
echo "    ✓ Config + models list"

# ─── 4. Manifest JSON ────────────────────────────────────────
echo "  [4/4] Manifest..."
pg_bytes=$(stat -c%s "$DEST/pg_$TS.dump" 2>/dev/null || echo 0)
minio_bytes=$(stat -c%s "$DEST/minio_$TS.tar.gz" 2>/dev/null || echo 0)
cat > "$DEST/manifest_$TS.json" <<EOF
{
  "timestamp": "$TS",
  "db_name": "${POSTGRES_DB:-apuestas}",
  "files": {
    "postgres": "pg_$TS.dump",
    "minio": "minio_$TS.tar.gz",
    "env": "env_$TS.txt",
    "models": "models_$TS.txt"
  },
  "sizes_bytes": {
    "postgres": $pg_bytes,
    "minio": $minio_bytes
  },
  "hostname": "$(hostname)",
  "created_at": "$(date -Iseconds)"
}
EOF
echo "    ✓ Manifest OK"

# ─── Retención (14 días) ─────────────────────────────────────
find "$BACKUP_DIR" -maxdepth 1 -type d -name "20*" -mtime +14 \
    -exec rm -rf {} + 2>/dev/null || true

total_size=$(du -sh "$DEST" | cut -f1)
echo ""
echo "✅ Backup completo → $DEST ($total_size)"
echo "   Retención: últimos 14 días en $BACKUP_DIR"
echo "   Off-site: \`make backup-offsite\` para subir a B2"
