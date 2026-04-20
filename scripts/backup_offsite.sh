#!/usr/bin/env bash
# backup_offsite.sh — Sync backups/ a Backblaze B2 via rclone.
# Requiere rclone configurado con remote "b2:" apuntando a Backblaze.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

command -v rclone >/dev/null || fail "rclone no instalado — curl https://rclone.org/install.sh | sudo bash"

REMOTE="${RCLONE_REMOTE:-b2:apuestas-backups}"
LOCAL_DIR="backups"

# Validar remote existe
if ! rclone lsd "$REMOTE" >/dev/null 2>&1; then
  fail "Remote '$REMOTE' no accesible. Config con: rclone config"
fi

echo "▶ Sync '$LOCAL_DIR/' → '$REMOTE'..."
rclone sync \
  --progress \
  --transfers 4 \
  --exclude '*.tmp' \
  --exclude 'pg_*.manifest.json' \
  "$LOCAL_DIR/" "$REMOTE/" 2>&1 | tail -20

# Aplicar lifecycle: retención 90 días
echo "▶ Purgando offsite >90 días..."
rclone delete --min-age 90d "$REMOTE/"

size=$(rclone size "$REMOTE" --json 2>/dev/null | grep -o '"bytes":[0-9]*' | head -1 | cut -d: -f2)
if [[ -n "$size" ]]; then
  ok "Offsite size: $(numfmt --to=iec "$size")"
fi
ok "Backup offsite completado"
