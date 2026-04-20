#!/usr/bin/env bash
# trivy_scan.sh — Escaneo vulnerabilidades imágenes Docker con Trivy.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

command -v trivy >/dev/null || fail "Trivy no instalado — https://aquasecurity.github.io/trivy/"

mkdir -p reports/trivy
ts=$(date +%F)

# 1. Scan filesystem
echo "▶ Trivy filesystem scan..."
trivy fs \
  --severity CRITICAL,HIGH \
  --ignore-unfixed \
  --format table \
  --output "reports/trivy/fs_${ts}.txt" \
  . || warn "Vulnerabilidades encontradas (fs)"
ok "FS report: reports/trivy/fs_${ts}.txt"

# 2. Scan imagen runtime
if docker images apuestas --format "{{.Repository}}:{{.Tag}}" | grep -q apuestas; then
  echo "▶ Trivy image scan (apuestas:latest)..."
  trivy image \
    --severity CRITICAL,HIGH \
    --ignore-unfixed \
    --format table \
    --output "reports/trivy/image_${ts}.txt" \
    apuestas:latest || warn "Vulnerabilidades encontradas (image)"
  ok "Image report: reports/trivy/image_${ts}.txt"
else
  warn "Imagen 'apuestas:latest' no encontrada, skip image scan"
fi

# 3. SBOM CycloneDX
echo "▶ Trivy SBOM..."
trivy fs \
  --format cyclonedx \
  --output "reports/trivy/sbom_${ts}.json" \
  .
ok "SBOM: reports/trivy/sbom_${ts}.json"

echo ""
ok "Trivy scan completo. Reports en reports/trivy/"
