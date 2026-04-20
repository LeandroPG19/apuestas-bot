#!/usr/bin/env bash
# generate_sbom.sh — SBOM Python + Docker imagen (syft + cyclonedx-py).

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok() { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

mkdir -p reports/sbom
ts=$(date +%F)

# 1. Python deps SBOM (cyclonedx-py)
if command -v cyclonedx-py >/dev/null 2>&1; then
  echo "▶ SBOM Python deps..."
  cyclonedx-py environment \
    --output-format json \
    --output-file "reports/sbom/python_${ts}.cdx.json"
  ok "Python SBOM: reports/sbom/python_${ts}.cdx.json"
else
  warn "cyclonedx-py no instalado — pip install cyclonedx-bom"
fi

# 2. Docker image SBOM (syft)
if command -v syft >/dev/null 2>&1; then
  if docker images apuestas --format "{{.Repository}}" | grep -q apuestas; then
    echo "▶ SBOM Docker image con syft..."
    syft apuestas:latest \
      -o spdx-json="reports/sbom/image_spdx_${ts}.json" \
      -o cyclonedx-json="reports/sbom/image_cdx_${ts}.json"
    ok "Docker image SBOM: reports/sbom/image_*_${ts}.json"
  else
    warn "Imagen apuestas:latest no encontrada — docker compose build primero"
  fi
else
  warn "syft no instalado — curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh"
fi

# 3. Licenses
if uv run pip-licenses --version >/dev/null 2>&1; then
  echo "▶ License audit..."
  uv run pip-licenses --format=markdown --output-file="reports/sbom/licenses_${ts}.md"
  ok "Licenses report: reports/sbom/licenses_${ts}.md"
fi

echo ""
ok "SBOM generation completa."
