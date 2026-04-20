#!/usr/bin/env bash
# audit_deps.sh — Verifica que las versiones pineadas en pyproject.toml
# sigan siendo las más recientes y seguras al día de hoy.
#
# Integra: uv lock --upgrade + pip-audit + Context7 queries para las
# librerías críticas. Ejecutar trimestralmente o antes de releases.

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; }

echo "── Audit de dependencias — $(date +%F) ──"
echo ""

# ─── Check 1: uv lock está fresco ─────────────────────────────────────────
if [[ -f uv.lock ]]; then
    lock_age_days=$(( ( $(date +%s) - $(stat -c%Y uv.lock 2>/dev/null || stat -f%m uv.lock) ) / 86400 ))
    if (( lock_age_days > 90 )); then
        warn "uv.lock tiene ${lock_age_days} días — considera `uv lock --upgrade`"
    else
        ok "uv.lock reciente (${lock_age_days} días)"
    fi
else
    warn "uv.lock no existe — ejecuta `uv lock` primero"
fi

# ─── Check 2: Upgrade disponible (dry-run) ────────────────────────────────
echo ""
echo "▶ Chequeando upgrades disponibles..."
if command -v uv >/dev/null 2>&1; then
    if uv lock --upgrade --dry-run 2>&1 | tee /tmp/uv_upgrade_$$.log | grep -qE "Would upgrade|would be upgraded"; then
        warn "Hay upgrades disponibles:"
        grep -E "Would upgrade|would be upgraded" /tmp/uv_upgrade_$$.log | head -20
    else
        ok "Todas las deps están en su última versión compatible"
    fi
    rm -f /tmp/uv_upgrade_$$.log
else
    fail "uv no instalado — instala con: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# ─── Check 3: Vulnerabilidades conocidas ──────────────────────────────────
echo ""
echo "▶ Ejecutando pip-audit..."
if uv run pip-audit --desc --strict 2>&1 | tail -30; then
    ok "pip-audit completado"
else
    fail "pip-audit encontró vulnerabilidades (ver arriba)"
fi

# ─── Check 4: Python version match ────────────────────────────────────────
echo ""
py_version=$(grep 'requires-python' pyproject.toml | awk -F'"' '{print $2}')
echo "▶ Python requerido: $py_version"
if python3.14 --version >/dev/null 2>&1; then
    ok "python3.14 disponible"
    if python3.14t --version >/dev/null 2>&1; then
        ok "python3.14t (free-threaded) disponible"
    else
        warn "python3.14t NO disponible — free-threading caerá a GIL"
    fi
else
    fail "python3.14 NO instalado — uv lo descargará al hacer venv"
fi

# ─── Check 5: Librerías críticas vía Context7 (manual) ────────────────────
echo ""
echo "▶ Verificación manual requerida (Context7 queries):"
echo "   Ejecuta en Claude Code o agent:"
echo "   - Context7 query /fastapi/fastapi → latest stable"
echo "   - Context7 query /pola-rs/polars → latest stable"
echo "   - Context7 query /mlflow/mlflow → latest stable"
echo "   - Context7 query /lightgbm-org/LightGBM → latest stable"
echo "   - Context7 query /dmlc/xgboost → latest stable"
echo ""
echo "── Audit completo ──"
