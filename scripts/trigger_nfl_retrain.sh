#!/usr/bin/env bash
# Disparo manual del retrain NFL (Deuda 6).
# Uso: bash scripts/trigger_nfl_retrain.sh
# Útil cuando el backtest walk-forward deja clara la necesidad de retrain
# antes de que ADWIN/Page-Hinkley lo detecten automáticamente.

set -euo pipefail
cd "$(dirname "$0")/.."

export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5434}"

SPORT="${1:-nfl}"

echo "═══════════════════════════════════════════════════════════"
echo "  Retrain manual para sport=${SPORT}"
echo "═══════════════════════════════════════════════════════════"

uv run python -m apuestas.flows.retrain_on_drift --sport "${SPORT}"
