#!/usr/bin/env bash
# Arranca la TUI del bot en terminal. Muestra dashboard+bankroll+drift en vivo.
# Uso:  make tui  (o: bash scripts/tui.sh)

set -euo pipefail
cd "$(dirname "$0")/.."

# Usar venv de test con deps instaladas (cp313 hasta que cp314 wheels estén listas)
VENV="${VENV:-/tmp/test-venv}"
if [ ! -d "$VENV" ]; then
    echo "❌ No hay venv en $VENV. Correr primero uv sync o crear uno."
    exit 1
fi

# Cargar variables .env
set -a
# shellcheck disable=SC1091
source .env 2>/dev/null || true
set +a

# Override localhost para ejecución fuera de Docker
export POSTGRES_HOST="${POSTGRES_HOST_OVERRIDE:-localhost}"
export POSTGRES_PORT="${POSTGRES_HOST_PORT:-5434}"
export PYTHONPATH="src"

echo "▶ Arrancando TUI Textual..."
echo "  DB:       localhost:${POSTGRES_PORT}"
echo "  LLM:      ${LLM_BACKEND:-llama_local}"
echo "  Tip:      'q' para salir, tabs para navegar pantallas"
echo ""

exec "$VENV/bin/python" -m apuestas.tui
