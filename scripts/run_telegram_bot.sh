#!/usr/bin/env bash
# Wrapper para systemd: sanea env y arranca el bot.
set -u
cd "$(dirname "$0")/.."

unset PYTHON_GIL

# Overrides para correr desde host (no container):
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5434
export DATABASE_URL="postgresql+asyncpg://apuestas:change-me-to-long-random-string@localhost:5434/apuestas"
export MLFLOW_TRACKING_URI=http://localhost:5000
export PREFECT_API_URL=http://localhost:4200/api
export PYTHONPATH=src

exec .venv/bin/python -m apuestas.bot.telegram
