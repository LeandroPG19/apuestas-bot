.DEFAULT_GOAL := help
.PHONY: help install cold-start up down restart status logs shell analyze \
        migrate makemigration rollback seed smoke-test test lint typecheck format \
        backup backup-offsite restore capture-on capture-off capture-status \
        retrain promote rollback-model notebook profile-worker \
        chaos dr-drill audit-python audit-images sbom clean \
        install-nvidia-toolkit install-models install-mcp build ps stats \
        live-scores settle-bets post-mortem fiscal-export

COMPOSE := docker compose
COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

## ─── Ayuda ──────────────────────────────────────────────────────────────────

help: ## Muestra esta ayuda
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mComandos disponibles:\033[0m\n"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } \
		/^##[^#]/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 4) }' $(MAKEFILE_LIST)
	@echo ""

## ─── Instalación y primera vez ──────────────────────────────────────────────

install: ## Wizard de instalación completa (nvidia toolkit, modelos, .env)
	@echo "▶ Paso 1/5: Verificar prerequisitos"
	@bash scripts/check_prereqs.sh
	@echo "▶ Paso 2/5: Instalar nvidia-container-toolkit (requiere sudo)"
	@$(MAKE) install-nvidia-toolkit
	@echo "▶ Paso 3/5: Descargar modelos GGUF (Qwen 2.5 7B + BGE-M3)"
	@$(MAKE) install-models
	@echo "▶ Paso 4/5: Instalar servidores MCP (cuba-memorys + cuba-search)"
	@$(MAKE) install-mcp
	@echo "▶ Paso 5/5: Generar .env desde .env.example"
	@[ -f .env ] || cp .env.example .env
	@echo "✅ Instalación base completa. Edita .env con tus API keys, luego: make cold-start"

install-nvidia-toolkit: ## Instala nvidia-container-toolkit (requiere sudo)
	@bash scripts/install_nvidia_toolkit.sh

install-models: ## Descarga Qwen 2.5 7B Q4_K_M a ./models/
	@bash scripts/download_models.sh

install-mcp: ## Registra servidores MCP cuba-memorys y cuba-search (si existen)
	@bash scripts/install_mcp.sh

cold-start: ## Build imágenes + migrate + seed inicial (primera vez, ~30 min)
	@[ -f .env ] || (echo "❌ Falta .env. Ejecuta: make install"; exit 1)
	@echo "▶ Pulling imágenes base..."
	$(COMPOSE) pull
	@echo "▶ Building imágenes custom..."
	$(COMPOSE) build
	@echo "▶ Levantando servicios de datos..."
	$(COMPOSE) up -d postgres valkey minio
	@sleep 8
	@echo "▶ Aplicando migraciones Alembic..."
	$(MAKE) migrate
	@echo "▶ Levantando stack completo..."
	$(MAKE) up
	@echo "✅ Cold start completo. Verifica: make status"

## ─── Ciclo operativo ────────────────────────────────────────────────────────

up: ## Levanta el stack completo (requiere GPU + cold-start previo)
	$(COMPOSE_GPU) up -d
	@echo "✅ Stack arriba. make status | make logs"

down: ## Apaga el stack con shutdown graceful
	@echo "▶ Shutdown graceful (flush caches, checkpoint DB)..."
	$(COMPOSE_GPU) stop --timeout 30
	$(COMPOSE_GPU) down --remove-orphans
	@echo "✅ Stack apagado limpio."

restart: down up ## Reinicia el stack

status: ## Estado servicios + VRAM + picks activos
	@echo "─── Contenedores ───"
	@$(COMPOSE_GPU) ps
	@echo ""
	@echo "─── GPU ───"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader || echo "GPU no disponible"
	@echo ""
	@echo "─── RAM contenedores ───"
	@docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}" | head -30

ps: ## Alias de status
	@$(COMPOSE_GPU) ps

stats: ## docker stats stream
	@docker stats

logs: ## Logs de todos los servicios (tail -f)
	$(COMPOSE_GPU) logs -f --tail=200

shell: ## Abre shell en contenedor api
	$(COMPOSE) exec api bash

## ─── Análisis on-demand ─────────────────────────────────────────────────────

analyze: ## Ejecuta análisis 360° completo de eventos próximos 48 h
	@echo "▶ Ejecutando deep_analysis flow..."
	$(COMPOSE) exec api python -m apuestas.flows.deep_analysis
	@echo "✅ Reporte enviado a Telegram + dashboard."

tui: ## 🎯 ÚNICA interfaz: TUI Textual (dashboard + picks + bankroll + post-mortems + calibración)
	@bash scripts/tui.sh

tui-dev: ## TUI en modo dev con hot-reload (textual-dev)
	VENV=$${VENV:-/tmp/test-venv} $$VENV/bin/textual run --dev apuestas.tui.app:ApuestasTUI

## ─── Base de datos ──────────────────────────────────────────────────────────

migrate: ## Aplica migraciones Alembic pendientes
	$(COMPOSE) exec api alembic upgrade head

makemigration: ## Genera nueva migración ej. make makemigration MSG="add_foo"
	@[ -n "$(MSG)" ] || (echo "❌ Uso: make makemigration MSG=\"descripción\""; exit 1)
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(MSG)"

rollback: ## Rollback última migración
	$(COMPOSE) exec api alembic downgrade -1

seed: ## Carga datos históricos iniciales (NBA 5y, MLB 3y, NFL 5y, EPL 3y)
	$(COMPOSE) exec api python -m apuestas.scripts.seed_historical

## ─── Testing y QA ───────────────────────────────────────────────────────────

smoke-test: ## Verifica health de todos los servicios
	@bash scripts/smoke_test.sh

test: ## Corre suite pytest completa
	$(COMPOSE) exec api pytest -x -v

test-unit: ## Solo tests unitarios
	$(COMPOSE) exec api pytest tests/unit -x -v

test-integration: ## Tests con testcontainers
	$(COMPOSE) exec api pytest tests/integration -x -v -m integration

test-e2e: ## Tests end-to-end con fixtures
	$(COMPOSE) exec api pytest tests/e2e -x -v -m e2e

lint: ## ruff check
	uv run ruff check src tests

typecheck: ## mypy strict
	uv run mypy src

format: ## ruff format
	uv run ruff format src tests
	uv run ruff check --fix src tests

## ─── Modelos ML ─────────────────────────────────────────────────────────────

retrain: ## Retrain todos los modelos (o SPORT=nba)
	$(COMPOSE) exec api python -m apuestas.flows.retrain_weekly $(if $(SPORT),--sport $(SPORT),)

promote: ## Promover shadow → production si CLV mejora
	@[ -n "$(MODEL)" ] || (echo "❌ Uso: make promote MODEL=nba_moneyline"; exit 1)
	$(COMPOSE) exec api python -m apuestas.ml.registry promote --model $(MODEL)

rollback-model: ## Rollback modelo a versión X
	@[ -n "$(MODEL)" ] && [ -n "$(VERSION)" ] || (echo "❌ Uso: make rollback-model MODEL=nba_moneyline VERSION=3"; exit 1)
	$(COMPOSE) exec api python -m apuestas.ml.registry rollback --model $(MODEL) --version $(VERSION)

## ─── Backtest y análisis ────────────────────────────────────────────────────

backtest: ## Backtest walk-forward ej. make backtest SPORT=nba SEASONS=2023-24,2024-25
	$(COMPOSE) exec api python -m apuestas.ml.backtest --sport $(SPORT) --seasons "$(SEASONS)"

notebook: ## Ejecuta notebook parametrizado ej. make notebook NB=05_pick_postmortem BET_ID=123
	@[ -n "$(NB)" ] || (echo "❌ Uso: make notebook NB=05_pick_postmortem [BET_ID=X]"; exit 1)
	$(COMPOSE) exec api papermill notebooks/$(NB).ipynb reports/notebooks/$(NB)_$(shell date +%F).ipynb $(if $(BET_ID),-p bet_id $(BET_ID),)

## ─── Settlement + post-match ────────────────────────────────────────────────

live-scores: ## Ingesta scores de partidos finalizados últimas 48h
	$(COMPOSE) exec api python -m apuestas.flows.live_scores

settle-bets: ## Liquida bets pendientes con match finished + post_mortem
	$(COMPOSE) exec api python -m apuestas.flows.settle_bets

post-mortem: ## Genera post_mortems para bets settled sin PM
	$(COMPOSE) exec api python -m apuestas.flows.post_mortem

## ─── Fiscal MX ──────────────────────────────────────────────────────────────

fiscal-export: ## Exporta CSV SAT ej. make fiscal-export MONTH=2026-04 UNIT_MXN=100
	@[ -n "$(MONTH)" ] || (echo "❌ Uso: make fiscal-export MONTH=YYYY-MM [UNIT_MXN=100]"; exit 1)
	$(COMPOSE) exec api python -m apuestas.reports.fiscal_exporter --month $(MONTH) --unit-mxn $(or $(UNIT_MXN),100)

## ─── Capture job (CLV independiente) ────────────────────────────────────────

capture-on: ## Activa capture job permanente de closing line (systemd user)
	systemctl --user daemon-reload
	systemctl --user enable --now apuestas-capture.timer
	@echo "✅ Capture timer activo. Ver: systemctl --user status apuestas-capture.timer"

capture-off: ## Desactiva capture job
	systemctl --user disable --now apuestas-capture.timer

capture-status: ## Estado del capture timer
	@systemctl --user status apuestas-capture.timer || echo "Timer no instalado. make capture-on primero."

## ─── Backups ────────────────────────────────────────────────────────────────

backup: ## pg_dump + snapshot MinIO a ./backups/
	@bash scripts/backup.sh

backup-offsite: ## rclone sync ./backups/ a Backblaze B2
	@bash scripts/backup_offsite.sh

restore: ## Restore desde backup ej. make restore FILE=backups/pg_2026-04-19.dump
	@[ -n "$(FILE)" ] || (echo "❌ Uso: make restore FILE=backups/pg_YYYY-MM-DD.dump"; exit 1)
	@bash scripts/restore.sh $(FILE)

## ─── Performance y profiling ────────────────────────────────────────────────

profile-worker: ## py-spy flame graph de un worker ej. make profile-worker PID=1234
	@[ -n "$(PID)" ] || (echo "❌ Uso: make profile-worker PID=<pid>"; exit 1)
	$(COMPOSE) exec worker-ml py-spy record -o /tmp/flame_$(PID).svg --pid $(PID) --duration 60

## ─── Chaos engineering y DR ─────────────────────────────────────────────────

chaos: ## Mata un contenedor aleatorio (workers) para probar resiliencia
	@bash scripts/chaos.sh

dr-drill: ## Disaster recovery drill (restore a DB temporal)
	@bash scripts/dr_drill.sh

## ─── Seguridad ──────────────────────────────────────────────────────────────

audit-python: ## pip-audit de vulnerabilidades en deps
	uv run pip-audit --desc

audit-images: ## Trivy scan de imágenes Docker
	@bash scripts/trivy_scan.sh

audit-deps: ## Audit completo: uv lock check + pip-audit + Context7 reminders
	@bash scripts/audit_deps.sh

sbom: ## Genera SBOM con syft
	@bash scripts/generate_sbom.sh

## ─── CI/CD local ────────────────────────────────────────────────────────────

install-hooks: ## Instala pre-commit hooks (una sola vez tras clone)
	uv run pre-commit install --install-hooks
	uv run pre-commit install --hook-type commit-msg --hook-type pre-push
	@echo "✅ Hooks instalados. Ejecutar en todos los archivos: make pre-commit-all"

pre-commit: ## Corre pre-commit sobre archivos en staging
	uv run pre-commit run

pre-commit-all: ## Corre pre-commit sobre TODOS los archivos del repo
	uv run pre-commit run --all-files

secrets-baseline: ## Regenera .secrets.baseline inicial
	uv run detect-secrets scan --baseline .secrets.baseline
	@echo "✅ Baseline regenerado. Revisar antes de commit."

ci-local: lint typecheck test ## Ejecuta la misma suite que CI de GitHub localmente
	@echo "✅ CI local verde"

## ─── Utilidades ─────────────────────────────────────────────────────────────

build: ## (Re)build imágenes Docker
	$(COMPOSE) build

clean: ## Limpia pycache, coverage, caches ruff/mypy
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .coverage reports/coverage
	@echo "✅ Limpio."
