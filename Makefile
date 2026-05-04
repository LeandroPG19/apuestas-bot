.DEFAULT_GOAL := help
.PHONY: help install cold-start up down restart status logs shell analyze \
        migrate makemigration rollback seed smoke-test test lint typecheck format \
        backup backup-offsite restore capture-on capture-off capture-status \
        retrain promote rollback-model notebook profile-worker \
        chaos dr-drill audit-python audit-images sbom clean \
        install-nvidia-toolkit install-models install-mcp build ps stats \
        live-scores

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

tui: ## 🎯 UN SOLO COMANDO: arranca todo + abre TUI + al salir pregunta si apagar
	@bash scripts/tui.sh

tui-persist: ## TUI pero deja servicios corriendo al cerrar (modo 24/7)
	@bash scripts/tui.sh --no-start

tui-full-stop: ## TUI que apaga TODO al cerrarla (Docker + bot + timer)
	@bash scripts/tui.sh --stop-on-exit

go: ## 🚀 Arranca stack background sin TUI (Docker + bot + timer 6h)
	@bash scripts/start.sh

stop: ## 🛑 Detiene bot + auto-análisis (deja Docker base up)
	@bash scripts/stop.sh

operate: ## 🎮 CLI operativa (status | catchup | analyze | full | picks | bet ID ODDS)
	@bash scripts/operate.sh $(filter-out $@,$(MAKECMDGOALS))

telegram-setup: ## 🤖 Setup interactivo del bot Telegram (pide token → captura chat_id → escribe .env)
	@.venv/bin/python scripts/setup_telegram.py

autopilot: ## 🚀 Lanza seed + retrain + backtest en background. Notifica a Telegram cada etapa.
	@mkdir -p logs
	@nohup bash scripts/autopilot.sh > logs/autopilot.log 2>&1 & disown; \
	 echo "🤖 Autopilot corriendo en background"; \
	 echo "   Logs: tail -f logs/autopilot.log"

autopilot-status: ## Estado del autopilot
	@if pgrep -f autopilot.sh >/dev/null; then \
	   echo "✅ corriendo (pid $$(pgrep -f autopilot.sh | head -1))"; \
	   tail -15 logs/autopilot.log 2>/dev/null; \
	 else \
	   echo "⚠  no corriendo"; \
	   tail -5 logs/autopilot.log 2>/dev/null; \
	 fi

telegram-logs: ## 📋 Logs en vivo del bot Telegram
	@tail -f logs/telegram.log

analyze-logs: ## 📋 Logs en vivo del auto-análisis
	@tail -f logs/analyze.log

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

seed: ## Carga datos históricos (make seed SPORT=nba SEASONS=2023,2024,2025)
	$(COMPOSE) exec api python -m apuestas.scripts.seed_historical \
		--sport $(or $(SPORT),nba) \
		--seasons $(or $(SEASONS),2024,2025) \
		$(if $(LEAGUE),--league $(LEAGUE),)

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

retrain-sprint11: ## Retrain con stack Sprint 11 completo (stacker LGBM + focal + TabPFN opcional)
	@echo "🚀 Sprint 11 stack: market_stacker + focal + book_power + xT/clutch/stuff+"
	APUESTAS_USE_MARKET_STACKER=true \
	APUESTAS_USE_FOCAL_LOSS=true \
	APUESTAS_ENABLE_XT=true \
	APUESTAS_ENABLE_NBA_CLUTCH=true \
	APUESTAS_ENABLE_MLB_STUFF_PLUS=true \
	APUESTAS_USE_BOOK_POWER=true \
	APUESTAS_SPRINT11_SOFT_TAGS=true \
	APUESTAS_MLB_POISSON_ENSEMBLE=true \
	uv run python -m apuestas.flows.retrain_weekly $(if $(SPORT),--sport $(SPORT),)

ablation-elo: ## Ablation Elo on/off para un sport (make ablation-elo SPORT=nba YEARS=2023-24,2024-25)
	@[ -n "$(SPORT)" ] || (echo "❌ Uso: make ablation-elo SPORT=nba YEARS=2023-24,2024-25"; exit 1)
	@[ -n "$(YEARS)" ] || (echo "❌ Falta YEARS"; exit 1)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/retrain_elo_ablation.py --sport $(SPORT) --years $(YEARS)

backtest-dc: ## Backtest Dixon-Coles vs baseline (make backtest-dc LEAGUE=4 SEASONS=2023-24,2024-25)
	@[ -n "$(LEAGUE)" ] || (echo "❌ Uso: make backtest-dc LEAGUE=4 SEASONS=2023-24,2024-25"; exit 1)
	@[ -n "$(SEASONS)" ] || (echo "❌ Falta SEASONS"; exit 1)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/backtest_dc_vs_lgbm.py --league $(LEAGUE) --seasons $(SEASONS)

ingest-football-data: ## Bulk odds soccer football-data.co.uk (18 ligas 2018+)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/ingest_football_data_co_uk.py $(if $(SINCE),--since $(SINCE),)

ingest-clubelo: ## Elo soccer pre-calculado clubelo.com (ratings diarios)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/ingest_clubelo.py --since $(if $(SINCE),$(SINCE),2018-01-01) --step-days 14

ingest-statsbomb: ## Event-level soccer StatsBomb Open Data (75 competitions)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/ingest_statsbomb_open.py

ingest-nflfastr: ## NFL play-by-play EPA/CPOE nflfastR
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/ingest_nflfastr.py --since $(if $(SINCE),$(SINCE),2018)

ingest-sackmann-tennis: ## Tennis ATP+WTA stats desde Jeff Sackmann GitHub
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/ingest_sackmann_tennis.py --since $(if $(SINCE),$(SINCE),2018)

ingest-fangraphs: ## MLB FanGraphs team stats (wRC+/FIP/xFIP/WAR)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/ingest_fangraphs_team.py --since $(if $(SINCE),$(SINCE),2018)

ingest-all-bulk: ## Fase 1: todas las fuentes bulk (football-data+clubelo+statsbomb+nflfastR+sackmann+fangraphs)
	@echo "🚀 Fase 1 bulk ingest — ~30 min total"
	$(MAKE) ingest-football-data
	$(MAKE) ingest-clubelo
	$(MAKE) ingest-statsbomb
	$(MAKE) ingest-nflfastr
	$(MAKE) ingest-sackmann-tennis
	$(MAKE) ingest-fangraphs

ingest-data-status: ## Muestra conteo de filas por fuente
	docker exec apuestas-postgres psql -U apuestas -d apuestas -c "\
	SELECT 'football-data' AS src, COUNT(*)::bigint AS n FROM odds_history_archive \
	UNION ALL SELECT 'clubelo', COUNT(*) FROM team_elo_daily WHERE source='clubelo' \
	UNION ALL SELECT 'statsbomb', COUNT(*) FROM statsbomb_events \
	UNION ALL SELECT 'nfl_epa', COUNT(*) FROM nfl_epa_plays \
	UNION ALL SELECT 'sackmann', COUNT(*) FROM tennis_matches_sackmann \
	UNION ALL SELECT 'fangraphs', COUNT(*) FROM fangraphs_team_stats_daily \
	UNION ALL SELECT 'pitcher_statcast', COUNT(*) FROM pitcher_game_stats \
	UNION ALL SELECT 'nba_pbp', COUNT(*) FROM play_by_play WHERE sport_code='nba' \
	UNION ALL SELECT 'book_power_ratings', COUNT(*) FROM team_elo_daily WHERE source != 'clubelo' \
	ORDER BY n DESC;"

ingest-nba-pbp: ## Ingesta PBP NBA (make ingest-nba-pbp DATE=2024-12-25 o START=... END=...)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python -m apuestas.ingest.nba_pbp $(if $(DATE),--date $(DATE),) $(if $(START),--start $(START) --end $(END),)

ingest-mlb-statcast: ## Ingesta Statcast MLB (make ingest-mlb-statcast DATE=2024-07-01 o START=... END=...)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python -m apuestas.ingest.mlb_statcast $(if $(DATE),--date $(DATE),) $(if $(START),--start $(START) --end $(END),)

capture-closing-lines: ## Captura closing odds Pinnacle para picks vivos en ventana 30-60min
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python -m apuestas.flows.capture_closing_lines

clv-stats: ## Muestra CLV rolling 30d + 7d
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python -c "import asyncio; from apuestas.flows.capture_closing_lines import clv_rolling_stats; \
	r30 = asyncio.run(clv_rolling_stats(30)); r7 = asyncio.run(clv_rolling_stats(7)); \
	print(f'CLV 30d: {r30}'); print(f'CLV 7d: {r7}')"

seed-venue-coords: ## Seed coords 30 MLB + 32 NFL + 30 NBA venues (habilita weather)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/seed_venue_coords.py

seed-venues-soccer: ## Seed coords ~80 estadios soccer + mapping team→venue
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/seed_venues_soccer.py

ingest-weather: ## Open-Meteo weather archive (make ingest-weather SINCE=2023-04-01)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python scripts/ingest_weather_archive.py --since $(if $(SINCE),$(SINCE),2023-04-01)

enrich-features: ## Ejecuta enrichment diario (book_power + placeholders PBP/Statcast)
	DATABASE_URL="postgresql+asyncpg://apuestas:$${POSTGRES_PASSWORD}@localhost:$${POSTGRES_HOST_PORT:-5434}/apuestas" \
	POSTGRES_HOST=localhost POSTGRES_PORT=$${POSTGRES_HOST_PORT:-5434} \
	uv run python -m apuestas.flows.enrich_features

sprint11-status: ## Check estado features Sprint 10/11 activadas via env
	@echo "📊 Feature flags Sprint 10/11:"
	@echo "  APUESTAS_USE_MARKET_STACKER = $${APUESTAS_USE_MARKET_STACKER:-false}"
	@echo "  APUESTAS_USE_FOCAL_LOSS = $${APUESTAS_USE_FOCAL_LOSS:-false}"
	@echo "  APUESTAS_USE_TABPFN_STACKER = $${APUESTAS_USE_TABPFN_STACKER:-false}"
	@echo "  APUESTAS_ENABLE_XT = $${APUESTAS_ENABLE_XT:-true}"
	@echo "  APUESTAS_ENABLE_NBA_CLUTCH = $${APUESTAS_ENABLE_NBA_CLUTCH:-true}"
	@echo "  APUESTAS_ENABLE_MLB_STUFF_PLUS = $${APUESTAS_ENABLE_MLB_STUFF_PLUS:-true}"
	@echo "  APUESTAS_MLB_POISSON_ENSEMBLE = $${APUESTAS_MLB_POISSON_ENSEMBLE:-false}"
	@echo "  APUESTAS_USE_BOOK_POWER = $${APUESTAS_USE_BOOK_POWER:-true}"
	@echo "  APUESTAS_SPRINT11_SOFT_TAGS = $${APUESTAS_SPRINT11_SOFT_TAGS:-true}"
	@echo "  APUESTAS_ELO_FEATURES_DISABLED = $${APUESTAS_ELO_FEATURES_DISABLED:-false}"

seed-player-logs: ## Populatear player_game_logs (make seed-player-logs SPORT=nba SEASONS=2023-24,2024-25)
	@[ -n "$(SPORT)" ] || (echo "❌ Uso: make seed-player-logs SPORT=nba [SEASONS=CSV]"; exit 1)
	$(COMPOSE) exec api python -m apuestas.scripts.seed_player_game_logs_$(SPORT) $(if $(SEASONS),--seasons $(SEASONS),)

drift-monitor: ## Ejecuta drift monitor + auto-retrain si detecta degradación
	$(COMPOSE) exec api python -m apuestas.monitors.drift_monitor

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

## ─── Resultados post-match ──────────────────────────────────────────────────

live-scores: ## Ingesta scores de partidos finalizados últimas 48h
	$(COMPOSE) exec api python -m apuestas.flows.live_scores

# settle-bets, post-mortem y fiscal-export retirados en pivote detector puro
# (2026-04-23). El bot ya no gestiona banca, PnL, ni reportes SAT.

## ─── Capture job (closing line independiente) ───────────────────────────────

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
