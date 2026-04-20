# Onboarding — primera instalación

Tiempo estimado: **30–60 min** (limitado por descargas de modelos + imágenes Docker).

## 1. Prerequisitos mínimos

- Laptop/desktop con **GPU NVIDIA ≥6 GB VRAM** (RTX 4050 Mobile verificado)
- **≥14 GB RAM**, **≥100 GB disco libre**
- Linux (Pop!_OS 24.04 / Ubuntu 24.04 / derivados)
- Docker + Compose plugin
- NVIDIA driver ≥ 550, CUDA ≥ 12.0
- `uv` ([docs.astral.sh/uv](https://docs.astral.sh/uv/))

## 2. Clonar y preparar

```bash
cd ~/proyectos/apuestas
cp .env.example .env
```

Edita `.env` y reemplaza:
- `POSTGRES_PASSWORD` — genera con `openssl rand -base64 32`
- `VALKEY_PASSWORD` — ídem
- `MINIO_ROOT_PASSWORD` — ídem
- `API_FOOTBALL_KEY` — de [api-sports.io](https://rapidapi.com/api-sports/api/api-football/) tier Pro $19
- `THE_ODDS_API_KEY` — de [the-odds-api.com](https://the-odds-api.com/) free 500 créditos/mes
- `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` — crea bot con @BotFather
- (Opcional) `OPENWEATHERMAP_KEY` free tier 1M req/mes
- (Opcional) `REDDIT_CLIENT_ID` y `REDDIT_CLIENT_SECRET`

## 3. Instalación automatizada

```bash
make install        # verifica prerequisitos + instala nvidia-toolkit + descarga modelos
make cold-start     # build + migrate + levantar stack (~20 min primera vez)
make smoke-test     # valida todo responde
```

## 4. Verificación

- Health check: `curl http://localhost:8001/health`
- Prefect UI: http://localhost:4200
- MLflow UI: http://localhost:5000
- MinIO console: http://localhost:9001
- Métricas Prometheus: http://localhost:8001/metrics

## 5. Operación diaria

```bash
make up             # encender stack cuando quieras analizar
make analyze        # barrido completo de eventos próximos 48 h
make status         # ver estado + VRAM + RAM
make logs           # tail de logs (Ctrl-C para salir)
make down           # apagar limpio cuando termines
```

## 6. CLV capture persistente (opcional pero recomendado)

Permite tracking de closing line aunque el stack esté apagado:

```bash
cd capture/systemd && bash install.sh
make capture-on
systemctl --user status apuestas-capture.timer
```

## 7. Siguiente paso

Lee [docs/runbook.md](runbook.md) para operaciones cotidianas y
[docs/anti-patterns-checklist.md](anti-patterns-checklist.md) antes de
entrenar modelos o ejecutar dinero real.
