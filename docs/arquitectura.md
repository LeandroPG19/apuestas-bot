# Arquitectura del proyecto

Referencia completa: [plan de implementación](../../../.claude/plans/analiza-a-detalle-analiza-radiant-cherny.md).

## Diagrama alto nivel

```
┌────────────── Laptop Ryzen 7 + RTX 4050 (modo on-demand) ──────────────┐
│                                                                        │
│  ┌─ Aplicación (Docker Compose) ───────────────────────────────────┐  │
│  │ api          FastAPI + granian     (bind 127.0.0.1:8001)        │  │
│  │ worker-ingest TaskIQ (odds, news, fixtures)                     │  │
│  │ worker-ml    TaskIQ (features, predict, LLM)                    │  │
│  │ worker-scrape TaskIQ (camoufox on-demand)                       │  │
│  │ prefect      Prefect 3 server + agent                           │  │
│  │ llm          llama.cpp + Qwen 2.5 7B Q4    (GPU 4.7 GB VRAM)    │  │
│  │ embed        TEI + BGE-M3 INT8              (GPU 0.6 GB VRAM)   │  │
│  │ telegram     python-telegram-bot polling                        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ Datos ─────────────────────────────────────────────────────────┐  │
│  │ postgres     TimescaleDB 2.20 + pgvector 0.8 HNSW               │  │
│  │ valkey       cache + broker TaskIQ                              │  │
│  │ minio        artifacts MLflow, parquet OLAP                     │  │
│  │ mlflow       experiment tracking + model registry               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ Observabilidad (opcional, Fase 11) ─────────────────────────────┐  │
│  │ signoz-otel + clickhouse + grafana                              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ Fuera de Docker (systemd user) ────────────────────────────────┐  │
│  │ apuestas-capture.timer  cada 5 min ~30 MB — CLV tracking        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

## Decisiones arquitectónicas clave

1. **On-demand, NO 24/7**: usuario decide cuándo levantar el stack. `make up` / `make down`.
2. **Solo LAN**: Telegram long-polling, dashboards en `127.0.0.1` + LAN.
3. **Multi-deporte día 1**: NBA + MLB + NFL + Fútbol + Boxeo.
4. **LLM local**: Qwen 2.5 7B Q4_K_M + BGE-M3 INT8 en RTX 4050. $0 costo LLM.
5. **Análisis espejo**: cada pick analiza AMBOS equipos (home + away) en 9 capas + venue/travel.
6. **Post-mortem automático**: al terminar cada partido, compara predicción vs realidad y aprende.
7. **MCP integrado**: cuba-memorys (memoria persistente) + cuba-search (research web).

## Flujo de una sesión típica

1. Usuario: `make up` → stack completo arriba en <60 s (imágenes cacheadas).
2. Usuario: `make analyze` → flow `deep_analysis.py` corre barrido 48 h.
3. Por cada evento próximo:
   - Catchup data (9 capas × 2 equipos)
   - Validación Pandera
   - Features rolling home/away/diff
   - Modelo LightGBM + calibración + MAPIE conformal
   - LLM Qwen análisis espejo JSON estructurado
   - EV + Kelly ¼ + de-vigging Shin
   - Registro en `predictions` + `decision_log`
   - cuba-memorys registra decisión
4. Picks emitidos → Telegram con reporte detallado.
5. Usuario confirma toma manual en casa (TOS semi-automatizado).
6. Capture job (si activo) captura closing line.
7. `live_scores.py` + `settle_bets.py` actualiza resultado.
8. `post_mortem.py` compara predicción vs realidad, escribe `post_mortems`, feedback a cuba-memorys.
9. Usuario: `make down` cuando termina.

## Puertos expuestos

| Puerto host | Servicio         | Notas                              |
|-------------|------------------|------------------------------------|
| 8001        | FastAPI          | `/health`, `/docs`, `/metrics`     |
| 5433        | Postgres         | 5432 ocupado por PG host           |
| 5000        | MLflow           | UI                                 |
| 4200        | Prefect          | UI + API                           |
| 9001        | MinIO console    | Solo localhost                     |
| 3000        | Grafana          | Fase 11                            |
| 3301        | SigNoz UI        | Fase 11                            |

## Carpetas principales

Ver [README.md](../README.md). Layout completo en [plan §3](../../../.claude/plans/analiza-a-detalle-analiza-radiant-cherny.md).
