# Disaster Recovery Runbook

## Escenarios y procedimientos

### A. Corrupción Postgres
**Síntomas**: queries fallan con errores extraños, WAL inconsistente, extensions crashean.

```bash
# 1. Bajar todo menos PG+Valkey+MinIO
make down
# 2. Restore desde último backup bueno
docker compose up -d postgres
make restore FILE=backups/pg_YYYY-MM-DD_HHMM.dump
# 3. Verificar
docker compose exec postgres psql -U apuestas -c "\dt"
docker compose exec postgres psql -U apuestas -c "SELECT max(start_time) FROM matches"
# 4. Reanudar stack — catchup re-ingesta desde ingest_checkpoints
make up
```

**Tiempo objetivo**: < 30 min.

### B. Disco lleno (>95% /)
```bash
# Cleanup agresivo
docker system prune -af --volumes
sudo journalctl --vacuum-size=500M
find /var/lib/docker/containers -name "*.log" -size +100M -exec truncate -s 0 {} \;
# Comprimir chunks antiguos TimescaleDB
docker compose exec postgres psql -U apuestas -c \
  "SELECT compress_chunk(c) FROM show_chunks('odds_history', older_than => INTERVAL '1 day') c"
# Retention manual si TimescaleDB retention no alcanzó
docker compose exec postgres psql -U apuestas -c \
  "SELECT drop_chunks('odds_history', older_than => INTERVAL '1 year')"
```

### C. Modelo envenenado (produce picks absurdos)
```bash
# Promote anterior versión
make rollback-model MODEL=nba_moneyline VERSION=3
# Verifica en MLflow
docker compose exec api python -m apuestas.ml.registry list-versions --model nba_moneyline
```

### D. API externa rota (API-Football 500 sostenido)
1. Circuit breaker abre solo → fallback a scraping ESPN
2. Alerta Telegram automática
3. Verifica status en [status.api-sports.io](https://status.api-sports.io/)
4. Si rota >6 h: desactivar ingesta ese source en `bot_state`

### E. Laptop muerta (reemplazo)
Requisitos: último backup offsite + acceso a repo.

```bash
# En máquina nueva
git clone <repo>
cd apuestas
cp /path/to/backed/.env.local .env
make install         # nvidia toolkit, modelos
make cold-start      # build, migrate (DB vacía)
# Restore desde B2
rclone sync b2:bucket/pg_backups/ backups/
make restore FILE=backups/pg_2026-04-19_0300.dump
# Re-activa capture
cd capture/systemd && bash install.sh
make capture-on
```

**Tiempo objetivo**: < 2 h (dominante: descarga modelos + imágenes).

### F. Secretos comprometidos
1. Rotar **inmediatamente**: Postgres pwd, Valkey pwd, API keys, Telegram token
2. `git log --all -S OLD_SECRET` → no debe aparecer; si sí, rewrite history + force push + rotate remoto
3. Invalidar tokens en @BotFather, api-sports, the-odds-api
4. `detect-secrets scan --baseline .secrets.baseline`

## Drill trimestral

`make dr-drill` ejecuta un restore completo sobre DB temporal y valida integridad.
Agendar en calendario cada 3 meses.
