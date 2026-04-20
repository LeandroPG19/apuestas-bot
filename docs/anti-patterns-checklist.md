# Anti-patterns checklist

Revisar **ANTES** de cada deploy de modelo nuevo o cambio crítico de pipeline.
Derivado del blueprint §9. Si alguna respuesta es "sí" → fix antes de promote.

## Datos

- [ ] **¿Usas stats finales de temporada para predecir juegos tempranos?** → LEAKAGE
- [ ] **¿KFold aleatorio en vez de `TimeSeriesSplit`?** → LEAKAGE
- [ ] **¿Normalización/scaling `fit` sobre test set?** → usa `Pipeline` con `fit` solo en train
- [ ] **¿Features rolling incluyen el partido target?** → debe cerrar en t-1
- [ ] **¿Injury data con timestamp retroactivo?** → snapshots propios
- [ ] **¿Closing line como feature?** → solo ground truth para CLV, NO feature
- [ ] **¿Park factors del año actual incluidos en features año actual?** → usar 3–5 años previos
- [ ] **¿Dataset con equipos/jugadores solo actuales?** → survivorship bias

## Modelado

- [ ] **¿Métrica primaria es accuracy?** → usa log-loss / Brier / ECE
- [ ] **¿Sin calibración explícita?** → CalibratedClassifierCV isotonic/Platt
- [ ] **¿Overfitting a régimen histórico?** (COVID, cambios reglas, VAR) → retrain + drift check
- [ ] **¿p-hacking: probaste 100 estrategias y quedaste con la mejor?** → Bonferroni / pre-registrar hipótesis
- [ ] **¿Kelly con P no calibrada?** → error 1% en P puede flipear EV
- [ ] **¿Full Kelly?** → siempre ¼ Kelly + cap 5%

## Betting

- [ ] **¿Doblar stake tras pérdida (martingala)?** → ruina garantizada con edge finito
- [ ] **¿No trackear CLV?** → 3 meses ROI+ sin CLV+ es suerte, regresará
- [ ] **¿Ignorar margin bookmaker en cálculo EV?** → Shin/Power/Multiplicative obligatorio
- [ ] **¿Apostar sin threshold EV ≥ 3%?** → edge <3% cubierto por error de calibración
- [ ] **¿Bankroll combinado con dinero de vivir?** → drawdown 40% rompe psicológicamente
- [ ] **¿Automatizar ejecución violando TOS Caliente/Strendus?** → cuenta cerrada + fondos retenidos

## Sistema

- [ ] **¿Sin tests?** → min 70% cobertura `betting/`, `features/`, `ml/`
- [ ] **¿Sin backup?** → `make backup` al menos semanal
- [ ] **¿Secretos en git?** → `detect-secrets` + `gitleaks` en pre-commit
- [ ] **¿Sin rate limit coordinado?** → Valkey-backed `aiolimiter`
- [ ] **¿Sin circuit breaker en APIs externas?** → `stamina` + `pybreaker`
- [ ] **¿Ingesta sin idempotencia?** → dedupe por external_id + checksum
- [ ] **¿Postgres sin pg_stat_statements?** → queries lentas invisibles

## LLM

- [ ] **¿LLM genera probabilidades directas?** → NO. Solo NER/RAG/explicación
- [ ] **¿Sin validación schema JSON del output LLM?** → msgspec + retry con prompt correctivo
- [ ] **¿Sin grammar GBNF?** → fuerza JSON válido en llama.cpp
- [ ] **¿Prompts sin versionar?** → `prompts/*.yaml` versionados
- [ ] **¿Sin blending controlado?** → solo activar si mejora log-loss en ablation

## Significancia estadística

- [ ] **¿Declaras skill con <500 bets?** → edge 2% necesita n ≈ 9,600 (medido por ROI) o n ≈ 65 (medido por CLV)
- [ ] **¿Anuncias ">10% ROI consistente"?** → small sample / survivorship / fraude
- [ ] **Ground truth: sharp profesional ≈ 2-5% ROI long-term** (Miller & Davidow, Buchdahl)

## Semáforos automáticos (el bot te avisa)

- CLV_7d < −2% → pausar automáticamente
- drawdown_30d > 20% → reducir Kelly ¼ → ⅛
- loss_streak ≥ 6 en sport → pausar sport 72 h
- `confidence=high` aciertos caen >10pp → pausar picks high-conf
- calibration_gap > 0.05 en bucket con n≥30 → alerta crítica + flag review

Si cualquiera salta: **/pausar** en Telegram y revisar antes de continuar.
