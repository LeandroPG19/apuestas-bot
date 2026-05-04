# SOTA Roadmap abril 2026 — Apuestas bot

Fecha: 2026-04-24
Base de investigación: 9 papers recientes + análisis codebase actual.

## Posicionamiento actual del bot vs SOTA

**Tu bot YA implementa 16 técnicas SOTA**:

| # | Técnica | Implementado en | Ref |
|---|---|---|---|
| 1 | Stacking ensemble (LGBM+XGB+CatBoost+LogReg meta) | `train_base.py` | Nature Scientific Reports 2025 |
| 2 | Venn-Abers calibration | `ml/calibrate.py` | Vovk 2025 ICML |
| 3 | MAPIE conformal prediction | `ml/train_base.py` | Angelopoulos 2023 |
| 4 | Focal loss tabular | `ml/stacker.py` | Mukhoti 2020 NeurIPS |
| 5 | Isotonic post-hoc calibration | `ml/calibrate.py` | Niculescu-Mizil 2005 |
| 6 | TimeSeriesSplit gap=7d purged | `ml/train_base.py` | Bergmeir 2012 |
| 7 | Brier/ECE/BSS as primary (not accuracy) | `ml/kpi_gate.py` | Walsh & Joshi 2024 |
| 8 | TabPFN v1 stacker | `ml/tabpfn_stacker.py` | Hollmann 2025 Nature |
| 9 | FT-Transformer | `ml/ft_transformer.py` | Gorishniy 2023 |
| 10 | Kalman filter live betting | `betting/live_kalman.py` | Ötting 2024 |
| 11 | ADWIN concept drift monitor | `monitors/concept_drift.py` | Bifet 2007 |
| 12 | Closing line predictor | `betting/closing_line_predictor.py` | Buchdahl 2023 |
| 13 | Book power ratings | `betting/book_power_ratings.py` | Propio + Levitt 2004 |
| 14 | Dixon-Coles ξ time decay | `ml/dixon_coles.py` | DC 1997 |
| 15 | StatsBomb xT + VAEP proxy | `features/soccer_xt.py` | Karun Singh 2018 |
| 16 | Market consensus (Pinnacle + Polymarket + Kalshi) | `betting/market_consensus.py` | Propio + Buchdahl |

## Oportunidades SOTA abril 2026 (papers nuevos investigados)

### Tier 1 — Crítico, alto ROI, bajo costo (**procedo ahora**)

#### 🔥 **A. Backfill features rolling 2015-2026** (~2h background)
**Problema**: `team_stats_rolling_{home,away}` populado solo para fechas recientes. Bloquea backtest OOS sobre 100k matches históricos.
**Solución**: script batch que calcula rolling 5/10/20 por `(team_id, sport)` sobre toda la tabla `matches` → populates `team_stats_rolling_{home,away}`.
**Beneficio**: desbloquea Tier B y C.

#### 🔥 **B. Error analysis categórico** ✅ HECHO HOY
Script `scripts/error_analysis.py`. Reveló:
- MLB spreads midline 1.80-2.20 → 0% win rate (n=3) → threshold ↑
- Soccer midline home → 25% win rate (n=4) → bajar confianza mid-odds

#### 🔥 **C. Grid-search thresholds por (sport, market, odds_bucket)**
**Problema**: threshold global uniforme. Pero MLB spreads perdedor, MLB favs underdogs ganador → thresholds deberían ser por odds-bucket.
**Solución**: barrido (ev_min, draw_max, conformal_width) × sport × market. Output: `config/thresholds_optimal.yaml`.
**Beneficio**: ROI +3-8% sin cambiar modelo.

### Tier 2 — Alta prioridad técnica (**plan documentado**)

#### 🧠 **D. TabPFN v1 → v2.5 upgrade** (arXiv 2511.08667, nov 2025)
TabPFN v2.5 tiene 20× capacidad (50k filas, 2000 features). v1 estaba limitado a datasets pequeños.
**Migración**:
```bash
uv add tabpfn==2.5.0
# Update ml/tabpfn_stacker.py: nuevo API priorlabs
```
**Beneficio**: Brier −0.005 a −0.015 en sports con dataset pequeño (tennis, mlb spreads).

#### 🎲 **E. MC Dropout sequential calibration NBA** (MDPI Information ene 2026)
RNN con dropout activo en inference → 20-50 samples → intervalos calibrados por pick.
**Implementación**: `src/apuestas/ml/mc_dropout_nba.py` wrapping the NBA ensemble.
**Beneficio**: p_lower / p_upper realistas → conformal filter más preciso.

#### ⚽ **F. Bayesian hierarchical xG soccer** (Scholtes & Karakuş, PMC 2025)
Reemplaza DC puro con Bayes jerárquico sobre **xG** (not goals). Captura "suerte" fuera de control del equipo.
**Implementación**: nuevo `src/apuestas/ml/bayesian_xg_soccer.py` con PyMC (ya tienes PyMC en deps).
**Beneficio soccer**: Brier ~0.19 (vs 0.22 LGBM actual en top-5 EU). ROI +2-4%.

### Tier 3 — Proyectos grandes (**backlog definido**)

#### 📊 **G. Temporal Fusion Transformer MLB pitcher** (ScienceDirect 2025)
TFT superó XGBoost/LGBM en pitcher ERA prediction. Maneja series temporales + covariates estáticas + encoded lineups.
**Costo**: 3-5 días. Pytorch Forecasting library.
**Beneficio**: modelo MLB pitcher-specific → resuelve sobreemisión spreads +1.5.

#### 🧠 **H. 1D-CNN + Transformer hybrid soccer** (PMC 2025)
CNN local spatial + Transformer temporal. 75-80% accuracy con play-by-play sequence.
**Prerequisito**: StatsBomb event data ya ingesteada (tienes 3.1M eventos).
**Costo**: 5-7 días. Modelo nuevo end-to-end.
**Beneficio**: alternativa a DC/LGBM; mejor captura de momentum in-game.

#### 🎯 **I. Quantile regression player props**
Output distributional para props (puntos, rebotes, Ks, goleadores). Mejor que point-estimate para EV edges.
**Costo**: 2-3 días por sport (NBA, MLB).
**Beneficio**: abre mercado props que hoy está débil en bot.

## Lo que NO voy a hacer (y por qué)

### ❌ Modelos "LLM para predicción de resultados"
Papers marketing claim 9.87% ROI usando LLMs. Sin source-code auditable, probable cherry-picking.

### ❌ Reinforcement learning para stake sizing
No aplica (bot pivoteó a detector puro, sin bankroll). Si vuelve bankroll, Kelly es óptimo Bayes (Ziemba 2017).

### ❌ Monte Carlo tree search live betting
Overkill para mercado retail. Latencia >500ms por pick.

## Progresión esperada del bot — 60 días

| Semana | Acción | Hit rate | ROI |
|---|---|---|---|
| 0 (hoy) | baseline post-fixes | 33% | −39% |
| 1-2 | Tier 1 A-C completo | 38-42% | −15% a −25% |
| 3-4 | Tier 2 D + E | 42-45% | −8% a −15% |
| 5-6 | Tier 2 F (Bayes xG) | 44-48% | −3% a −10% |
| 7-8 | Tier 3 G (TFT MLB) | 45-50% | +0% a −5% |

**Cap teórico**: ROI +3-5% (tope por vig mercado). Llegar a positive ROI sostenido requiere todas las 3 tiers + disciplina de bet sizing manual usuario.

## Referencias

- Walsh & Joshi 2024 — *ML Appl vol 16*
- Hollmann 2025 — *Nature* (TabPFN)
- arXiv 2511.08667 — *TabPFN-2.5* (nov 2025)
- Scholtes & Karakuş 2025 — *PMC interpretable xG*
- MDPI Information 17(1)56 — *MC Dropout NBA* (ene 2026)
- ScienceDirect 2025 — *TFT MLB Pitcher ERA*
- Nature Sci Reports 2025 — *Stacked ensemble NBA*
- PMC 2025 — *Deep learning sport event outcomes*
- arXiv 2410.21484 — *Systematic review ML sports betting*
