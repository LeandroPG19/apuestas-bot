# Pull Request

## Resumen

<!-- 1-2 líneas: qué cambia y por qué -->

## Tipo de cambio

- [ ] feat: nueva funcionalidad
- [ ] fix: corrección de bug
- [ ] refactor: sin cambio funcional
- [ ] test: añadir/mejorar tests
- [ ] docs: solo documentación
- [ ] ci: CI/CD o workflows
- [ ] chore: maintenance (deps, lint config)
- [ ] perf: mejora rendimiento

## Contexto

<!-- Ticket/issue relacionado, motivación, background -->

## Cambios principales

<!-- Bulleted list de archivos/módulos/decisiones relevantes -->

-
-

## Testing

- [ ] Tests unitarios añadidos/actualizados
- [ ] Tests de integración si aplica
- [ ] Tests corridos localmente: `make test`
- [ ] Coverage no cae por debajo de 70%
- [ ] Smoke test local si afecta infra: `make smoke-test`

## Anti-patterns check (modelos ML o betting)

Si este PR toca `src/apuestas/ml/`, `src/apuestas/betting/`, `src/apuestas/risk/` o `features/`, confirmar:

- [ ] Sin data leakage temporal (rolling cierra t-1, TimeSeriesSplit con gap)
- [ ] Métricas primarias: log-loss / Brier / ECE (no accuracy)
- [ ] Calibración obligatoria: CalibratedClassifierCV + conformal MAPIE
- [ ] Kelly fraction ≤ 0.25, cap 5%
- [ ] Threshold EV ≥ 3%
- [ ] Validé el anti-patterns checklist: `docs/anti-patterns-checklist.md`

## Seguridad

- [ ] No secretos en el diff (verificado con `detect-secrets` y `gitleaks`)
- [ ] Sin interpolación de input en SQL (parametrizado siempre)
- [ ] Input validado en boundaries (Pandera/Pydantic)
- [ ] Nuevas deps pasaron `pip-audit`

## Checklist

- [ ] Pre-commit hooks pasaron (ruff, mypy, detect-secrets)
- [ ] `make lint` y `make typecheck` ok
- [ ] Documentación actualizada si API cambió
- [ ] CHANGELOG.md actualizado (si aplica)
- [ ] Breaking changes documentados

## Screenshots / outputs (si aplica)

<!-- Para features visibles, pegar screenshots o output de Telegram/dashboards -->

## Deploy notes

<!-- Cualquier paso especial post-merge: migrations, env vars nuevas, etc. -->
