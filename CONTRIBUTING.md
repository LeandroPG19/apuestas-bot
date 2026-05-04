# Contributing to apuestas-bot

¡Gracias por tu interés en contribuir! Este documento describe el proceso para
proponer cambios al proyecto.

## Idioma

- **Comunicación** (issues, PRs, discusiones): español o inglés.
- **Código, commits, docstrings**: inglés.
- **Documentación de usuario** (README, runbooks): español.

## Antes de empezar

1. Revisa los [issues abiertos](https://github.com/LeandroPG19/apuestas-bot/issues) y
   [discussions](https://github.com/LeandroPG19/apuestas-bot/discussions) para evitar
   duplicar trabajo.
2. Si tu cambio es grande (>200 líneas, refactor, nueva feature), abre primero un
   issue describiendo la propuesta.

## Setup de desarrollo

```bash
git clone https://github.com/LeandroPG19/apuestas-bot.git
cd apuestas-bot
cp .env.example .env                    # llena las variables mínimas
uv sync --all-groups                    # instala deps + dev deps
uv run pre-commit install               # hooks: ruff, mypy, detect-secrets, gitleaks
make cold-start                         # build + migrate + seed
uv run pytest -x                        # tests pasan en local
```

## Workflow de PR

1. **Fork** el repo y crea una rama desde `main`:
   ```bash
   git checkout -b feature/mi-feature   # o bugfix/, refactor/, docs/
   ```

2. **Implementa** el cambio. Convenciones del proyecto:
   - `ruff` line-length 99
   - `mypy` strict mode
   - Type hints obligatorios en funciones públicas
   - Tests para todo código nuevo (al menos happy path + 1 edge case)
   - **NO mockear la base de datos** en tests (usar la integration DB)

3. **Verifica** antes de commitear:
   ```bash
   uv run pytest -x                     # tests
   uv run ruff check                    # lint
   uv run ruff format                   # format
   uv run mypy src/                     # types
   uv run pre-commit run --all-files    # secrets + format + lint
   ```

4. **Conventional commits** (en inglés):
   ```
   feat: add Dixon-Coles model for soccer
   fix: correct EV threshold for NBA playoffs
   refactor: extract devig logic to dedicated module
   test: add edge cases for conformal prediction width
   docs: update README with new API setup
   chore: bump LightGBM to 4.6.0
   ci: add pytest matrix for Python 3.14
   ```

5. **Push y abre PR**:
   - Usa el [PR template](.github/PULL_REQUEST_TEMPLATE.md)
   - Vincula el issue: `Closes #123`
   - Espera CI verde + 1 review

## Reglas de código

### NO hacer

- ❌ Hardcodear secrets, tokens, paths absolutos
- ❌ Usar `accuracy` como métrica primaria — siempre log-loss/Brier/ECE
- ❌ Saltarse temporal-leakage en features (todo rolling cierra en `t-1`)
- ❌ Agregar features sin gap = 7d en `TimeSeriesSplit`
- ❌ Commitear código sin tests

### SÍ hacer

- ✅ Validar en system boundaries (Pydantic en API, Zod en TS)
- ✅ Pinear deps en `pyproject.toml` con `==`
- ✅ Documentar el "porqué" (no el "qué") cuando no sea obvio
- ✅ Mantener funciones < 50 líneas
- ✅ Extraer cuando una lógica se repite 3+ veces

## Áreas donde más se necesita ayuda

- **Modelos por deporte**: Tennis, NHL, Boxing, MMA (ver [BACKLOG.md](BACKLOG.md))
- **Integraciones nuevas**: bookmakers regionales, scrapers Caliente/Codere
- **Performance**: SQL queries, batching de telegram, cache patterns
- **Documentación**: traducciones, ejemplos, tutoriales
- **Tests**: coverage actual está en ~60%, target 80%

## Reporte de bugs

Usa el [issue template](https://github.com/LeandroPG19/apuestas-bot/issues/new?template=bug_report.yml).
Incluye:

- Versión del bot (`git rev-parse HEAD`)
- Versión Python (`python --version`)
- Logs relevantes (`logs/telegram.log`, `logs/analyze.log`)
- Reproducción mínima

## Reporte de vulnerabilidades

**NO** abras un issue público. Sigue el proceso en [.github/SECURITY.md](.github/SECURITY.md).

## Licencia

Al contribuir aceptas que tu código se licencie bajo MIT (igual que el resto del proyecto).
