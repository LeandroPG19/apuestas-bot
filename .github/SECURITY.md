# Política de seguridad

## Reportar vulnerabilidades

Repositorio **privado**. Si detectas una vulnerabilidad:

1. **NO** abrir un issue público — usa GitHub Security Advisories
   (Security → Report a vulnerability).
2. Proporciona: descripción, pasos para reproducir, impacto potencial,
   y cualquier CVE conocido.
3. Se responderá en <48 h.

## Alcance

Bot de apuestas personal, 100% local. Las preocupaciones principales son:

- **Secretos** (.env, API keys) — never commit.
- **SQL injection** — parametrizar siempre, nunca string interpolation.
- **Dependency CVEs** — pip-audit semanal vía GitHub Actions.
- **Docker images** — Trivy scan (CRITICAL+HIGH) bloqueante en CI.
- **Supply chain** — todas las deps con `==` pin exacto + lock file.
- **Scraping ToS** — respetar robots.txt, rate limits.

## Flujo de deps

1. Dependabot abre PRs semanales para Actions + Docker.
2. `deps-audit.yml` abre PR semanal para Python (uv lock --upgrade).
3. pip-audit + Trivy corren en cada PR.

## Secretos

Nunca committear:
- `.env`
- `uv.lock` puede contener URLs privadas — revisar antes.
- Credenciales de API externas en ningún archivo.
- Modelo GGUF / binarios ML (solo en `./models/`, gitignored).

## Pre-commit hooks obligatorios

- `detect-secrets` (Yelp)
- `gitleaks`
- `check-added-large-files` (maxkb=1024)
- `detect-private-key`

## Tests de regresión

Cada release verifica:
- OWASP Top 10 via Semgrep SAST.
- Trivy FS + image scan.
- pip-audit.
- CodeQL analysis (security-and-quality query pack).
