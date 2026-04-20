# CI/CD — local + GitHub

## Local (pre-commit hooks)

Instalación una sola vez:

```bash
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type commit-msg --hook-type pre-push
detect-secrets scan > .secrets.baseline
```

Hooks configurados en `.pre-commit-config.yaml`:

| Hook | Cuándo | Bloquea commit si |
|---|---|---|
| trailing-whitespace, EOF, yaml/json/toml | pre-commit | Sintaxis inválida |
| check-added-large-files (1 MB) | pre-commit | Archivo > 1 MB |
| detect-private-key | pre-commit | Llaves privadas en diff |
| no-commit-to-branch | pre-commit | Commit directo a main/master/production |
| ruff check + ruff-format | pre-commit | Lint errors / formato |
| mypy strict | pre-commit | Type errors en src/ |
| detect-secrets | pre-commit | Secretos detectados |
| gitleaks | pre-commit | Secrets en histórico |
| shellcheck | pre-commit | Shell warnings |
| hadolint-docker | pre-commit | Dockerfile warnings |
| alembic-check | pre-commit | Migraciones sintácticamente inválidas |
| pytest unit (rápidos) | **pre-push** | Tests unit fallan |
| conventional-pre-commit | commit-msg | Mensaje no sigue convención |

Ejecutar todos manualmente:
```bash
uv run pre-commit run --all-files
```

Correr un hook específico:
```bash
uv run pre-commit run ruff --all-files
```

## GitHub Actions (repo privado)

### Workflows

| Workflow | Trigger | Jobs |
|---|---|---|
| `.github/workflows/ci.yml` | push + PR | lint, typecheck, unit-tests (matrix GIL 0/1), integration-tests (PG+Valkey services), docker-build |
| `.github/workflows/security.yml` | push + PR + weekly | trivy-fs, trivy-image, gitleaks, pip-audit, semgrep, codeql |
| `.github/workflows/deps-audit.yml` | weekly (lunes 05:00 UTC) | uv lock --upgrade + PR automático + issue si vulns |
| `.github/workflows/deploy.yml` | push main / tag v*.*.* | Build + push GHCR + SBOM |

### Secretos GitHub

Configurar en **Settings → Secrets and variables → Actions**:

| Secret | Uso |
|---|---|
| `GITHUB_TOKEN` | Auto (otorgado por GitHub) |
| `CODECOV_TOKEN` | Codecov upload (opcional) |
| `DOCKER_REGISTRY_TOKEN` | Solo si push a registro custom |

### Configuración de repo privado

Tras `gh repo create apuestas --private`:

```bash
gh repo edit apuestas --enable-issues --enable-projects --enable-wiki=false
gh repo edit apuestas --delete-branch-on-merge
gh repo edit apuestas --default-branch develop
```

### Branch protection (main + develop)

```bash
gh api -X PUT repos/:owner/apuestas/branches/main/protection \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "lint",
      "typecheck",
      "unit-tests (0)",
      "unit-tests (1)",
      "integration-tests",
      "docker-build",
      "trivy-scan",
      "gitleaks",
      "pip-audit",
      "semgrep",
      "codeql"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
EOF
```

Mismo para `develop` con protecciones ligeramente menos estrictas.

### Dependabot

`.github/dependabot.yml` maneja:
- GitHub Actions (weekly)
- Docker base image (weekly)

Python deps NO via dependabot — usa `deps-audit.yml` que corre `uv lock --upgrade` (dependabot no soporta uv lock nativo aún).

### Rulesets recomendados

En **Settings → Rulesets** añadir:
1. **Push a main** bloqueado sin PR
2. **Tag protection**: sólo `v*.*.*` semver
3. **Status checks required** (listados arriba)
4. **Require signed commits** (opcional, recomendado)

### Environments

**Settings → Environments** crear:
- `production` (branch main, required reviewers: tú, wait timer 10 min)
- `staging` (branch develop, sin wait)

Los secretos sensibles (API keys externas) se cargan **solo en environment production** durante el deploy manual, nunca en CI.

### Actions usage budget (repo privado)

Plan free GitHub: 2,000 minutos/mes repos privados.
- CI típico: ~8 min/run → ~7 runs/día × 30d = 1,680 min/mes ✓
- Security weekly: ~15 min
- Deps audit weekly: ~10 min

Si se acerca al límite, reducir matrix o mover jobs pesados a self-hosted runner.

## Comandos Make asociados

```bash
make lint          # ruff check src tests
make format        # ruff format + --fix
make typecheck     # mypy strict src
make test          # pytest suite completa
make test-unit     # solo unit
make test-integration  # requiere services
make audit-deps    # scripts/audit_deps.sh
make audit-python  # pip-audit
make audit-images  # trivy sobre imágenes
make sbom          # syft SBOM
```
