# GitHub setup — repo privado + CI/CD completo

## 1. Crear repo privado

```bash
cd ~/proyectos/apuestas
git init -b develop
git add .
git commit -m "chore: initial commit (Fases 1-8 completas)"

# Crear repo privado en GitHub con gh CLI
gh auth login    # una sola vez
gh repo create apuestas --private --source=. --remote=origin --push

# Verificar privacidad
gh repo view --json visibility,isPrivate
```

## 2. Configuración inicial del repo

```bash
# Deshabilitar wiki, habilitar issues/projects, delete branches al merge
gh repo edit apuestas \
  --enable-issues \
  --enable-projects \
  --enable-wiki=false \
  --delete-branch-on-merge \
  --default-branch develop
```

## 3. Branch protection rules

### Main (producción)

```bash
gh api -X PUT "repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/branches/main/protection" \
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
      "semgrep"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1,
    "require_last_push_approval": true
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true,
  "block_creations": false
}
EOF
```

### Develop (integración)

Mismos checks pero sin `enforce_admins`:

```bash
gh api -X PUT "repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/branches/develop/protection" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["lint", "typecheck", "unit-tests (0)", "integration-tests"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

## 4. Secrets

### Action secrets (repo)

```bash
# Opcional: Codecov upload
gh secret set CODECOV_TOKEN --body "<token>"

# API keys externas (ya en .env local, NO se usan en CI por default)
# Solo añadir si tests necesitan API real, lo cual debe EVITARSE (usar fixtures)
```

### Environments

```bash
# Crear environments
gh api -X PUT "repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/environments/staging" \
  -f wait_timer=0

gh api -X PUT "repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/environments/production" \
  -f wait_timer=600 \
  -F "reviewers[0][type]=User" \
  -F "reviewers[0][id]=$(gh api /user -q .id)"
```

## 5. Verificación setup

```bash
# Repo debe ser privado
gh repo view --json visibility,isPrivate,hasIssuesEnabled
# "visibility": "PRIVATE", "isPrivate": true, "hasIssuesEnabled": true

# Branch protection activa
gh api "repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/branches/main/protection" -q .required_status_checks.contexts

# Workflows habilitados
gh workflow list
# Debe listar: CI, Security, Dependencies audit, Docker build + push

# Dependabot configurado
gh api "repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/vulnerability-alerts" -i | head -1
# HTTP/2.0 204 No Content (alerts habilitados)
```

## 6. Ajustes Settings UI (una sola vez)

En **Settings → Code security**:
- ✅ Private vulnerability reporting
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Dependabot version updates
- ✅ Code scanning (CodeQL configurado via workflow)
- ✅ Secret scanning
- ✅ Push protection for secret scanning

En **Settings → Actions → General**:
- Actions permissions: **"Allow <user>, and select non-<user>"**
- Artifact and log retention: 30 días (reduce costo)
- Fork PR workflows: disabled (repo privado)

En **Settings → Rules → Rulesets**:
- New branch ruleset: "main + develop"
- Enforcement: Active
- Target: `main`, `develop`
- Rules: Require PRs, signed commits (opcional), linear history

## 7. Primera ejecución

```bash
# Push inicial triggers CI
git push origin develop

# Verificar
gh run list --limit 5
gh run watch  # espera el último run

# Si hay fallas:
gh run view --log-failed
```

## 8. Flujo de trabajo recomendado

```bash
# Feature branch
git switch -c feature/tenis-elo
# ... hacer cambios ...
uv run pre-commit run --all-files
git commit -m "feat(tennis): Elo surface split features"
git push -u origin feature/tenis-elo

# Crear PR
gh pr create --base develop --title "feat(tennis): Elo surface split" --body-file PR.md

# Esperar CI verde + approve
gh pr checks
gh pr merge --squash --delete-branch
```

## 9. Costos GitHub repo privado

| Item | Free tier limit | Nuestro consumo estimado |
|---|---|---|
| Actions minutes | 2,000 min/mes | ~1,700 min/mes (CI + security + audit) |
| Storage (artifacts + packages) | 500 MB | ~100 MB (imágenes + SBOM) |
| Codespaces | 120 h core/mes | 0 (no usamos) |
| Dependabot | ilimitado | ✅ |
| Code scanning | ilimitado | ✅ |

Si se acerca al límite de Actions:
1. Reducir matrix `gil: [0, 1]` → solo `0` (ahorra ~50%)
2. Cache más agresivo en `uv sync` (ya configurado)
3. Self-hosted runner en la laptop (complejo, evitar)

## 10. Emergencia: revocar acceso

Si laptop comprometida:

```bash
# Rotar token GitHub CLI
gh auth refresh --hostname github.com --scopes repo,workflow

# Rotar secretos del repo
gh secret list --app actions  # ver
gh secret delete CODECOV_TOKEN
gh secret set CODECOV_TOKEN --body "<nuevo>"

# Revocar sessions activas
# Settings → Personal access tokens → revocar todos
# Luego re-login: gh auth login
```
