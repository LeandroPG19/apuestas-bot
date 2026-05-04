# Dependabot status — 2026-05-04

Estado de las 29 alertas de Dependabot detectadas en el initial release.

## ✅ Resueltas (24 de 29)

### Direct deps actualizadas en `pyproject.toml`

| Paquete | Antes | Después | CVE / Severity |
|---|---|---|---|
| `lightgbm` | 4.5.0 | **4.6.0** | CVE-2024-43598 RCE — HIGH |
| `orjson` | 3.11.2 | **3.11.6** | CVE-2025-67221 recursion — HIGH (×2 alerts) |
| `jupyterlab` | 4.5.1 | **4.5.7** | CVE-2026-40171 XSS token theft — HIGH |
| `python-dotenv` | 1.0.1 | **1.2.2** | symlink overwrite — MEDIUM (×2 alerts) |

### Transitive deps forzadas vía `[tool.uv]` `override-dependencies`

| Paquete | Antes | Después | CVEs |
|---|---|---|---|
| `aiohttp` | 3.13.3 | **3.13.5** | CVE-2026-34518/19/20 + 6 más — 2 MEDIUM, 6 LOW |
| `gitpython` | 3.1.46 | **3.1.49** | cmd injection vía git options — HIGH (×2) |
| `urllib3` | 2.5.0 | **2.6.3** | CVE-2025-66418/71 + CVE-2026-21441 — 3 HIGH |
| `requests` | 2.32.5 | **2.33.1** | tmp file reuse en `extract_zipped_paths()` — MEDIUM |

### GitHub Actions

| Action | Antes | Después | CVE |
|---|---|---|---|
| `aquasecurity/trivy-action` | 0.29.0 | **0.35.0** | CVE-2026-33634 supply chain — **CRITICAL** |

## ⏸ Diferidas (5 de 29) — bloqueadas por dependency chain

Documentamos por qué cada una se difiere y la mitigación temporal.

### `pydantic-ai==0.4.2` — SSRF (HIGH, CVE-2026-25580, ×2 alerts)

- **Fix requiere**: `pydantic-ai >= 1.56.0`
- **Bloqueado por**: 1.56.0 arrastra `mcp >= 1.25.0` y `opentelemetry-sdk 1.39.x-1.40.x`,
  lo cual a su vez requiere instrumentation-fastapi `0.61b0`. El proyecto pinea
  OTel 1.34.0 + instrumentation 0.55b0 por estabilidad probada.
- **Mitigación**: el `pydantic-ai.URL.download` no se usa directamente desde paths
  con input usuario. El SSRF requiere que pase user-controlled URL.
- **Plan**: migrar OTel/mcp chain en sprint 15, luego upgrade pydantic-ai.

### `mcp==1.15.0` — DNS rebinding (HIGH, CVE-2025-66416)

- **Fix requiere**: `mcp >= 1.23.0`
- **Bloqueado por**: arrastra `pydantic-ai >= 1.56.0` (ver arriba)
- **Mitigación**: el bot usa MCP sólo en stdio mode (subprocess local), no HTTP.
  El DNS rebinding sólo afecta el HTTP transport mode (que no usamos).
- **Plan**: ligado al upgrade de pydantic-ai.

### `pytest==8.4.2` — tmpdir handling (MEDIUM)

- **Fix requiere**: `pytest >= 9.0.3`
- **Bloqueado por**: `seleniumbase` (transitive de `camoufox` para scraping
  Caliente.mx) pinea `pytest==8.4.1-8.4.2` estrictamente hasta versión 4.46.x.
- **Mitigación**: vulnerabilidad sólo aplica en entornos de test compartidos
  (CI multi-tenant), no producción.
- **Plan**: monitorear seleniumbase 4.47+ que soporta pytest 9. Mientras tanto,
  los tests corren en GitHub Actions con runners ephemeral (no shared tmpdir).

### `transformers==4.57.6` — Trainer RCE (MEDIUM)

- **Fix requiere**: `transformers >= 5.0.0rc3` (release candidate)
- **Bloqueado por**: 5.0.0 es pre-release; introduce breaking changes en API
  que afectarían modelos LLM custom que cargamos vía TEI.
- **Mitigación**: el `Trainer` class no se usa en este proyecto. Sólo usamos
  transformers para `AutoTokenizer` (no afectado por el CVE).
- **Plan**: upgrade cuando 5.0.0 final salga (estimado Q3 2026).

### `pip` (MEDIUM, sin patched version)

- **Fix requiere**: ninguna versión patched disponible al cierre
- **Mitigación**: `pip` se usa sólo en CI ephemeral; no se ejecuta como user
  process del bot. Riesgo bajo por contexto de uso.
- **Plan**: monitor pypa/pip releases.

## Re-scan periódico

```bash
# Ver alerts abiertas
gh api /repos/LeandroPG19/apuestas-bot/dependabot/alerts \
  --jq '.[] | select(.state=="open") | "\(.security_advisory.severity): \(.dependency.package.name)"'

# Re-resolver lock con upgrades
uv lock --upgrade

# Audit local
uv run pip-audit -r <(uv export --no-dev)
```

Próximo review: **2026-06-04** (1 mes después del release).
