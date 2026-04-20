#!/usr/bin/env bash
# install_mcp.sh — Detecta servidores MCP cuba-memorys y cuba-search del usuario.

set -euo pipefail

echo "▶ Buscando servidores MCP locales..."

for cmd in cuba-memorys-server cuba-search-server; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "✓ $cmd encontrado en $(which $cmd)"
  else
    echo "⚠ $cmd no en PATH — edita .env manualmente con la ruta correcta:"
    echo "    CUBA_MEMORYS_STDIO_CMD=..."
    echo "    CUBA_SEARCH_STDIO_CMD=..."
  fi
done

MCP_CONFIG_CLAUDE="$HOME/.config/Claude/claude_desktop_config.json"
MCP_CONFIG_CODE="$HOME/.claude/settings.json"

for cfg in "$MCP_CONFIG_CLAUDE" "$MCP_CONFIG_CODE"; do
  if [[ -f "$cfg" ]]; then
    echo ""
    echo "▶ Detectado config MCP en: $cfg"
    if grep -q "cuba-memorys\|cuba-search" "$cfg" 2>/dev/null; then
      echo "  Tienes cuba-memorys o cuba-search configurado allí."
      echo "  Revisa 'command' + 'args' en ese JSON para .env APUESTAS."
    fi
  fi
done

echo ""
echo "✅ Verificación MCP completa. Ajusta .env si es necesario."
