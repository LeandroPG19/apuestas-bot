#!/usr/bin/env bash
# download_models.sh — Descarga Qwen 2.5 7B Q4_K_M GGUF + BGE-M3 (cachea TEI).

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"
QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
QWEN_FILE="$MODELS_DIR/qwen2.5-7b-instruct-q4_k_m.gguf"

mkdir -p "$MODELS_DIR"

if [[ -f "$QWEN_FILE" ]]; then
  actual_size=$(stat -c%s "$QWEN_FILE" 2>/dev/null || stat -f%z "$QWEN_FILE")
  if [[ $actual_size -gt 4500000000 ]]; then
    echo "✓ Qwen 2.5 7B Q4_K_M ya descargado ($(du -h "$QWEN_FILE" | cut -f1))"
  else
    echo "⚠ Archivo incompleto, re-descargando..."
    rm -f "$QWEN_FILE"
  fi
fi

if [[ ! -f "$QWEN_FILE" ]]; then
  echo "▶ Descargando Qwen 2.5 7B Q4_K_M (~4.7 GB)..."
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf \
      --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
  else
    curl -L --progress-bar -o "$QWEN_FILE" "$QWEN_URL"
  fi
fi

echo "✓ Qwen GGUF listo en $QWEN_FILE"

echo "▶ BGE-M3 se descarga automáticamente al primer arranque del contenedor 'embed'"
echo "   (cachea en volumen Docker embed_cache, ~568M modelo)"

echo ""
echo "✅ Modelos listos. Arranca stack con: make up"
