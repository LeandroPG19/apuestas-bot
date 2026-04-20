#!/usr/bin/env bash
# check_prereqs.sh — Verifica prerequisitos para correr el stack localmente.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; exit 1; }

echo "── Verificación de prerequisitos ──"

# Docker
command -v docker >/dev/null 2>&1 || fail "Docker no instalado"
docker_version=$(docker --version | awk '{print $3}' | tr -d ',')
ok "Docker instalado ($docker_version)"

docker compose version >/dev/null 2>&1 || fail "docker compose plugin no disponible"
ok "Docker Compose plugin disponible"

# NVIDIA driver + CUDA
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi no disponible — instala driver NVIDIA"
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
ok "NVIDIA driver $driver_version · CUDA $cuda_version"

# GPU capacity
vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ $vram_total -lt 5800 ]]; then
  warn "VRAM total $vram_total MiB — stack requiere 6 GB para Qwen + BGE-M3"
else
  ok "VRAM total $vram_total MiB (≥6 GB recomendado)"
fi

# RAM
ram_gb=$(free -g | awk '/^Mem:/{print $2}')
if [[ $ram_gb -lt 14 ]]; then
  warn "RAM total ${ram_gb} GB — recomendado ≥14 GB para el stack completo"
else
  ok "RAM total ${ram_gb} GB"
fi

# Disco libre
disk_free=$(df -BG --output=avail / | tail -1 | tr -d 'G ')
if [[ $disk_free -lt 100 ]]; then
  warn "Espacio libre en / ${disk_free} GB — recomendado ≥100 GB"
else
  ok "Espacio libre ${disk_free} GB"
fi

# nvidia-container-toolkit
if docker info 2>/dev/null | grep -q "nvidia"; then
  ok "nvidia-container-toolkit configurado en Docker"
else
  warn "nvidia-container-toolkit NO configurado — necesario para GPU passthrough"
  warn "Ejecuta: make install-nvidia-toolkit"
fi

# uv
command -v uv >/dev/null 2>&1 || fail "uv no instalado — https://docs.astral.sh/uv/"
ok "uv $(uv --version | awk '{print $2}')"

# Python 3.13
if uv python list --only-installed 2>/dev/null | grep -q "3.13"; then
  ok "Python 3.13 disponible para uv"
else
  warn "Python 3.13 no detectado — uv lo descargará automáticamente en cold-start"
fi

# Puertos libres (solo advertencia)
for p in 3000 3301 4200 5000 5433 6379 8001 9001; do
  if ss -tlnp 2>/dev/null | grep -q ":${p} "; then
    warn "Puerto ${p} ya ocupado en host — puede causar conflictos"
  fi
done

echo ""
ok "Prerequisitos verificados. Listo para make cold-start."
