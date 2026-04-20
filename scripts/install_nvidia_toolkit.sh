#!/usr/bin/env bash
# install_nvidia_toolkit.sh — Instala nvidia-container-toolkit en Pop!_OS / Ubuntu.
# Requiere sudo.

set -euo pipefail

if docker info 2>/dev/null | grep -q "nvidia"; then
  echo "✓ nvidia-container-toolkit ya configurado. Skip."
  exit 0
fi

echo "▶ Añadiendo repo NVIDIA container toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "▶ Actualizando e instalando..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "▶ Configurando Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "▶ Verificando..."
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi 2>&1 | head -5; then
  echo "✅ nvidia-container-toolkit funcionando."
else
  echo "⚠ Verifica manualmente: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
fi
