#!/usr/bin/env bash
# install.sh — instala las systemd units del capture job en ~/.config/systemd/user/

set -euo pipefail

TARGET="$HOME/.config/systemd/user"
mkdir -p "$TARGET"
mkdir -p "$HOME/proyectos/apuestas/logs"

cp -v apuestas-capture.service apuestas-capture.timer "$TARGET/"

systemctl --user daemon-reload
echo "✅ Units instaladas. Activar con: make capture-on"
