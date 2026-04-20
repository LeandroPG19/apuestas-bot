"""Setup interactivo Telegram: token → chat_id automático → write .env → test.

Flujo:
1. Te pide el token (tú lo copias de @BotFather).
2. Valida el token con getMe de la API.
3. Te pide enviar cualquier mensaje al bot desde tu cuenta personal.
4. Long-polling hasta capturar tu chat_id automáticamente.
5. Escribe TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID al .env del proyecto.
6. Envía mensaje de prueba y confirma que llegue.

Uso:
    apuestas telegram-setup
    # o directamente
    python scripts/setup_telegram.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def _update_env(updates: dict[str, str]) -> None:
    """Idempotente: actualiza líneas existentes, crea las que falten."""
    if not ENV_PATH.exists():
        ENV_PATH.write_text("# Apuestas Bot .env\n", encoding="utf-8")
    text = ENV_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key, _, _val = stripped.partition("=")
        key = key.strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


async def _validate_token(token: str) -> dict[str, str]:
    """Retorna dict con bot info si el token es válido, {} si no."""
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"https://api.telegram.org/bot{token}/getMe")
        except httpx.HTTPError as exc:
            print(f"❌ Error de red: {exc}")
            return {}
    if r.status_code != 200:
        print(f"❌ API Telegram respondió {r.status_code}")
        return {}
    data = r.json()
    if not data.get("ok"):
        print(f"❌ Token inválido: {data.get('description', '?')}")
        return {}
    info = data["result"]
    return {
        "username": info.get("username", ""),
        "first_name": info.get("first_name", ""),
        "id": str(info.get("id", "")),
        "can_read_all": str(info.get("can_read_all_group_messages", False)),
    }


async def _discover_chat_id(token: str, *, timeout_s: int = 120) -> str | None:
    """Long-polling hasta capturar el primer chat_id que escriba al bot."""
    print(
        "\n⏳ Esperando tu mensaje...\n"
        "   Abre Telegram → busca tu bot → envíale cualquier mensaje (ej: /start)"
    )
    offset = 0
    deadline = asyncio.get_event_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=35) as c:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await c.get(
                    f"https://api.telegram.org/bot{token}/getUpdates",
                    params={"offset": offset, "timeout": 25},
                )
            except httpx.HTTPError as exc:
                print(f"⚠  error polling: {exc}")
                await asyncio.sleep(2)
                continue
            data = r.json()
            updates = data.get("result", [])
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message") or u.get("edited_message") or {}
                chat = msg.get("chat", {})
                chat_id = chat.get("id")
                if chat_id is not None:
                    chat_name = (
                        chat.get("username") or chat.get("first_name") or chat.get("title") or "?"
                    )
                    print(f"\n✅ Captured chat_id={chat_id} · {chat_name}")
                    return str(chat_id)
    return None


async def _send_test(token: str, chat_id: str) -> bool:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": (
                    "✅ *Apuestas Bot configurado*\n\n"
                    "Token + chat_id guardados en `.env`.\n"
                    "El bot te enviará aquí los picks con EV ≥ 3% cuando "
                    "corras `apuestas analyze`.\n\n"
                    "Comandos disponibles: /analyze /today /bankroll /clv /pausar /resumir"
                ),
                "parse_mode": "Markdown",
            },
        )
    return r.status_code == 200 and r.json().get("ok", False)


async def main() -> int:
    print("═" * 60)
    print("  🤖 Setup Telegram bot — Apuestas")
    print("═" * 60)
    print()
    print("Prerequisitos (los haces tú en Telegram, <2 min):")
    print("  1. Abre Telegram → busca @BotFather (el con checkmark azul)")
    print("  2. Envíale:   /newbot")
    print("  3. Nombre: ej. 'Mi Apuestas Bot'")
    print("  4. Username: terminar en 'bot' (ej. leandro_apuestas_bot)")
    print("  5. BotFather te da un token tipo:")
    print("        123456789:ABCdefGHIjklMNO...")
    print()

    token = _ask("👉 Pega el token aquí: ").strip()
    if not re.match(r"^\d+:[A-Za-z0-9_-]{30,}$", token):
        print("❌ Token con formato inválido. Debe ser <números>:<string>")
        return 1

    print("\n⏳ Validando token con la API de Telegram...")
    info = await _validate_token(token)
    if not info:
        return 1
    print(f"✅ Bot: @{info['username']} (id={info['id']})")

    chat_id = await _discover_chat_id(token, timeout_s=180)
    if not chat_id:
        print(
            "\n❌ Timeout de 3 min sin recibir mensaje.\n"
            "   Abre Telegram, busca a @"
            f"{info['username']} y envíale /start. Luego re-corre este script."
        )
        return 1

    print(f"\n💾 Escribiendo a .env: {ENV_PATH}")
    _update_env(
        {
            "TELEGRAM_BOT_TOKEN": token,
            "TELEGRAM_CHAT_ID": chat_id,
        }
    )
    print("✅ Variables guardadas: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")

    print("\n📨 Enviando mensaje de confirmación...")
    ok = await _send_test(token, chat_id)
    if ok:
        print("✅ Mensaje de prueba enviado. Revisa tu Telegram.")
    else:
        print("⚠  No pude enviar el mensaje de test (el token/chat se guardó igual)")
    print()
    print("🎉 Telegram listo. Para arrancar el bot long-polling:")
    print("   apuestas telegram start      (lo deja corriendo en tu terminal)")
    print("   # o como servicio systemd:")
    print("   apuestas telegram enable     (auto-start en boot)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
