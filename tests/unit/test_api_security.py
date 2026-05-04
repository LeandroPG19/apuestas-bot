"""Tests de hardening del API: rate limit, security headers, metrics guard."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    """Cliente fresh con memory backend (aislado entre tests)."""
    os.environ.setdefault("APUESTAS_ENV", "local")
    os.environ["VALKEY_URL"] = ""  # forzar memory backend determinista
    from apuestas.api.main import app, limiter

    limiter.reset()
    return TestClient(app)


def test_security_headers_presentes(client: TestClient) -> None:
    r = client.get("/")
    for hdr in (
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
        "Permissions-Policy",
        "Cross-Origin-Resource-Policy",
        "Content-Security-Policy",
    ):
        assert hdr in r.headers, f"header faltante: {hdr}"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_rate_limit_version_expuesto_en_headers(client: TestClient) -> None:
    r = client.get("/version")
    assert r.status_code == 200
    assert r.headers.get("X-RateLimit-Limit") == "30"
    assert r.headers.get("X-RateLimit-Remaining") == "29"


def test_rate_limit_version_bloquea_tras_exceso(client: TestClient) -> None:
    for _ in range(30):
        client.get("/version")
    r = client.get("/version")
    assert r.status_code == 429
    assert r.headers.get("Retry-After") == "60"
    body = r.json()
    assert body["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_health_tolera_120_per_min(client: TestClient) -> None:
    """Docker healthcheck polling no debe quedar bloqueado."""
    for _ in range(100):
        assert client.get("/health").status_code == 200


def test_rate_limit_independiente_por_endpoint(client: TestClient) -> None:
    """Agotar /version NO debe afectar a /health ni a /."""
    for _ in range(30):
        client.get("/version")
    assert client.get("/version").status_code == 429
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_metrics_guard_sin_token_permite_sin_config(client: TestClient) -> None:
    """Si APUESTAS_METRICS_TOKEN vacío, /metrics sigue abierto (backward-compat)."""
    r = client.get("/metrics")
    assert r.status_code in (200, 404)


_telegram_available = True
try:
    import telegram  # noqa: F401
except ImportError:
    _telegram_available = False


@pytest.mark.skipif(not _telegram_available, reason="python-telegram-bot opcional")
def test_html_escape_neutraliza_injection() -> None:
    """_escape_md (ahora HTML-safe) neutraliza tags HTML inyectados.

    El bot corre con `parse_mode=HTML` (Gap UX). El escape previene que content
    externo (narrativas LLM, DB) inyecte tags activos como `<a href=...>`.
    """
    from apuestas.bot.telegram import _escape_md

    payload = 'inj <a href="http://evil">click</a> <script>alert(1)</script>'
    escaped = _escape_md(payload)
    assert "&lt;a href=" in escaped
    assert "&lt;script&gt;" in escaped
    assert "<a " not in escaped
    assert "<script>" not in escaped


@pytest.mark.skipif(not _telegram_available, reason="python-telegram-bot opcional")
def test_html_escape_respeta_vacio() -> None:
    from apuestas.bot.telegram import _escape_md

    assert _escape_md("") == ""
    # Plain sin caracteres especiales pasa sin cambios
    assert _escape_md("plain text") == "plain text"
    # Ampersand se escapa
    assert _escape_md("a & b") == "a &amp; b"


@pytest.mark.skipif(not _telegram_available, reason="python-telegram-bot opcional")
def test_telegram_auth_fail_closed_sin_config() -> None:
    """_chat_authorized rechaza si TELEGRAM_CHAT_ID no está configurado."""
    from unittest.mock import MagicMock

    from apuestas.bot.telegram import _chat_authorized

    update = MagicMock()
    update.effective_chat.id = 12345

    import apuestas.bot.telegram as tg_mod

    original_settings = tg_mod.get_settings()
    saved = original_settings.apis.telegram_chat_id
    try:
        original_settings.apis.telegram_chat_id = None
        assert _chat_authorized(update) is False
    finally:
        original_settings.apis.telegram_chat_id = saved
