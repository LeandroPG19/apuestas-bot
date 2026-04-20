"""Fixtures pytest compartidas."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _env_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secretos mínimos para que pydantic-settings no falle en tests."""
    for k, v in {
        "POSTGRES_PASSWORD": "test-password",
        "VALKEY_PASSWORD": "test-valkey",
    }.items():
        if not os.environ.get(k):
            monkeypatch.setenv(k, v)
