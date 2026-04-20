"""Loader y registro de prompts versionados YAML."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)

PROMPTS_ROOT = Path(__file__).resolve().parents[3] / "prompts"


@dataclass(slots=True, frozen=True)
class PromptTemplate:
    name: str
    version: str
    description: str
    grammar: str | None
    schema: str | None
    temperature: float
    max_tokens: int
    system: str
    user_template: str

    @property
    def full_id(self) -> str:
        return f"{self.name}/{self.version}"

    def render(self, **kwargs: Any) -> str:
        """Renderiza el user_template con format strings simples."""
        try:
            return self.user_template.format(**kwargs)
        except KeyError as exc:
            msg = f"Prompt {self.full_id} requiere variable: {exc}"
            raise ValueError(msg) from exc


def _find_yaml(category: str, version: str) -> Path:
    """Busca prompts/{category}/{version}.yaml o prompts/{category}_{version}.yaml."""
    candidates = [
        PROMPTS_ROOT / category / f"{version}.yaml",
        PROMPTS_ROOT / f"{category}_{version}.yaml",
        PROMPTS_ROOT / category / f"{category}_{version}.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    msg = f"Prompt not found: category={category} version={version}. Searched: {candidates}"
    raise FileNotFoundError(msg)


@lru_cache(maxsize=32)
def load_prompt(category: str, version: str = "v1") -> PromptTemplate:
    """Carga y cachea un prompt desde YAML."""
    path = _find_yaml(category, version)
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    return PromptTemplate(
        name=data["name"],
        version=data["version"],
        description=data.get("description", ""),
        grammar=data.get("grammar"),
        schema=data.get("schema"),
        temperature=float(data.get("temperature", 0.1)),
        max_tokens=int(data.get("max_tokens", 1024)),
        system=data["system"].strip(),
        user_template=data["user_template"].strip(),
    )


def clear_cache() -> None:
    """Útil en tests / hot-reload de prompts."""
    load_prompt.cache_clear()
