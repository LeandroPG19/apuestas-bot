"""Tests del loader de prompts YAML."""

from __future__ import annotations

import pytest

from apuestas.llm.prompts import clear_cache, load_prompt


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_cache()


def test_load_pre_match_v1() -> None:
    p = load_prompt("pre_match", "v1")
    assert p.name == "pre_match"
    assert p.version == "v1"
    assert p.grammar == "pre_match_analysis"
    assert p.schema == "apuestas.schemas.llm.PreMatchAnalysis"
    assert p.temperature <= 0.2
    assert "home_team_analysis" in p.system.lower() or "espejada" in p.system.lower()


def test_load_nlp_ner_v1() -> None:
    p = load_prompt("nlp/ner", "v1")
    assert p.grammar == "ner_extraction"
    assert p.schema.endswith("NERExtraction")


def test_load_post_mortem_v1() -> None:
    p = load_prompt("post_mortem", "v1")
    assert p.grammar == "post_mortem"
    assert "prediction_quality" in p.system.lower()


def test_load_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist", "v99")


def test_render_with_vars() -> None:
    p = load_prompt("nlp/ner", "v1")
    out = p.render(content="LeBron jugará esta noche", lang="es", source="ESPN")
    assert "LeBron" in out
    assert "ESPN" in out


def test_render_missing_var_raises() -> None:
    p = load_prompt("nlp/ner", "v1")
    with pytest.raises(ValueError, match="requiere variable"):
        p.render(content="hola")  # faltan lang y source
