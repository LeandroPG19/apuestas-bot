"""Tests de GBNF grammars."""

from __future__ import annotations

import pytest

from apuestas.llm.grammars import GRAMMARS, get_grammar


def test_get_grammar_known() -> None:
    assert get_grammar("pre_match_analysis").startswith("root")
    assert get_grammar("ner_extraction").startswith("root")
    assert get_grammar("post_mortem").startswith("root")


def test_get_grammar_unknown() -> None:
    with pytest.raises(KeyError):
        get_grammar("nonexistent_grammar")


def test_grammars_define_root() -> None:
    for name, grammar in GRAMMARS.items():
        assert "root ::=" in grammar, f"Grammar {name} sin root"


def test_grammars_define_enums_in_pre_match() -> None:
    g = get_grammar("pre_match_analysis")
    # GBNF escapa comillas internas; matcheamos el nombre del literal
    assert "sharp" in g
    assert "public" in g
    assert "favors_home" in g
    assert "favors_away" in g
    assert "low" in g and "medium" in g and "high" in g


def test_post_mortem_grammar_has_outcome_enums() -> None:
    g = get_grammar("post_mortem")
    assert "won" in g
    assert "lost" in g
    assert "void" in g
