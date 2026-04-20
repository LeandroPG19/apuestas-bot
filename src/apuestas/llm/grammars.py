"""GBNF grammars para forzar JSON structured output en llama.cpp.

llama.cpp server acepta el campo `grammar` en /v1/completion o `json_schema`
en /v1/chat/completions. Usamos grammars por su mayor control.
"""

from __future__ import annotations

# Grammar base de JSON válido + campos específicos.
# Derivado de https://github.com/ggerganov/llama.cpp/blob/master/grammars/json.gbnf
JSON_BASE = r"""
root   ::= object
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? ws "}"
array  ::= "[" ws (value ("," ws value)*)? ws "]"
value  ::= object | array | string | number | boolean | null
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
number ::= "-"? ("0" | [1-9][0-9]*) ("." [0-9]+)? ([eE][-+]?[0-9]+)?
boolean ::= "true" | "false"
null   ::= "null"
ws     ::= [ \t\n\r]*
""".strip()

# Grammar específica para PreMatchAnalysis (versión simplificada que
# valida estructura top-level, los detalles se validan con msgspec).
PRE_MATCH_ANALYSIS = (
    JSON_BASE
    + "\n"
    + r"""
# Override root para exigir campos obligatorios
root ::= "{" ws
    "\"home_team_analysis\"" ws ":" ws object ws "," ws
    "\"away_team_analysis\"" ws ":" ws object ws "," ws
    "\"matchup_context\"" ws ":" ws object ws "," ws
    "\"contradictions_found\"" ws ":" ws array ws "," ws
    "\"line_movement_assessment\"" ws ":" ws line_assessment ws "," ws
    "\"overall_edge_direction\"" ws ":" ws edge_direction ws "," ws
    "\"confidence_in_analysis\"" ws ":" ws confidence ws "," ws
    "\"summary_es\"" ws ":" ws string ws
"}"

line_assessment ::= "\"sharp\"" | "\"public\"" | "\"neutral\"" | "\"unknown\""
edge_direction ::= "\"favors_home\"" | "\"favors_away\"" | "\"neutral\""
confidence ::= "\"low\"" | "\"medium\"" | "\"high\""
""".strip()
)

# Grammar para NER/extraction
NER_EXTRACTION = (
    JSON_BASE
    + "\n"
    + r"""
root ::= "{" ws
    "\"persons\"" ws ":" ws array ws "," ws
    "\"teams\"" ws ":" ws array ws "," ws
    "\"injuries\"" ws ":" ws array ws "," ws
    "\"suspensions\"" ws ":" ws array ws "," ws
    "\"transfers\"" ws ":" ws array ws "," ws
    "\"sentiment\"" ws ":" ws sentiment ws "," ws
    "\"sentiment_score\"" ws ":" ws number ws
"}"

sentiment ::= "\"positive\"" | "\"neutral\"" | "\"negative\""
""".strip()
)

POST_MORTEM = (
    JSON_BASE
    + "\n"
    + r"""
root ::= "{" ws
    "\"outcome\"" ws ":" ws outcome ws "," ws
    "\"prediction_quality\"" ws ":" ws quality ws "," ws
    "\"what_went_right\"" ws ":" ws array ws "," ws
    "\"what_went_wrong\"" ws ":" ws array ws "," ws
    "\"unexpected_factors\"" ws ":" ws array ws "," ws
    "\"if_we_had_known\"" ws ":" ws string ws "," ws
    "\"transferable_lesson\"" ws ":" ws string ws "," ws
    "\"tag_for_pattern_detection\"" ws ":" ws array ws
"}"

outcome ::= "\"won\"" | "\"lost\"" | "\"void\""
quality ::= "\"accurate\"" | "\"off\"" | "\"very_off\""
""".strip()
)


GRAMMARS: dict[str, str] = {
    "pre_match_analysis": PRE_MATCH_ANALYSIS,
    "ner_extraction": NER_EXTRACTION,
    "post_mortem": POST_MORTEM,
}


def get_grammar(name: str) -> str:
    if name not in GRAMMARS:
        msg = f"Unknown grammar: {name}. Available: {list(GRAMMARS)}"
        raise KeyError(msg)
    return GRAMMARS[name]
