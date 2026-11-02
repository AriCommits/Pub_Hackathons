"""JSON Schema dicts for each deterministic tool."""

from __future__ import annotations

CALCULATOR_SCHEMA: dict = {
    "type": "object",
    "required": ["expression"],
    "additionalProperties": False,
    "properties": {
        "expression": {
            "type": "string",
            "description": "Arithmetic expression using + - * / ** // % and parentheses.",
        }
    },
}

SEARCH_DOCS_SCHEMA: dict = {
    "type": "object",
    "required": ["query"],
    "additionalProperties": False,
    "properties": {
        "query": {"type": "string", "description": "Keyword query over the local document corpus."}
    },
}

QUERY_DATABASE_SCHEMA: dict = {
    "type": "object",
    "required": [],
    "additionalProperties": False,
    "properties": {
        "product": {"type": "string"},
        "region": {"type": "string"},
        "year": {"type": "integer"},
    },
}

LOOKUP_RECORD_SCHEMA: dict = {
    "type": "object",
    "required": ["record_id"],
    "additionalProperties": False,
    "properties": {
        "record_id": {"type": "string", "description": "Record id such as REC-001."}
    },
}

FINISH_SCHEMA: dict = {
    "type": "object",
    "required": ["answer"],
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string", "description": "Final answer to the user. Marks the episode done."}
    },
}

SCHEMAS: dict[str, dict] = {
    "calculator": CALCULATOR_SCHEMA,
    "search_docs": SEARCH_DOCS_SCHEMA,
    "query_database": QUERY_DATABASE_SCHEMA,
    "lookup_record": LOOKUP_RECORD_SCHEMA,
    "finish": FINISH_SCHEMA,
    "no_tool": FINISH_SCHEMA,
}


def validate_args(tool: str, arguments: dict) -> tuple[bool, str]:
    if tool not in SCHEMAS:
        return False, f"unknown tool {tool}"
    schema = SCHEMAS[tool]
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"
    if not schema.get("additionalProperties", True):
        extra = set(arguments) - set(schema.get("properties", {}))
        if extra:
            return False, f"unexpected fields: {sorted(extra)}"
    for key in schema.get("required", []):
        if key not in arguments:
            return False, f"missing required field {key}"
    props = schema.get("properties", {})
    for key, value in arguments.items():
        expected = props.get(key, {}).get("type")
        if expected == "string" and not isinstance(value, str):
            return False, f"{key} must be a string"
        if expected == "integer" and not isinstance(value, int):
            return False, f"{key} must be an integer"
    return True, "ok"
