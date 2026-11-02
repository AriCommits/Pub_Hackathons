import pytest

from actproto.tools.registry import PRIMARY_TOOLS, call_tool, tool_names
from actproto.tools.schemas import SCHEMAS, validate_args


def test_registry_lists_five_primary_tools():
    assert tool_names() == list(PRIMARY_TOOLS)
    assert set(PRIMARY_TOOLS) == {"calculator", "search_docs", "query_database", "lookup_record", "finish"}
    for name in PRIMARY_TOOLS:
        assert name in SCHEMAS


def test_no_tool_alias_matches_finish():
    a = call_tool("finish", {"answer": "ok"})
    b = call_tool("no_tool", {"answer": "ok"})
    assert a == b
    assert a["ok"] is True


def test_calculator_happy_path():
    result = call_tool("calculator", {"expression": "17 * 23"})
    assert result["ok"]
    assert result["data"]["value"] == pytest.approx(391.0)


def test_calculator_rejects_attribute_access():
    result = call_tool("calculator", {"expression": "__import__('os').system('echo hi')"})
    assert result["ok"] is False
    assert "error" in result
    result2 = call_tool("calculator", {"expression": "(lambda: 1)()"})
    assert result2["ok"] is False


def test_lookup_missing_id():
    result = call_tool("lookup_record", {"record_id": "REC-404"})
    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_query_and_search_happy_paths():
    db = call_tool("query_database", {"product": "widget", "year": 2024})
    assert db["ok"] and db["data"]["count"] == 1
    docs = call_tool("search_docs", {"query": "Widget specification"})
    assert docs["ok"] and docs["data"]["count"] >= 1


def test_schema_rejects_extra_fields():
    ok, reason = validate_args("calculator", {"expression": "1+1", "hack": True})
    assert ok is False
    assert "unexpected" in reason
