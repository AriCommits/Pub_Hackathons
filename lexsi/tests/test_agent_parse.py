from actproto.agent.parse import ParseError, ToolCall, parse_tool_call


def test_parse_tool_call_tag():
    text = '<tool_call>{"name": "calculator", "arguments": {"expression": "1+2"}}</tool_call>'
    parsed = parse_tool_call(text)
    assert isinstance(parsed, ToolCall)
    assert parsed.name == "calculator"
    assert parsed.arguments["expression"] == "1+2"


def test_parse_fenced_json():
    text = '```json\n{"name": "finish", "arguments": {"answer": "5"}}\n```'
    parsed = parse_tool_call(text)
    assert parsed.name == "finish"


def test_parse_invalid_fails_closed():
    parsed = parse_tool_call("I think we should maybe calculate something")
    assert isinstance(parsed, ParseError)


def test_parse_multiple_json_refuses():
    text = (
        '<tool_call>{"name": "calculator", "arguments": {"expression": "1"}}</tool_call>'
        '<tool_call>{"name": "finish", "arguments": {"answer": "1"}}</tool_call>'
    )
    parsed = parse_tool_call(text)
    assert isinstance(parsed, ParseError)
    assert "multiple" in parsed.message
