from actproto.agent.schema_regen import regenerate_arguments


def test_wrong_tool_arguments_are_not_reused():
    prev = {"name": "calculator", "arguments": {"expression": "17*23"}}
    args = regenerate_arguments(
        "lookup_record",
        "Look up REC-001 and report the price.",
        previous_call=prev,
        model_fn=None,
    )
    assert "expression" not in args
    assert args.get("record_id") != "17*23"
    assert args.get("record_id") == "REC-001"


def test_same_tool_keeps_valid_args():
    prev = {"name": "calculator", "arguments": {"expression": "2+2"}}
    args = regenerate_arguments("calculator", "Calculate 9*9", previous_call=prev)
    assert args == {"expression": "2+2"}
