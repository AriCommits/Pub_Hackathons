from actproto.agent.project import build_scripted_project


def test_scripted_calculator_then_finish():
    proj = build_scripted_project(
        [
            {"name": "calculator", "arguments": {"expression": "2+3"}},
            {"name": "finish", "arguments": {"answer": "5"}},
        ]
    )
    log = proj.infer("what is 2+3?")
    assert log.final_answer == "5"
    assert log.final_state.get("done") is True
    names = [r.name for r in log.tool_results]
    assert names == ["calculator", "finish"]
    assert log.tool_results[0].output["ok"] is True
    assert log.tool_results[0].output["data"]["value"] == 5.0


def test_build_project_requires_policy():
    from actproto.agent.project import build_project
    from actproto.configs.schema import load_default

    try:
        build_project(load_default())
        raised = False
    except ValueError:
        raised = True
    assert raised
