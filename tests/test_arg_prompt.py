import numpy as np

from actproto.agent.abstention import Threshold
from actproto.agent.arg_prompt import argument_prompt
from actproto.agent.routed_strategy import build_routed_project
from actproto.configs.schema import ExperimentConfig


PROTOTYPES = {
    "lookup_record": np.array([1.0, 0.0], dtype=np.float32),
    "calculator": np.array([0.0, 1.0], dtype=np.float32),
}
THRESHOLD = Threshold(min_score=0.1, min_margin=0.0)


def _unexpected_inner(_state):
    raise AssertionError("normal routed calls must not ask the inner policy to choose a tool")


def test_argument_prompt_names_only_selected_tool_and_argument_keys():
    prompt = argument_prompt(
        "lookup_record",
        {"record_id": {"type": "string", "description": "Record id such as REC-001."}},
    )

    assert prompt == "Use this tool only: lookup_record(record_id). Emit one tool call."
    assert "REC-001" not in prompt
    assert "calculator" not in prompt


def test_routed_project_uses_valid_generated_arguments():
    generated_messages = []

    def generate_fn(messages):
        generated_messages.append(messages)
        return '<tool_call>{"name":"lookup_record","arguments":{"record_id":"REC-777"}}</tool_call>'

    project = build_routed_project(
        ExperimentConfig(max_agent_steps=1),
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=_unexpected_inner,
        vector_fn=lambda _state: np.array([1.0, 0.0], dtype=np.float32),
        generate_fn=generate_fn,
    )

    log = project.infer("Look up REC-001")

    assert log.tool_results[0].name == "lookup_record"
    assert log.tool_results[0].arguments == {"record_id": "REC-777"}
    assert generated_messages[0][-1] == {
        "role": "user",
        "content": (
            "Look up REC-001\n\n"
            "Use this tool only: lookup_record(record_id). Emit one tool call."
        ),
    }
    assert [message["role"] for message in generated_messages[0]] == ["system", "user"]
    assert len(generated_messages) == 1


def test_routed_project_discards_different_tool_arguments():
    def generate_fn(_messages):
        return '<tool_call>{"name":"calculator","arguments":{"expression":"17*23"}}</tool_call>'

    project = build_routed_project(
        ExperimentConfig(max_agent_steps=1),
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=_unexpected_inner,
        vector_fn=lambda _state: np.array([1.0, 0.0], dtype=np.float32),
        generate_fn=generate_fn,
    )

    log = project.infer("Look up REC-001 and report the price.")

    assert log.tool_results[0].name == "lookup_record"
    assert log.tool_results[0].arguments == {"record_id": "REC-001"}
