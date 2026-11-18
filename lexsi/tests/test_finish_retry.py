import numpy as np

from actproto.agent.abstention import Threshold
from actproto.agent.routed_strategy import build_routed_project
from actproto.configs.schema import ExperimentConfig


PROTOTYPES = {
    "finish": np.array([1.0, 0.0], dtype=np.float32),
    "lookup_record": np.array([0.0, 1.0], dtype=np.float32),
}
THRESHOLD = Threshold(min_score=0.1, min_margin=0.0)


def _unexpected_inner(_state):
    raise AssertionError("a routed finish must not ask the inner policy to choose the tool")


def _finish_call(answer: str) -> str:
    return f'<tool_call>{{"name":"finish","arguments":{{"answer":"{answer}"}}}}</tool_call>'


def test_empty_finish_retries_once_and_keeps_the_second_answer():
    generated = []

    def generate_fn(messages):
        generated.append(messages)
        if len(generated) == 1:
            return _finish_call("")
        return _finish_call("Stockholm")

    project = build_routed_project(
        ExperimentConfig(max_agent_steps=1),
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=_unexpected_inner,
        vector_fn=lambda _state: np.array([1.0, 0.0], dtype=np.float32),
        generate_fn=generate_fn,
    )

    log = project.infer("What is the capital of Sweden?")

    assert log.tool_results[0].arguments == {"answer": "Stockholm"}
    assert len(generated) == 2
    assert "The finish answer must be non-empty. Emit one tool call." in generated[1][-1]["content"]


def test_two_empty_finish_answers_fall_through_to_the_heuristic():
    calls = {"n": 0}

    def generate_fn(_messages):
        calls["n"] += 1
        return _finish_call("   ")

    project = build_routed_project(
        ExperimentConfig(max_agent_steps=1),
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=_unexpected_inner,
        vector_fn=lambda _state: np.array([1.0, 0.0], dtype=np.float32),
        generate_fn=generate_fn,
    )

    log = project.infer("What is the capital of Sweden?")

    assert log.tool_results[0].name == "finish"
    assert log.tool_results[0].arguments == {"answer": ""}
    assert calls["n"] == 2


def test_lookup_record_still_generates_once():
    calls = {"n": 0}

    def generate_fn(_messages):
        calls["n"] += 1
        return '<tool_call>{"name":"lookup_record","arguments":{"record_id":"REC-001"}}</tool_call>'

    project = build_routed_project(
        ExperimentConfig(max_agent_steps=1),
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=_unexpected_inner,
        vector_fn=lambda _state: np.array([0.0, 1.0], dtype=np.float32),
        generate_fn=generate_fn,
    )

    log = project.infer("Look up REC-001")

    assert log.tool_results[0].arguments == {"record_id": "REC-001"}
    assert calls["n"] == 1
