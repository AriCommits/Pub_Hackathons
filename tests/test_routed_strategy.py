from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from actproto.agent.abstention import Threshold
from actproto.agent.harness import ToolResult
from actproto.agent.project import build_scripted_project
from actproto.agent.routed_strategy import build_routed_project
from actproto.agent.strategy import AgentState, scripted_policy
from actproto.agent.trajectory import EventLog, append_tool_turn
from actproto.configs.schema import ExperimentConfig
from actproto.evaluation import live_runner


PROTOTYPES = {
    "lookup_record": np.array([1.0, 0.0], dtype=np.float32),
    "calculator": np.array([0.0, 1.0], dtype=np.float32),
}
THRESHOLD = Threshold(min_score=0.1, min_margin=0.0)


def test_routed_project_falls_through_on_identical_successful_call():
    inner = scripted_policy(
        [
            {"name": "calculator", "arguments": {"expression": "17*23"}},
            {"name": "finish", "arguments": {"answer": "done"}},
        ]
    )
    proj = build_routed_project(
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=inner,
        vector_fn=lambda _state: np.array([1.0, 0.0], dtype=np.float32),
    )

    log = proj.infer("Look up REC-001")

    assert [result.name for result in log.tool_results] == ["lookup_record", "finish"]
    assert log.tool_results[0].arguments == {"record_id": "REC-001"}
    assert log.final_answer == "done"


def test_routed_project_changes_route_from_transcript():
    inner = scripted_policy(
        [
            {"name": "lookup_record", "arguments": {"record_id": "REC-001"}},
            {"name": "calculator", "arguments": {"expression": "2+2"}},
            {"name": "finish", "arguments": {"answer": "done"}},
        ]
    )

    def vector_fn(state):
        has_tool_message = any(message.get("role") == "tool" for message in state.log.messages)
        return np.array([0.0, 1.0] if has_tool_message else [1.0, 0.0], dtype=np.float32)

    proj = build_routed_project(
        ExperimentConfig(max_agent_steps=3),
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=inner,
        vector_fn=vector_fn,
    )

    log = proj.infer("Look up REC-001, then calculate 2+2")

    assert [result.name for result in log.tool_results] == ["lookup_record", "calculator", "finish"]


def test_routed_project_retries_identical_failed_call():
    inner = scripted_policy(
        [
            {"name": "lookup_record", "arguments": {"record_id": "REC-999"}},
            {"name": "lookup_record", "arguments": {"record_id": "REC-999"}},
        ]
    )
    proj = build_routed_project(
        ExperimentConfig(max_agent_steps=2),
        prototypes=PROTOTYPES,
        threshold=THRESHOLD,
        inner_policy=inner,
        vector_fn=lambda _state: np.array([1.0, 0.0], dtype=np.float32),
    )

    log = proj.infer("Look up REC-999")

    assert [result.name for result in log.tool_results] == ["lookup_record", "lookup_record"]
    assert all(result.arguments == {"record_id": "REC-999"} for result in log.tool_results)
    assert all(result.output["ok"] is False for result in log.tool_results)


def test_make_vector_fn_renders_transcript_after_step_zero(monkeypatch):
    encoded_messages = []

    def fake_encode(_loaded, _prompt, messages=None):
        encoded_messages.append(messages)
        return {"input_ids": SimpleNamespace(shape=(1, 7))}

    @contextmanager
    def fake_hooked(*_args, **_kwargs):
        yield SimpleNamespace(numpy_vector=lambda seq_len: np.array([seq_len], dtype=np.float32))

    class FakeModel:
        def __call__(self, **_batch):
            return None

    monkeypatch.setattr(live_runner, "_encode_prompt", fake_encode)
    monkeypatch.setattr(live_runner, "hooked", fake_hooked)
    vector_fn = live_runner.make_vector_fn(
        SimpleNamespace(model=FakeModel()),
        ExperimentConfig(),
    )
    log = EventLog(
        task="Look up REC-001",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Look up REC-001"},
        ],
    )

    vector_fn(AgentState(task=log.task, step=0, log=log))
    append_tool_turn(
        log,
        ToolResult("lookup_record", {"record_id": "REC-001"}, {"ok": True, "data": {"id": "REC-001"}}),
        "c0",
    )
    vector_fn(AgentState(task=log.task, step=1, log=log, last_result=log.tool_results[-1].output))

    assert encoded_messages[0] is None
    assert encoded_messages[1][-2]["role"] == "assistant"
    assert encoded_messages[1][-2]["content"].startswith("<tool_call>")
    assert encoded_messages[1][-1]["role"] == "user"
    assert encoded_messages[1][-1]["content"].startswith("Tool result:\n")


def test_scripted_project_still_works_without_router():
    proj = build_scripted_project([{"name": "finish", "arguments": {"answer": "hi"}}])
    log = proj.infer("hello")
    assert log.final_answer == "hi"
