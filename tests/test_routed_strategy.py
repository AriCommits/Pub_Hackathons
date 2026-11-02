import numpy as np

from actproto.agent.abstention import Threshold
from actproto.agent.project import build_scripted_project
from actproto.agent.routed_strategy import build_routed_project
from actproto.agent.strategy import scripted_policy


def test_routed_project_forces_lookup_and_regenerates_args():
    prototypes = {
        "lookup_record": np.array([1.0, 0.0], dtype=np.float32),
        "calculator": np.array([0.0, 1.0], dtype=np.float32),
    }
    inner = scripted_policy(
        [
            {"name": "calculator", "arguments": {"expression": "17*23"}},
            {"name": "finish", "arguments": {"answer": "done"}},
        ]
    )
    proj = build_routed_project(
        prototypes=prototypes,
        threshold=Threshold(min_score=0.1, min_margin=0.0),
        inner_policy=inner,
        vector_fn=lambda _state: np.array([1.0, 0.0], dtype=np.float32),
    )
    log = proj.infer("Look up REC-001")
    assert log.tool_results[0].name == "lookup_record"
    assert log.tool_results[0].arguments.get("record_id") == "REC-001"
    assert "expression" not in log.tool_results[0].arguments


def test_scripted_project_still_works_without_router():
    proj = build_scripted_project([{"name": "finish", "arguments": {"answer": "hi"}}])
    log = proj.infer("hello")
    assert log.final_answer == "hi"
