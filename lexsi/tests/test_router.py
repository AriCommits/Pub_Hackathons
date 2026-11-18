import numpy as np

from actproto.agent.abstention import Threshold
from actproto.agent.router import route_vector


def test_confident_route_selects_tool():
    prototypes = {
        "calculator": np.array([1.0, 0.0], dtype=np.float32),
        "finish": np.array([0.0, 1.0], dtype=np.float32),
    }
    vec = np.array([1.0, 0.0], dtype=np.float32)
    decision = route_vector(vec, prototypes, Threshold(min_score=0.5, min_margin=0.1))
    assert decision["abstain"] is False
    assert decision["tool"] == "calculator"


def test_low_confidence_abstains():
    prototypes = {
        "calculator": np.array([1.0, 0.0], dtype=np.float32),
        "finish": np.array([0.99, 0.01], dtype=np.float32),
    }
    vec = np.array([1.0, 0.0], dtype=np.float32)
    decision = route_vector(vec, prototypes, Threshold(min_score=0.99, min_margin=0.5))
    assert decision["abstain"] is True
    assert decision["tool"] is None


def test_missing_threshold_fails_open():
    prototypes = {"calculator": np.array([1.0, 0.0], dtype=np.float32)}
    vec = np.array([1.0, 0.0], dtype=np.float32)
    decision = route_vector(vec, prototypes, None)
    assert decision["abstain"] is True
