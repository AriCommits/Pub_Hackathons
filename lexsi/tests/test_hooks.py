import torch

from actproto.activations.hooks import hooked
from actproto.activations.positions import resolve_layer
from actproto.activations.store import read_dump, write_dump


def test_default_layer_is_two_thirds(tiny_lm):
    assert resolve_layer(tiny_lm, "auto") == (2 * tiny_lm.n_layers) // 3


def test_hook_captures_before_disable(tiny_lm):
    ids = torch.randint(0, 32, (1, 5))
    with hooked(tiny_lm, layer="auto") as hook:
        tiny_lm(input_ids=ids)
        vec = hook.numpy_vector(seq_len=5)
        assert vec.shape == (8,)
        layer_idx = hook.layer_index
    # After disable, a second forward should not update last via a live handle.
    tiny_lm(input_ids=torch.randint(0, 32, (1, 5)))
    assert hook._handle is None
    assert layer_idx == resolve_layer(tiny_lm, "auto")


def test_store_roundtrip(tmp_path):
    records = [
        {
            "task_id": "t1",
            "step": 0,
            "gold_tool": "calculator",
            "pred_tool": None,
            "layer": 2,
            "token_index": 4,
            "model_id": "toy",
            "revision": None,
            "seed": 0,
            "vector": [0.1, 0.2, 0.3],
        }
    ]
    write_dump(tmp_path, records)
    vectors, index = read_dump(tmp_path)
    assert vectors.shape == (1, 3)
    assert index[0]["task_id"] == "t1"
    assert index[0]["gold_tool"] == "calculator"
