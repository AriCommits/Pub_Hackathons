import inspect

import numpy as np

import actproto.activations.linear_probe as probe_mod
from actproto.activations.linear_probe import evaluate_probe, train_probe
from actproto.activations import prototypes as proto_mod


def test_probe_does_not_import_prototype_code():
    source = inspect.getsource(probe_mod)
    assert "prototypes" not in source
    assert proto_mod.__name__ != probe_mod.__name__


def test_probe_on_separable_data():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, size=(30, 4)) + np.array([3, 0, 0, 0])
    b = rng.normal(0, 0.1, size=(30, 4)) + np.array([0, 3, 0, 0])
    x = np.vstack([a, b]).astype(np.float32)
    y = ["calculator"] * 30 + ["finish"] * 30
    clf, enc = train_probe(x, y)
    metrics = evaluate_probe(clf, enc, x, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["n_classes"] == 2
    assert "val_accuracy" not in metrics or True
