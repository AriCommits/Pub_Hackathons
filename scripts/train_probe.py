#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.activations.linear_probe import evaluate_probe, save_probe, train_probe  # noqa: E402
from actproto.activations.prototypes import labels_from_index  # noqa: E402
from actproto.activations.store import read_dump  # noqa: E402


def _split(vectors, index, name):
    mask = [row.get("split") == name for row in index]
    import numpy as np

    if not any(mask):
        return vectors, labels_from_index(index)
    idx = np.array(mask)
    return vectors[idx], labels_from_index([row for row, m in zip(index, mask) if m])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a linear probe on frozen activations.")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    vectors, index = read_dump(args.dump)
    train_v, train_y = _split(vectors, index, "train")
    val_v, val_y = _split(vectors, index, "val")
    test_v, test_y = _split(vectors, index, "test")
    clf, enc = train_probe(train_v, train_y)
    out = Path(args.out)
    save_probe(out / "linear_probe.joblib", clf, enc)
    metrics = {
        "val_accuracy": evaluate_probe(clf, enc, val_v, val_y)["accuracy"],
        "test_accuracy": evaluate_probe(clf, enc, test_v, test_y)["accuracy"],
        "n_classes": len(enc.classes_),
        **evaluate_probe(clf, enc, test_v, test_y),
    }
    (out / "probe_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"val_accuracy": metrics["val_accuracy"], "test_accuracy": metrics["test_accuracy"], "n_classes": metrics["n_classes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
