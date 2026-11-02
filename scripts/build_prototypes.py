#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.activations.prototypes import fit_prototypes, labels_from_index  # noqa: E402
from actproto.activations.similarity import cosine_scores, rank_scores  # noqa: E402
from actproto.activations.store import read_dump  # noqa: E402
from actproto.activations.threshold import choose_threshold  # noqa: E402


def _split(vectors, index, name):
    mask = [row.get("split") == name for row in index]
    if not any(mask):
        return vectors, labels_from_index(index)
    idx = np.array(mask)
    return vectors[idx], labels_from_index([row for row, m in zip(index, mask) if m])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit tool prototypes and a val-only abstention threshold.")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args(argv)
    vectors, index = read_dump(args.dump)
    train_v, train_y = _split(vectors, index, "train")
    val_v, val_y = _split(vectors, index, "val")
    prototypes = fit_prototypes(train_v, train_y)
    threshold = choose_threshold(val_v, val_y, prototypes, min_coverage=args.min_coverage)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "prototypes.npz", **{k: v for k, v in prototypes.items()})
    metrics = {"threshold": threshold, "n_prototypes": len(prototypes), "tools": sorted(prototypes)}
    if args.eval_test:
        test_v, test_y = _split(vectors, index, "test")
        acc = sum(rank_scores(cosine_scores(v, prototypes))["top1"] == y for v, y in zip(test_v, test_y))
        metrics["test_accuracy"] = acc / max(len(test_y), 1)
    (out / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in metrics if k != "threshold"} | {"threshold": threshold}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
