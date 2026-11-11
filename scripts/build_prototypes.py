#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.activations.prototypes import fit_prototypes, labels_from_index  # noqa: E402
from actproto.activations.similarity import cosine_scores, rank_scores  # noqa: E402
from actproto.activations.store import read_dump  # noqa: E402
from actproto.activations.threshold import choose_threshold  # noqa: E402
from actproto.utils.paths import PathEscapeError, contain_under  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def _split(vectors, index, name):
    mask = [row.get("split") == name for row in index]
    if not any(mask):
        return vectors, labels_from_index(index)
    import numpy as np

    idx = np.array(mask)
    return vectors[idx], labels_from_index([row for row, m in zip(index, mask) if m])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit tool prototypes and a val-only abstention threshold.")
    add_trial_argument(parser)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--dump", default=None, help="Activation dump directory. Defaults to trial.yaml activations_dump.")
    parser.add_argument("--prototypes", default=None, help="Output npz path. Defaults to prototypes.npz in prototypes_dir.")
    parser.add_argument("--metrics", default=None, help="Output metrics json. Defaults to prototype_metrics.json.")
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    try:
        dump = (
            contain_under(trial.root, args.dump, label="dump")
            if args.dump
            else trial.resolve(trial.paths.activations_dump)
        )
    except PathEscapeError as exc:
        raise SystemExit(str(exc)) from exc
    if dump is None or not dump.exists():
        raise SystemExit(f"activation dump not found: {dump}")
    vectors, index = read_dump(dump)
    train_v, train_y = _split(vectors, index, "train")
    val_v, val_y = _split(vectors, index, "val")
    prototypes = fit_prototypes(train_v, train_y)
    threshold = choose_threshold(
        val_v,
        val_y,
        prototypes,
        min_coverage=trial.prototype_fit.min_coverage,
    )
    out_dir = trial.resolve(trial.paths.prototypes_dir)
    assert out_dir is not None
    try:
        proto_path = (
            contain_under(trial.root, args.prototypes, label="prototypes")
            if args.prototypes
            else out_dir / "prototypes.npz"
        )
        metrics_path = (
            contain_under(trial.root, args.metrics, label="metrics")
            if args.metrics
            else trial.resolve(trial.paths.prototype_metrics())
        )
    except PathEscapeError as exc:
        raise SystemExit(str(exc)) from exc
    assert metrics_path is not None
    proto_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez(proto_path, **{k: v for k, v in prototypes.items()})
    metrics = {"threshold": threshold, "n_prototypes": len(prototypes), "tools": sorted(prototypes)}
    if args.eval_test:
        test_v, test_y = _split(vectors, index, "test")
        acc = sum(rank_scores(cosine_scores(v, prototypes))["top1"] == y for v, y in zip(test_v, test_y))
        metrics["test_accuracy"] = acc / max(len(test_y), 1)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in metrics if k != "threshold"} | {"threshold": threshold}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
