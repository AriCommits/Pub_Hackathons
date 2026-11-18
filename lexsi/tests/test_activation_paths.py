import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from actproto.activations.store import write_dump
from lexsi.scripts import build_prototypes
from lexsi.scripts import extract_activations

ROOT = Path(__file__).resolve().parents[1]


def test_extract_plan_only_uses_override_paths(capsys):
    assert (
        extract_activations.main(
            [
                "--plan-only",
                "--split",
                "train",
                "--limit",
                "1",
                "--adapter",
                "results/adapters/b1_assistant_only",
                "--out",
                "results/activations_b1a",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter"].endswith("b1_assistant_only")
    assert payload["out"].endswith("activations_b1a")
    assert payload["n_tasks"] == 1


def test_build_prototypes_writes_only_the_requested_files():
    # Paths must stay under the project root (contain_under); use an in-tree temp dir.
    work = Path(tempfile.mkdtemp(prefix=".pytest_proto_", dir=ROOT))
    try:
        records = []
        for split, label, vec in (
            ("train", "calculator", [1.0, 0.0]),
            ("train", "finish", [0.0, 1.0]),
            ("val", "calculator", [0.9, 0.1]),
            ("val", "finish", [0.1, 0.9]),
        ):
            records.append({"vector": np.array(vec, dtype=np.float32), "gold_tool": label, "split": split})
        dump = work / "dump"
        write_dump(dump, records)
        proto = work / "prototypes_b1a.npz"
        metrics = work / "prototype_metrics_b1a.json"

        assert (
            build_prototypes.main(
                ["--dump", str(dump), "--prototypes", str(proto), "--metrics", str(metrics), "--eval-test"]
            )
            == 0
        )
        assert proto.exists()
        assert metrics.exists()
        saved = np.load(proto)
        assert set(saved.files) == {"calculator", "finish"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
