#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.activations.layer_select import pick_layer, try_circuitkit, write_layer_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write configs/layer.yaml from a score dict (JSON).")
    parser.add_argument("--scores", required=True, help='JSON object mapping layer index to score, e.g. {"2": 0.8}')
    parser.add_argument("--out", default=str(ROOT / "actproto" / "configs" / "layer.yaml"))
    args = parser.parse_args(argv)
    raw = json.loads(Path(args.scores).read_text(encoding="utf-8")) if Path(args.scores).exists() else json.loads(args.scores)
    scores = {int(k): float(v) for k, v in raw.items()}
    layer, score = pick_layer(scores)
    method = "circuitkit" if try_circuitkit() else "sweep"
    write_layer_config(args.out, layer, method, score)
    print(json.dumps({"layer": layer, "score": score, "method": method}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
