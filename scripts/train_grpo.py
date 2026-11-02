#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.configs.schema import load_default  # noqa: E402
from actproto.tools.registry import as_harness_dict  # noqa: E402
from actproto.training.grpo import build_grpo_kwargs, try_create_trainer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AgentTune GRPO (stretch, lowest priority).")
    parser.add_argument("--plan-only", action="store_true", help="Print GRPO wiring summary without training.")
    args = parser.parse_args(argv)
    cfg = load_default()
    kwargs = build_grpo_kwargs(cfg.model_id, dataset=[], tools=list(as_harness_dict().values()))
    info = try_create_trainer(**kwargs)
    print(json.dumps({"kwargs_algorithm": kwargs["algorithm"], "create": {k: info[k] for k in info if k != "kwargs"}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
