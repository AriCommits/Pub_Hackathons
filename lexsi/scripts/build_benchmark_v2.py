"""Build an immutable v2 benchmark directory; no historical data is modified."""
import argparse
import json
from pathlib import Path

from actproto.data.benchmark_v2 import build_benchmark
from actproto.experiments.manifest import hash_file
from actproto.training.sft_dataset import format_tool_call


def training_row(item):
    messages = [{"role": "user", "content": item["task"]["prompt"]}]
    for call in item["gold_calls"]:
        messages.append({"role": "assistant", "content": format_tool_call(call["tool"], call["arguments"])})
        if call["tool"] != "finish":
            messages.append({"role": "tool", "content": json.dumps(call["output"], sort_keys=True)})
    return {"split": "train", "messages": messages}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/benchmarks/tooluse_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = Path(args.output)
    # Atomic reservation prevents accidental overwrite of existing frozen data.
    output.mkdir(parents=True, exist_ok=False)
    env, examples, rejects, manifest = build_benchmark(args.seed)
    for axis in ("train", "val", "iid", "stress"):
        with (output / f"{axis}.jsonl").open("x", encoding="utf-8") as handle:
            for item in examples:
                if item["axis"] == axis:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
    with (output / "train_messages.jsonl").open("x", encoding="utf-8") as handle:
        for item in examples:
            if item["axis"] == "train":
                handle.write(json.dumps(training_row(item), sort_keys=True) + "\n")
    fixture = {"identifier": env.identifier, "records": env.records, "sales": env.sales, "documents": env.documents}
    (output / "fixtures.json").write_text(json.dumps(fixture, sort_keys=True), encoding="utf-8")
    (output / "rejects.json").write_text(json.dumps(rejects, indent=2), encoding="utf-8")
    manifest["files"] = {p.name: hash_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    if rejects or any(manifest["counts"].get(k, 0) != v for k, v in manifest["target_counts"].items()):
        raise SystemExit("dataset incomplete; see rejects and manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
