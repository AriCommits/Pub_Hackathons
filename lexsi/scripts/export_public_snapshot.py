#!/usr/bin/env python3
"""Write the public Pub_Hackathons tree into a destination directory.

The destination receives a root README plus a lexsi/ project. Sprint plans,
agent logs, legacy adapters, the gated base tokenizer, and internal submission
notes are left out. The selected e2 weights are copied when the Git LFS blob
is smudged.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".agents", ".opencode", ".pytest_cache", "__pycache__", ".venv",
    "archive", "plan_1", "plan_3", "plan_4", "plan_5", "plan_6", "plan_7",
    "plan_8", "plan_9", "plan_10", "plan_11", "plan_12", "plan_13", "plan_14",
    "plan_15",
}
SKIP_FILES = {
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "FACTS.md", "COVERAGE.md", "PUBLISH.md", "FORM.md", "hf_article.md",
    "ACTIVE_PLANS.md", "AUDIT_2026-11-04.md", "EXPERIMENT_STATUS.md",
    "COLAB_ROLLOUT_DIAGNOSTICS.md", "TRAINING_ROADMAP.md", "plan_1.md",
    "brainstorm.md", "depth1.md",
}

ROOT_README = """# ActProto-LoRA

Verified tool use on Tiny Aya Global, for the Lexsi Track 3 hackathon.

A LoRA adapter answers synthetic business questions by calling local tools. A task counts only when the harness runs those calls. The selected policy is frozen supervised adapter **e2**.

The project lives in [`lexsi/`](lexsi/).

## Read this first

- [Write-up](lexsi/submission/ARTICLE.md)
- [Data card](lexsi/submission/DATACARD.md) and the [public data slice](lexsi/submission/data/)
- [Model card](lexsi/submission/MODEL_CARD.md)
- [How to reproduce](lexsi/submission/REPRODUCE.md)
- [Selected adapter](lexsi/results/sft_v2/e2/checkpoint-150)

Article: https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a

Dataset repo: https://huggingface.co/datasets/ArianMoridani/LexsiHackathon

Space: https://huggingface.co/spaces/ArianMoridani/LexsiHackathon

## What this repository is

This is the public code snapshot. It has the library, the tests, the write-up, the public data slice, and the selected LoRA weights. It does not include the gated Tiny Aya base weights, the base tokenizer, the sealed test split, or the internal sprint notes. Those notes stay in the development repository.

## License

Code, adapter, and synthetic data: [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/). See [lexsi/LICENSE.md](lexsi/LICENSE.md). The base model has its own gated license.
"""

SUBMISSION_README = """---
label: coverage
track: T3
audience: judges
status: draft
---

# Submission packet

| File | What it is |
| --- | --- |
| [ARTICLE.md](ARTICLE.md) | The Track 3 write-up |
| [DATACARD.md](DATACARD.md) | Dataset card |
| [data/](data/) | 1,200 training messages and 200 validation prompts |
| [MODEL_CARD.md](MODEL_CARD.md) | Card for frozen adapter e2 |
| [REPRODUCE.md](REPRODUCE.md) | Environment, layout, and commands |
| [space/](space/) | Results page. It does not load the model. |
"""


def _skip(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    if parts[0] == "data":
        return True
    if any(part in SKIP_DIRS or part == "__pycache__" for part in parts):
        return True
    if path.name in SKIP_FILES or path.suffix == ".pyc":
        return True
    if len(parts) >= 2 and parts[0] == "results" and parts[1] == "adapters":
        return True
    return False


def _public_status(text: str) -> str:
    old = """## Docs map

| File | Role |
| --- | --- |
| `STATUS.md` | Current handoff (this file) |
| `RESULTS_2026-11-14.md` | Committed numbers for gitignored `results/**` runs |
| `EXPERIMENT_STATUS.md` | Historical run log |
| `AUDIT_2026-11-04.md` | Historical compliance audit |
| `plan_*` | Historical sprint plans |
| `archive/` | Moved brainstorm / clutter |
| `.opencode/audits/` | Security + cleanliness audit reports |
"""
    new = """## Docs map

| File | Role |
| --- | --- |
| `STATUS.md` | Current handoff (this file) |
| `RESULTS_2026-11-14.md` | Committed numbers for runs whose raw logs are not in git |
| `METHOD.md` | Metric and hypothesis |
| `DATASET_V2.md` | Dataset contract |
| `SUBMISSION.md` | Pointer to the write-up |
| `../submission/` | The public packet |

Sprint plans, the long experiment log, and audit notes are not in this public snapshot.
"""
    if old not in text:
        raise SystemExit("STATUS docs map not found")
    text = text.replace(
        "- Colab / laptop GRPO diagnostics (not study evidence): [`COLAB_ROLLOUT_DIAGNOSTICS.md`](COLAB_ROLLOUT_DIAGNOSTICS.md), `scripts/diagnose_grpo_rollouts.py`\n",
        "- Colab diagnostics are not in this public snapshot. The diagnostic entry point in the full tree is `scripts/diagnose_grpo_rollouts.py`.\n",
    )
    return text.replace(old, new)


def export_public_snapshot(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    lex = dest / "lexsi"
    if lex.exists():
        shutil.rmtree(lex)
    for path in ROOT.rglob("*"):
        if not path.is_file() or _skip(path):
            continue
        target = lex / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (dest / "README.md").write_text(ROOT_README, encoding="utf-8")
    (dest / ".gitattributes").write_text(
        "lexsi/results/sft_v2/e2/checkpoint-150/adapter_model.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    (lex / "submission" / "README.md").write_text(SUBMISSION_README, encoding="utf-8")
    status_path = lex / "docs" / "STATUS.md"
    status_path.write_text(_public_status(status_path.read_text(encoding="utf-8")), encoding="utf-8")
    weights = lex / "results/sft_v2/e2/checkpoint-150/adapter_model.safetensors"
    if weights.is_file() and weights.read_bytes()[:40].startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SystemExit("e2 weights are still a Git LFS pointer; run git lfs pull before exporting")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True, help="Directory that will contain README.md and lexsi/")
    args = parser.parse_args(argv)
    export_public_snapshot(args.dest)
    print(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
