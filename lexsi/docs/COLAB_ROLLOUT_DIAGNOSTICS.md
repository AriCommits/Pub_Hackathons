# Colab rollout diagnostics (GRPO −0.1)

Use Google Colab (T4/L4) when the lab RTX 5060 Ti dual-GPU host is unavailable. Full claimed GRPO study runs still require that host; Colab is for **diagnostic dumps only**.

## Reviewer checks — where to run them

| # | Check | Laptop CPU | Colab T4/L4 | Lab 5060 Ti |
| --- | --- | --- | --- | --- |
| 1 | Dump 3–4 raw rollouts (prompt, completion, ledger) | Fixture dumps only (`--fixture-rollouts`) | **Yes** with Tiny Aya + uploaded e2 adapter (`--live-rollouts --allow-any-gpu`) | Yes (UUID pin) |
| 2 | Diff train vs eval prompt (template / system / tools / prefix) | **Yes** (`--prompt-diff`) | Yes (same, CPU fine) | Yes |
| 3 | Single-turn vs AgentTune multi-turn | **Yes from code + fixture dumps** (no GPU) | Optional live confirm | Optional |
| 4 | Truncation at `max_completion_length=256` vs EOS | Stats on fixtures only | **Yes** live truncation stats | Yes |

## Code finding (no GPU required)

Training does **not** use a single-turn TRL completion for the study path. It uses `BehaviorToolRollout` → `run_trusted_behavior_episode`, which samples, executes tools, appends `Tool result:` observations, and samples again until `finish` / failure. Live eval uses multi-turn `ReActStrategy` with the same `system_prompt()`.

`bounded_prefix_v1` reward **−0.1** means: failed task, **k=0** (no work-tool step matched), **exactly one** `invalid_attempts` (for example `parse_error` or `truncation` on the first decision). That ends the episode before any tool match — identical −0.1 on every rollout is consistent with a systematic first-turn failure under train sampling (`temperature=0.8`), not with “training is single-turn.”

## Secrets and artifacts

- **HF token**: Colab secret `HF_TOKEN` (or `huggingface-cli login`). Model `CohereLabs/tiny-aya-global` is gated; revision pinned in `actproto/configs/model.yaml`.
- **e2 adapter**: **local-only** on the Windows study host (`docs/SFT_BASELINE_MANIFEST.json` points at a machine path). Upload `results/sft_v2/e2/checkpoint-150` to Colab (Drive zip or `huggingface_hub` private repo). Without it, live dumps use the base model only (still useful for truncation / format probes).
- **AgentTune / full GRPO stack**: not required for these dumps. The diagnostic script calls the trusted episode helper directly (`torch` + `transformers` + `peft`).

## UUID pin (fails closed on Colab)

Checked-in GPU configs pin `device_uuid: 134c34ef-c155-b001-4e3a-b9134b072c21` (RTX 5060 Ti). Colab will **not** have that UUID. Without an override, device selection fails closed.

Diagnostic override (never for claimed study evidence):

```bash
export LEXSI_ALLOW_ANY_GPU=1
# or
python scripts/diagnose_grpo_rollouts.py --live-rollouts --allow-any-gpu --adapter /path/to/checkpoint-150
# Real GRPO train set (not the toy 1+1 demo task):
python scripts/diagnose_grpo_rollouts.py --live-rollouts --allow-any-gpu \
  --adapter /path/to/checkpoint-150 \
  --manifest data/benchmarks/tooluse_v2/manifest.json \
  --count 8 --output /content/manifest_live_rollouts.json
python scripts/train_grpo.py --smoke --live-gpu --allow-any-gpu --config actproto/configs/grpo_v2_gpu.yaml
```

You will see: `DIAGNOSTIC-ONLY: … bypasses the RTX 5060 Ti UUID pin`.

## Colab steps

1. Runtime → GPU (T4 is enough for a few Tiny Aya bf16 rollouts; ~6.5 GB weights).
2. Clone and install a thin stack (not the full Windows `requirements-rl-gpu.lock` unless you also want AgentTune smokes):

```bash
!git clone https://github.com/AriCommits/Hackathons.git
%cd Hackathons/lexsi
!pip -q install -e ".[train]"
```

3. Secrets → add `HF_TOKEN`. In a cell:

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
```

4. Upload or mount the e2 adapter directory to e.g. `/content/e2-checkpoint-150`.

5. CPU checks first (no GPU needed):

```bash
!python scripts/diagnose_grpo_rollouts.py --prompt-diff
!python scripts/diagnose_grpo_rollouts.py --explain-reward
!python scripts/diagnose_grpo_rollouts.py --fixture-rollouts --count 4
```

6. Live dumps on the Colab GPU:

```bash
!python scripts/diagnose_grpo_rollouts.py --live-rollouts --allow-any-gpu \
  --adapter /content/e2-checkpoint-150 --count 4 \
  --output /content/rollout_dumps.json
```

Inspect each rollout’s `decoded_completion`, `completion_reason`, and `ledger` (`invalid_attempts`, `reward`). Truncation stats are in `truncation_stats`.

## Notebook

Open [`notebooks/colab_grpo_rollout_diagnostics.ipynb`](../notebooks/colab_grpo_rollout_diagnostics.ipynb) in Colab (`File → Upload` or raw GitHub URL).

## Unattended one-shot (close laptop after start)

Use this when you have **&lt;2 minutes** on the laptop. Goal: start a Colab GPU cell, let it write results to Drive, then leave. You do **not** need to watch the run.

### Caveats (honest)

- Free Colab can still disconnect if the **browser tab** dies or the runtime idles. Leaving the tab open on a phone/another machine helps; Colab Pro is more reliable.
- There is no official “email me when done” API. The durable artifact is the JSON on Drive (and the printed `completion_reason` Counter in the cell output if the session survives).
- e2 adapter is still **local-only** on the Windows study host unless you already uploaded it to Drive. Without it, the dump uses base Tiny Aya (still useful for format / truncation / template probes).

### Prep once (can be done earlier)

1. Colab secret `HF_TOKEN` with access to gated `CohereLabs/tiny-aya-global`.
2. Optional: copy `checkpoint-150` to Drive as `MyDrive/lexsi/e2-checkpoint-150/`.
3. Open the notebook from this branch (or paste the unattended cell below).

### Fire-and-forget cell

Runtime → GPU, then run **one** cell:

```python
# Unattended GRPO rollout dump → Google Drive
import json, os, subprocess, sys
from collections import Counter
from pathlib import Path

from google.colab import drive, userdata

drive.mount("/content/drive", force_remount=False)
OUT_DIR = Path("/content/drive/MyDrive/lexsi/grpo_diag")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "live_rollouts.json"
SUMMARY = OUT_DIR / "live_rollouts_summary.json"

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
os.environ["LEXSI_ALLOW_ANY_GPU"] = "1"

REPO = Path("/content/Hackathons")
if not (REPO / "lexsi").exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b",
                           "cursor/lexsi-colab-rollout-diagnostics-8ccb",
                           "https://github.com/AriCommits/Hackathons.git", str(REPO)])
os.chdir(REPO / "lexsi")
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "-e", ".[train]"])

ADAPTER = Path("/content/drive/MyDrive/lexsi/e2-checkpoint-150")
cmd = [sys.executable, "scripts/diagnose_grpo_rollouts.py", "--live-rollouts",
       "--allow-any-gpu", "--count", "4", "--output", str(OUT)]
if ADAPTER.is_dir():
    cmd.extend(["--adapter", str(ADAPTER)])

print("STARTING unattended dump; safe to close laptop after this line prints.", flush=True)
subprocess.check_call(cmd)
payload = json.loads(OUT.read_text())
reasons = Counter(row["completion_reason"] for row in payload["rollouts"])
summary = {
    "output": str(OUT),
    "completion_reason_counts": dict(reasons),
    "rewards": [row["ledger"]["reward"] for row in payload["rollouts"]],
    "invalid_attempts": [row["ledger"]["invalid_attempts"] for row in payload["rollouts"]],
    "adapter": str(ADAPTER) if ADAPTER.is_dir() else None,
    "truncation_stats": payload.get("truncation_stats"),
}
SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("DONE", summary)
print("Forward this Counter later:", reasons)
```

After you return: open `MyDrive/lexsi/grpo_diag/live_rollouts_summary.json` (or the full `live_rollouts.json`) and forward the `completion_reason_counts` Counter. That answers template_mismatch vs truncation vs parse_error without another interactive session.

### Laptop time budget

| Step | On laptop |
| --- | --- |
| Open Colab + GPU runtime + paste/run unattended cell | ~1–2 min |
| Wait for clone/pip/model download + 4 rollouts | **0** (leave) |
| Later: open Drive summary / forward Counter | ~30 s when convenient |

## What still needs the lab machine

- Full 20-step GRPO study with the pinned UUID and protected output dirs
- Memory / dual-card isolation claims (`measure_rl_gpu_memory.py` still expects the 5060 Ti)
- Any result labeled as the official study run (do not claim Colab `--allow-any-gpu` runs as study evidence)
