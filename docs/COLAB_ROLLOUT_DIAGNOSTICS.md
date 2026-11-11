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

## What still needs the lab machine

- Full 20-step GRPO study with the pinned UUID and protected output dirs
- Memory / dual-card isolation claims (`measure_rl_gpu_memory.py` still expects the 5060 Ti)
- Any result labeled as the official study run (do not claim Colab `--allow-any-gpu` runs as study evidence)
