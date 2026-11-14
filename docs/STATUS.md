# Lexsi ActProto — current status

Canonical handoff. Prefer this file over older plan docs and the long `EXPERIMENT_STATUS.md` log.

## Produce LoRA adapters (next)

1. **SFT-v2 (preferred):** train / resume under `results/sft_v2/` via `scripts/train_sft_v2.py` or `scripts/run_sft_study.py`. Frozen selection remains `e2` (see manifest below).
2. **Legacy LoRA:** `scripts/train_lora.py` → `results/adapters/lora_run` (never overwrite preserved `results/adapters/b1` unless intentional).
3. **Eval:** `scripts/run_trial.py` (B0–B2) and/or `scripts/run_experiment_v2.py` (tool-use v2).
4. **Export:** `scripts/export_hf_artifacts.py`.

Stretch-only code (steering, alignment-loss stub, legacy reward helpers) lives under `actproto/stretch/`. Contaminated train-echo CLIs and the non-training `train_aligned` stub live under `scripts/archive/`.

## Frozen SFT baseline

- Record: [`SFT_BASELINE_MANIFEST.json`](SFT_BASELINE_MANIFEST.json)
- Selected id: `e2` under `results/sft_v2/e2/` (checkpoint-150 on the study host)
- Trial HF export default: `results/sft_v2/e2` (see `actproto/configs/trial.yaml` `export.adapter`)
- Historical LoRA B1 adapters remain at `results/adapters/b1` and `results/adapters/b1_assistant_only` for legacy B0–B2 runs; **do not train into `b1` by default** (`sft.yaml` defaults to `results/adapters/lora_run`)

## GRPO

- Live train entry: `scripts/train_grpo.py` → `actproto/training/grpo_run.py`
- Live configs under `actproto/configs/`:
  - `grpo_v2.yaml` — CPU smoke / plan
  - `grpo_v2_gpu.yaml` — GPU fixture smoke
  - `grpo_v2_gpu_prefix.yaml` — study profile (use a **new** `--output` under `results/rl_v2/`)
- Historical YAMLs live in `actproto/configs/archive/` and require `--allow-historical`
- Protected output dirs include `grpo_gpu_e2`, `grpo_gpu_tools`, `grpo_gpu_prefix`, `grpo_gpu_study`, smokes
- **Root cause of constant −0.1:** sampling while `model.train()` + gradient checkpointing produced truncated garbage → every episode `template_mismatch`. Fix: `_sample_peft_action` forces `eval()` for `generate`, then restores `train()`.
- Aggregate diagnostics: `python scripts/summarize_grpo_diagnostics.py results/rl_v2/<run>/step_diagnostics.jsonl`

```bash
# GPU study (Plan 11 CUDA env; refuses CPU default config)
python scripts/train_grpo.py --train --live-gpu \
  --config actproto/configs/grpo_v2_gpu_prefix.yaml \
  --output results/rl_v2/grpo_gpu_study_fix
```

## Live eval

- Legacy B0–B2: `python scripts/run_trial.py --run B0|B1|B2|all`
- Tool-use v2: `python scripts/run_experiment_v2.py …`
- `B2ordereval` / `B2fmteval` remain in `trial.yaml` as **diagnostic_train_echo**; CLIs moved to `scripts/archive/`.

## Ops notes

- Optional host UUID override: copy `actproto/configs/local.device.yaml.example` → `local.device.yaml` (gitignored)
- Colab / laptop GRPO diagnostics (not study evidence): [`COLAB_ROLLOUT_DIAGNOSTICS.md`](COLAB_ROLLOUT_DIAGNOSTICS.md), `scripts/diagnose_grpo_rollouts.py`
- Snapshot: [`RESULTS_2026-11-14.md`](RESULTS_2026-11-14.md)

## Docs map

| File | Role |
| --- | --- |
| `STATUS.md` | Current handoff (this file) |
| `RESULTS_2026-11-14.md` | Committed numbers for gitignored `results/**` runs |
| `EXPERIMENT_STATUS.md` | Historical run log |
| `AUDIT_2026-11-04.md` | Historical compliance audit |
| `plan_*` | Historical sprint plans |
| `archive/` | Moved brainstorm / clutter |
| `.opencode/audits/` | Security + cleanliness audit reports |
