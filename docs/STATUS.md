# Lexsi ActProto — current status

Canonical handoff. Prefer this file over older plan docs and the long `EXPERIMENT_STATUS.md` log.

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
- `B2ordereval` / `B2fmteval` are **diagnostic_train_echo** (prompts overlap curriculum training). Not held-out generalization.

## Active remediation

- [`plan_14/OVERVIEW.md`](plan_14/OVERVIEW.md) — security + cleanliness audit fixes (C1–C7 implemented on the audit branch)
- Optional host UUID override: copy `actproto/configs/local.device.yaml.example` → `local.device.yaml` (gitignored)
- Colab / laptop GRPO diagnostics (not study evidence): [`COLAB_ROLLOUT_DIAGNOSTICS.md`](COLAB_ROLLOUT_DIAGNOSTICS.md), `scripts/diagnose_grpo_rollouts.py`, `notebooks/colab_grpo_rollout_diagnostics.ipynb`. Use `--allow-any-gpu` / `LEXSI_ALLOW_ANY_GPU=1` only for dumps; UUID pin fails closed otherwise. For &lt;2 min laptop time, use the **Unattended one-shot** section (Drive mount + `live_rollouts_summary.json` Counter).

## Docs map

| File | Role |
| --- | --- |
| `STATUS.md` | Current handoff (this file) |
| `EXPERIMENT_STATUS.md` | Historical run log; may contain stale lead warnings |
| `AUDIT_2026-11-04.md` | Historical compliance audit (GRPO-stub claim is obsolete) |
| `plan_*` | Historical sprint plans |
| `.opencode/audits/` | Security + cleanliness audit reports |
