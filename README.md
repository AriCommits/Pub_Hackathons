# ActProto-LoRA

Activation-prototype tool routing for **Tiny Aya Global**, built for the Lexsi AI Safety and Alignment Hackathon (Track 3: Agentic Fine-tuning).

The system reads residual-stream activations immediately before tool choice, builds a cosine prototype per tool, and routes the call only when a validation-chosen confidence threshold is met. Arguments are regenerated against the selected tool’s JSON schema. Tool names are never swapped inside an already-generated call.

Ordinary LoRA tool-use fine-tuning is the baseline. A linear probe on the same activations is the representation baseline. The headline metric is **verified task success**, not claimed success.

Current handoff: [`docs/STATUS.md`](docs/STATUS.md). Historical B0–B2 / SFT / GRPO run log: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).

## Setup

Python 3.10+. The model repo `CohereLabs/tiny-aya-global` is gated.

```bash
cd lexsi
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"   # GPU train / live infer
huggingface-cli login                   # HF_TOKEN for Tiny Aya
```

## Produce LoRA adapters

Preferred (frozen SFT-v2 e2 path):

```bash
python scripts/train_sft_v2.py --help
python scripts/run_sft_study.py         # matrix study when needed
```

Legacy curriculum LoRA (B0–B2 stack):

```bash
python scripts/generate_dataset.py
python scripts/train_lora.py --plan-only          # drop flag for GPU training
# default output is results/adapters/lora_run — do not train into b1 by default
```

Export for submission:

```bash
python scripts/export_hf_artifacts.py --list-only
python scripts/export_hf_artifacts.py --repo your-org/actproto-lexsi
```

## Live eval

```bash
python scripts/run_trial.py --run all             # legacy B0, B1, B2
python scripts/run_experiment_v2.py --help        # tool-use v2
python scripts/compare_results.py
```

Supporting B0–B2 pipeline: `extract_activations.py` → `build_prototypes.py` → `train_probe.py`. Extra historical B2d1 / order / format variants live in [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md); contaminated train-echo CLIs are under `scripts/archive/`.

## GRPO (optional RL)

Needs the Plan 11 CUDA env (see [`docs/RL_GPU_ENV.md`](docs/RL_GPU_ENV.md)). Live entry: `scripts/train_grpo.py`.

| Role | Config | Notes |
| --- | --- | --- |
| CPU / plan | `actproto/configs/grpo_v2.yaml` | default `--config` |
| GPU smoke | `actproto/configs/grpo_v2_gpu.yaml` | `--smoke --live-gpu` |
| GPU study | `actproto/configs/grpo_v2_gpu_prefix.yaml` | `--train --live-gpu`; new `--output` under `results/rl_v2/` |
| Historical | `actproto/configs/archive/*` | require `--allow-historical` |

## Layout

```
actproto/
  configs/ training/ evaluation/ agent/ activations/ tools/ data/ …
  stretch/   # research-only (steering, alignment loss, legacy rewards)
scripts/     # live CLIs; scripts/archive/ for stubs & train-echo diagnostics
tests/
docs/STATUS.md
```

## Tests

```bash
python -m pytest
```

Trial paths: edit **`actproto/configs/trial.yaml`** ([`docs/TRIAL_CONFIG.md`](docs/TRIAL_CONFIG.md)).
