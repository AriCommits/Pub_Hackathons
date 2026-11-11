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

### GRPO (three interpreters)

GRPO needs the supervised `.venv` plus two more interpreters (CPU AgentTune smoke; CUDA torch `2.11.0+cu128` / AgentTune 1.1.3 / TRL 1.7.1). Recipe: `requirements-rl-gpu.txt` / `requirements-rl-gpu.lock`. Details: [`docs/RL_GPU_ENV.md`](docs/RL_GPU_ENV.md), [`docs/STATUS.md`](docs/STATUS.md).

| Role | Config | Notes |
| --- | --- | --- |
| CPU / plan | `actproto/configs/grpo_v2.yaml` | `use_cpu: true`; default `--config` |
| GPU smoke | `actproto/configs/grpo_v2_gpu.yaml` | `--smoke --live-gpu`; 5060 Ti UUID only |
| GPU study | `actproto/configs/grpo_v2_gpu_prefix.yaml` | `--train --live-gpu` with this config; writes `results/rl_v2/grpo_gpu_study` |
| Historical | `actproto/configs/archive/*` | e2 / tools YAMLs; require `--allow-historical` |

`--train --live-gpu` refuses the CPU default config. Reward constants live in code; `actproto/configs/reference/reward_v2.yaml` is documentation only. Tool-reward historical dirs (`grpo_gpu_tools`, `grpo_gpu_prefix`, …) are protected. The 200/200 macro 1.0 is frozen SFT e2, not GRPO.

## Trial config (paths + B0–B2)

Edit **`actproto/configs/trial.yaml`** for artifact paths, eval split, and run definitions. See **`docs/TRIAL_CONFIG.md`** for the full map.

All scripts accept `--config path/to/trial.yaml` (default: `actproto/configs/trial.yaml`).

## Reproduce B0–B2

```bash
python scripts/generate_dataset.py
python scripts/train_lora.py --plan-only          # drop flag for GPU training
python scripts/extract_activations.py --plan-only # then run without flag after LoRA
python scripts/build_prototypes.py
python scripts/train_probe.py
python scripts/run_trial.py --run all             # B0, B1, B2 live eval
python scripts/compare_results.py
```

Reports land under `results/b0_b2/{B0,B1,B2}/` (override in `trial.yaml`).

## Follow-up runs

`generate_text` stops after one closed `<tool_call>` block, or at `<|END_OF_TURN_TOKEN|>`. `--run all` still executes only B0, B1, and B2.

```bash
python scripts/run_trial.py --run B2d1r    # depth-1 retry: route hop 1, then the model, 2 steps
python scripts/train_lora.py --output-dir results/adapters/b1_assistant_only
python scripts/run_trial.py --run B2d1a    # depth-1 retry on that new adapter
python scripts/run_trial.py --run B2d1log  # same episode as B2d1r, with arguments saved
python scripts/query_arg_report.py         # classify query_database misses; does not train
python scripts/run_trial.py --run B2chain  # route every hop from the transcript, 4 steps
python scripts/train_lora.py --orders --output-dir results/adapters/b1_orders
python scripts/run_trial.py --run B2orders      # original test, order-curriculum adapter, 4 steps
python scripts/run_order_eval.py                # one copy of each curriculum order, 12 steps
python scripts/train_lora.py --formats --output-dir results/adapters/b1_formats
python scripts/run_trial.py --run B2fmt        # original test, multi-format adapter, 4 steps
python scripts/run_format_eval.py              # sequence, header-first, and header-last prompts
```

`B2d1r` and `B2chain` use the existing B1 adapter and prototypes. They write `results/b0_b2/B2d1r/` and `results/b0_b2/B2chain/` and do not replace B0, B1, B2, or B2d1.

## Layout

```
actproto/
  configs/   # trial.yaml, model.yaml, default.yaml, sft.yaml
  tools/ data/ agent/ activations/ training/ evaluation/ steering/ utils/
scripts/     # run_trial.py, compare_results.py, …
tests/
docs/METHOD.md
docs/TRIAL_CONFIG.md
```

## Tests

```bash
python -m pytest
```

## Hugging Face export

```bash
python scripts/export_hf_artifacts.py --list-only
python scripts/export_hf_artifacts.py --repo your-org/actproto-lexsi
```

The exporter refuses `.env` / token files and uses relative paths from `trial.yaml` → `export`.
