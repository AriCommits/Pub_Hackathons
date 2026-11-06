# ActProto-LoRA

Activation-prototype tool routing for **Tiny Aya Global**, built for the Lexsi AI Safety and Alignment Hackathon (Track 3: Agentic Fine-tuning).

The system reads residual-stream activations immediately before tool choice, builds a cosine prototype per tool, and routes the call only when a validation-chosen confidence threshold is met. Arguments are regenerated against the selected tool’s JSON schema. Tool names are never swapped inside an already-generated call.

Ordinary LoRA tool-use fine-tuning is the baseline. A linear probe on the same activations is the representation baseline. The headline metric is **verified task success**, not claimed success.

Live B0–B2 progress, environment notes, and the resume checklist are in [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md). Update that file as the run proceeds.

## Setup

Python 3.10+. The model repo `CohereLabs/tiny-aya-global` is gated.

```bash
cd lexsi
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"   # GPU train / live infer
huggingface-cli login                   # HF_TOKEN for Tiny Aya
```

GRPO uses two more interpreters, both outside this supervised `.venv`. The CPU AgentTune environment runs the already-verified mechanics smoke. A separate CUDA environment (torch `2.11.0+cu128`, AgentTune 1.1.3, TRL 1.7.1) is the one for new GPU GRPO work. The recipe is `requirements-rl-gpu.txt`; the freeze is `requirements-rl-gpu.lock`. Paths and the device-print command are in [`docs/RL_GPU_ENV.md`](docs/RL_GPU_ENV.md). `actproto/configs/grpo_v2.yaml` stays the CPU profile (`use_cpu: true`). `actproto/configs/grpo_v2_gpu.yaml` is the only profile that may set `use_cpu: false`, and only for the RTX 5060 Ti UUID. `create_trainer` accepts that flag only with `gpu_profile="grpo_v2_gpu"` and still rejects vLLM. `scripts/check_rl_gpu_env.py`, run in the CUDA environment with `CUDA_VISIBLE_DEVICES` unset, selects the 5060 Ti by UUID. `scripts/train_grpo.py --smoke --live-gpu` is the gated CUDA fixture smoke and uses `actproto/configs/grpo_v2_gpu.yaml`. Without `--live-gpu` that profile does not train. The tool-reward study config is `actproto/configs/grpo_v2_gpu_tools.yaml` (`results/rl_v2/grpo_gpu_tools`, temperature 0.8, top-p 0.95, two generations, beta 0, learning rate `5e-6`, vLLM off). The study sampler passes the named `default` chat template and allows 256 new tokens on each sampled turn; `max_completion_length` stays the loss denominator. `actproto/configs/reward_v2.yaml` documents `bounded_prefix_v1` (failure-only prefix credit, ceiling 0.30) and is not a live override. The trainer wrapper scores `bounded_prefix_v1`. `score_executed_session` stays on `bounded_shaped_v1`. The 20-step tool-reward runs are negative results. The 200/200 macro 1.0 is frozen SFT e2, not GRPO. `grpo_v2_gpu_e2.yaml` stays the record of the earlier 20-step run. `--train --live-gpu` on a `grpo_v2_gpu` config is the masked-child Tiny Aya run. It executes train-task tools in process, refuses the RTX 4060, and must write a new directory under `results/rl_v2` rather than `grpo_gpu_e2`, `grpo_gpu_smoke`, or `grpo_cpu_smoke`. The CPU profile still does not optimize. `scripts/measure_rl_gpu_memory.py` loads Tiny Aya bf16 on the 5060 Ti and writes `results/rl_v2/gpu_memory/tiny_aya_bf16.json`. It does not train or attach an adapter. Do not point that environment at `C:\Python314\python.exe` or reinstall the supervised venv.

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
