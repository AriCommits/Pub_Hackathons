---
label: reproduction
track: T3
audience: operators
status: draft
---

# Reproduction

Python 3.10 or newer. Commands assume the working directory is `lexsi/`.

## Layout

| Path | Role |
| --- | --- |
| `actproto/` | Library: tools, data generator, training, evaluation, configs |
| `scripts/` | Command-line entry points, including the public-data curator |
| `tests/` | Unit tests. No GPU is required for the default suite. |
| `docs/` | Status, method, and the plan history |
| `submission/` | This packet: write-up, datacard, model card, public data slice |
| `results/` | Adapter directories that are tracked, and run logs that are gitignored |
| `data/benchmarks/` | Local full builds, gitignored, including the sealed test |

## Environment

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"    # GPU train and live inference
huggingface-cli login                  # the base model is gated
```

GPU training needs an NVIDIA card with 16 GB. The project notes use an RTX 5060 Ti. An 8 GB RTX 4060 is the wrong device for bf16 Tiny Aya. CPU torch is enough to run the unit tests.

GRPO uses a separate interpreter from the supervised install. The pin and the device check are in `docs/RL_GPU_ENV.md`. A smoke that finishes is not evidence of a better policy than e2.

## Dataset

The builder refuses to overwrite an existing directory. Pick a new path if `data/benchmarks/tooluse_v2` is already there.

```bash
python -m scripts.build_benchmark_v2 --output data/benchmarks/tooluse_v2 --seed 42
python -m scripts.curate_submission_dataset --build data/benchmarks/tooluse_v2 --output submission/data
```

The second command writes training messages and validation prompts only.

## Supervised training

```bash
python scripts/train_sft_v2.py --help
```

The frozen artifact is `results/sft_v2/e2/checkpoint-150`. Do not train into `results/adapters/b1`. That path is a legacy adapter. The default curriculum LoRA output is `results/adapters/lora_run`.

## Validation

Score the validation split only. Do not pass `--final-eval`, and do not point the run at the test split. The sealed test is not part of the public reproduce path.

```bash
python scripts/run_experiment_v2.py --help
```

The primary contract is router off and greedy decoding, in `actproto/configs/sft_matrix.yaml`.

## GRPO

```bash
python scripts/train_grpo.py --help
```

Live configs are `actproto/configs/grpo_v2.yaml` (CPU plan), `actproto/configs/grpo_v2_gpu.yaml` (GPU smoke), and `actproto/configs/grpo_v2_gpu_prefix.yaml` (study). Historical configs under `actproto/configs/archive/` need `--allow-historical`. Write a new directory under `results/rl_v2/`. The selected policy remains e2.

## Tests

```bash
python -m pytest
```
