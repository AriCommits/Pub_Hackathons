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
