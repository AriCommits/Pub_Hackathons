# ActProto-LoRA

Activation-prototype tool routing for **Tiny Aya Global**, built for the Lexsi AI Safety and Alignment Hackathon (Track 3: Agentic Fine-tuning).

The system reads residual-stream activations immediately before tool choice, builds a cosine prototype per tool, and routes the call only when a validation-chosen confidence threshold is met. Arguments are regenerated against the selected tool’s JSON schema. Tool names are never swapped inside an already-generated call.

Ordinary LoRA tool-use fine-tuning is the baseline. A linear probe on the same activations is the representation baseline. The headline metric is **verified task success**, not claimed success.

## Setup

Python 3.10+. The model repo `CohereLabs/tiny-aya-global` is gated.

```bash
cd lexsi
python -m pip install -e ".[dev]"
# optional, for live infer / training:
python -m pip install -e ".[train]"
huggingface-cli login   # HF_TOKEN for Tiny Aya
```

## Reproduce B0–B2

```bash
python scripts/generate_dataset.py
python scripts/train_lora.py --dry-run          # drop --dry-run on GPU
python scripts/extract_activations.py --dry-run
python scripts/build_prototypes.py --dump results/activations
python scripts/train_probe.py --dump results/activations
python scripts/run_b0.py --dry-run --limit 20
python scripts/run_b1.py --dry-run --limit 20
python scripts/run_b2.py --dry-run --limit 20
python scripts/compare_results.py --dry-run
```

Live runs write tables under `results/b0_b2/`. `--dry-run` gold-replays tools so CI can run without weights.

## Layout

```
actproto/
  configs/ tools/ data/ agent/ activations/ training/ evaluation/ steering/ utils/
scripts/          # experiment CLIs
tests/
docs/METHOD.md
docs/SUBMISSION.md
```

## Tests

```bash
python -m pytest
```

## Hugging Face export

```bash
python scripts/export_hf_artifacts.py --dry-run
```

The exporter refuses `.env` / token files.
