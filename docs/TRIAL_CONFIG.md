# Trial configuration

All experiment paths, eval split/limit, hypothesis gates, and the B0–B2 run matrix live in one file:

`actproto/configs/trial.yaml`

Python loads it via `actproto.configs.trial_schema.load_trial()` (used by every `scripts/*.py` CLI through `--config`).

## What to edit for a new trial

| Section | Purpose |
| --- | --- |
| `paths.*` | Where artifacts go (activations dump, adapters, prototypes, reports, processed data). |
| `eval.split` / `eval.limit` | Which tasks `run_trial.py` evaluates (`test` by default). |
| `hypothesis.*` | Gates for `compare_results.py` (correctness drop pp, latency ratio). |
| `prototype_fit.min_coverage` | Validation threshold search in `build_prototypes.py`. |
| `runs.B0/B1/B2` | Adapter path, `router: true/false`, prototype/threshold files for B2. |
| `export.*` | Paths scanned by `export_hf_artifacts.py`. |

Paths are **relative to the lexsi project root** (the directory that contains `actproto/` and `scripts/`).

Model dtype, activation layer/token, and `model_id` stay in `default.yaml`, `model.yaml`, and optional `layer.yaml` so weight settings stay separate from trial layout.

## Command flow (MVP)

```bash
cd lexsi
# 1. Edit trial.yaml if needed
python scripts/generate_dataset.py
python scripts/train_lora.py --plan-only          # or drop flag on GPU
python scripts/extract_activations.py --plan-only # then run without flag
python scripts/build_prototypes.py
python scripts/train_probe.py
python scripts/run_trial.py --run all             # needs HF_TOKEN + GPU
python scripts/compare_results.py
```

## Overrides

Pass a copy of the YAML anywhere:

```bash
python scripts/run_trial.py --run B2 --config my_trials/ablation_01.yaml
```

Implement copies by duplicating `trial.yaml` and changing only `paths` or `runs` — no script flags required for each path.

## Stretch work (post-MVP)

B3 steering and B4 alignment/GRPO remain under `actproto/steering/` and `actproto/training/` but no longer have stub `run_b3.py` / `run_b4.py` entrypoints. Add new `runs.*` entries and extend `run_trial.py` when those systems are ready for eval.
