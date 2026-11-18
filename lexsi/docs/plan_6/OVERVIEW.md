# Plan 6: Prototypes for the assistant-only adapter

## Summary

`B2d1a` scored 36/104 because the new adapter's activations were compared with prototypes fit on `results/adapters/b1`. This plan extracts a new activation dump with `results/adapters/b1_assistant_only` loaded, fits a new prototype bank on the train split, and reruns the depth-1 episode with that adapter and that bank together. `results/prototypes.npz`, `results/prototype_metrics.json`, `results/activations`, and `results/adapters/b1` stay untouched.

## Goals

- `scripts/extract_activations.py` accepts `--adapter` and `--out`. The defaults remain the b1 adapter and `results/activations`.
- `scripts/build_prototypes.py` accepts `--dump`, `--prototypes`, and `--metrics`. The defaults remain `results/prototypes.npz` and `results/prototype_metrics.json`.
- A dump at `results/activations_b1a` is produced with `--split all` so train, val, and test rows are present. Prototypes are fit on train only. The threshold is chosen on val only.
- `results/prototypes_b1a.npz` and `results/prototype_metrics_b1a.json` are written. The metrics file includes `test_accuracy` from the test rows in the new dump.
- `B2d1ap` uses the assistant-only adapter, the new npz, and the new threshold, with the `B2d1r` episode shape. It writes `results/b0_b2/B2d1ap/`.
- One pointer in `docs/EXPERIMENT_STATUS.md` records prototype test accuracy, verified task success, and first-tool accuracy. First-tool accuracy is compared with 1.000.

## Non-Goals

- Overwriting `results/adapters/b1`, `results/adapters/b1_assistant_only`, `results/activations`, `results/prototypes.npz`, or `results/prototype_metrics.json`.
- Another training run, focal loss, or a change to `query_database` blank filters.
- Hop-2 or chained-router prototypes. The dump is still one vector per task, labeled with the first gold tool.
- Re-running B0, B1, B2, B2d1, B2d1r, B2d1log, B2d1a, or B2chain.

## Background / Context

Project root is `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter is `.venv\Scripts\python`. `load_model` picks the CUDA device with the most memory. Do not leave `CUDA_VISIBLE_DEVICES` set in the parent shell.

`extract_activations.py` always sets `config.adapter_path` from `trial.paths.adapter_b1` and writes `trial.paths.activations_dump`. `build_prototypes.py` always reads that dump and writes `prototypes.npz` plus `prototype_metrics.json` in `prototypes_dir`. Those are the files `B2d1r` still routes with. The assistant-only train is already done: 2 epochs, train loss 0.01047, masked token accuracy 1.0, output `results/adapters/b1_assistant_only`.

## Features / Tasks

### P1: Separate dump and prototype paths

**Files:** `scripts/extract_activations.py` (modify), `scripts/build_prototypes.py` (modify), `tests/test_activation_paths.py` (new)
**Complexity:** S
**Depends on:** none

`--adapter` overrides the adapter directory. `--out` overrides the dump directory. `--dump` overrides the dump read by the prototype fit. `--prototypes` is the npz path. `--metrics` is the json path. Omitted flags keep today's paths.

The test calls `extract_activations.main` with `--plan-only`, `--adapter`, and `--out`, and checks the printed paths. It builds a tiny dump with `write_dump` under `tmp_path`, runs `build_prototypes.main` with `--dump`, `--prototypes`, and `--metrics` pointed at that temp directory, and asserts `results/prototypes.npz` is not created by that call. No Tiny Aya load.

### E1: Fit the new bank and run B2d1ap

**Files:** `actproto/configs/trial.yaml` (modify), `scripts/run_trial.py` (modify), `tests/test_run_trial.py` (modify), `tests/test_trial_config.py` (modify), `docs/EXPERIMENT_STATUS.md` (append one pointer)
**Complexity:** M
**Depends on:** P1

Add `B2d1ap`: adapter `results/adapters/b1_assistant_only`, prototypes `results/prototypes_b1a.npz`, threshold `results/prototype_metrics_b1a.json`, `route_first_hop_only: true`, `max_agent_steps: 2`, `replicas: 1`. `--run all` stays B0, B1, B2.

After P1 is green, from `lexsi/`, with `CUDA_VISIBLE_DEVICES` unset:

```text
python scripts/extract_activations.py --split all --adapter results/adapters/b1_assistant_only --out results/activations_b1a
python scripts/build_prototypes.py --dump results/activations_b1a --prototypes results/prototypes_b1a.npz --metrics results/prototype_metrics_b1a.json --eval-test
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2d1ap
```

Unset `ACTPROTO_CUDA_DEVICES` after the eval. Append one pointer with prototype `test_accuracy`, verified task success, first-tool accuracy, and first-argument correctness. If first-tool accuracy is below the saved B2 rate of 1.000, say so and do not copy the new npz over `results/prototypes.npz`.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | Extraction and the fit use the installed stack. | |

## File Change Summary

| File | Action (new/modify/delete) |
| --- | --- |
| `scripts/extract_activations.py` | modify |
| `scripts/build_prototypes.py` | modify |
| `tests/test_activation_paths.py` | new |
| `actproto/configs/trial.yaml` | modify |
| `scripts/run_trial.py` | modify |
| `tests/test_run_trial.py` | modify |
| `tests/test_trial_config.py` | modify |
| `docs/EXPERIMENT_STATUS.md` | modify |

## Open Questions

- Prototype `test_accuracy` on this dump is first-tool accuracy of the cosine bank, not verified task success. The live `B2d1ap` number is the one that decides whether the pair replaces `b1` for later routing. This plan does not perform that replacement either way.
