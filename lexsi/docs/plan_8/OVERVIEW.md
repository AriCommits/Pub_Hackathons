# Plan 8: Several prompt formats for the same tool orders

## Summary

`B2ordereval` routed every script to `query_database` because each prompt was one list of every tool, and hop 1 scores the last token of that list. This plan keeps that sequence wording and adds two more layouts for the same catalogs: the execution line first, and the execution line last. Each layout includes "only step 1," the full order, and "step N then step 1." A new adapter is trained on the original 288 rows plus two copies of every format. Prototypes for that adapter are fit on the original train prompts plus one copy of the format prompts, labeled by the first gold tool. The original 104-task test and one copy of every format are scored separately. `results/adapters/b1`, `results/adapters/b1_orders`, and `results/prototypes.npz` stay in place.

## Goals

- `order_tasks(repeats=4)` stays 152 sequence-only tasks. `order_tasks(1, formats=ALL_FORMATS)` returns 266 tasks covering `sequence`, `header_first`, and `header_last`, and the instructions `only_first`, `full`, and `last_then_first`.
- A `header_last` prompt ends with the execution line, and that line names the tools in execution order. A `header_first` prompt starts with that line and then lists every catalog step. The finish answer is still the `|`-joined values of the executed tools and is not written in the prompt.
- `train_lora.py --formats --plan-only` reports 820 rows and does not change the 440-row `--orders` result.
- `extract_activations.py --formats` appends `order_tasks(1, formats=ALL_FORMATS)` as train rows. `--split all` without the flag stays 496 tasks.
- Training writes `results/adapters/b1_formats`. The dump, npz, and metrics are `results/activations_b1_formats`, `results/prototypes_b1_formats.npz`, and `results/prototype_metrics_b1_formats.json`.
- `B2fmt` scores the original test with that pair, hop 1 only, 4 steps. `scripts/run_format_eval.py` scores the 266 format tasks as `B2fmteval` with 12 steps.
- One `docs/EXPERIMENT_STATUS.md` pointer records both verified-success numbers and first-tool accuracy on `B2fmteval` split by format.

## Non-Goals

- Overwriting `b1`, `b1_assistant_only`, `b1_orders`, `results/prototypes.npz`, or `results/prototypes_b1_orders.npz`.
- Focal loss or resampling from the product of tool rates.
- Changing verified task success.
- Editing `combination_tasks()` or the recorded `B2combo` gold answers.

## Background / Context

Project root is `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter is `.venv\Scripts\python`. Train pins the largest GPU inside `train_lora.py`. Eval uses `ACTPROTO_CUDA_DEVICES=1` and unsets it. Do not leave `CUDA_VISIBLE_DEVICES` set in the parent shell.

`default.yaml` sets `activation.token` to `prompt_last`. Hop 1 encodes the system prompt plus the user prompt and scores the last token. On `B2ordereval` that token belonged to a list naming every tool, and all 38 first calls were `query_database`. The original 104-task prompts name one job, and first-tool accuracy there was 0.990 with `b1_orders`.

## Features / Tasks

### F1: Format the same catalog three ways

**Files:** `actproto/data/order_curriculum.py` (modify), `tests/test_order_curriculum.py` (modify)
**Complexity:** M
**Depends on:** none

`ALL_FORMATS` is `sequence`, `header_first`, `header_last`. Default `formats` is `sequence` only.

For a catalog of work tools numbered from 1:

- `only_first` gold is step 1, then finish. Header: `Only execute step 1: {tool}.`
- `full` gold is the catalog order, then finish. Header names every step and then the tools in that order.
- `last_then_first` gold is the last catalog step, then step 1, then finish. Header: `Execute steps N and 1: {tool N}, then {tool 1}.`

`header_first` prints the header, then `Step k:` lines for the whole catalog. `header_last` prints the catalog, then the header. `sequence` keeps today's wording and today's ids (`order-12-0`). New ids include the format and the instruction. The catalog lists every step even when the header executes a subset.

### T1: Train and extract the format mix

**Files:** `scripts/train_lora.py` (modify), `scripts/extract_activations.py` (modify), `tests/test_train_lora.py` (modify), `tests/test_order_curriculum.py` (modify)
**Complexity:** S
**Depends on:** F1

`--formats` appends `order_tasks(2, formats=ALL_FORMATS)` (532 tasks) to the 288 train rows. `--orders` is unchanged. Extract `--formats` appends one copy of `ALL_FORMATS` onto `train` or `all`, marked `split=train`.

GPU, from `lexsi/`, with `CUDA_VISIBLE_DEVICES` unset:

```text
python scripts/train_lora.py --formats --output-dir results/adapters/b1_formats
python scripts/extract_activations.py --split all --formats --adapter results/adapters/b1_formats --out results/activations_b1_formats
python scripts/build_prototypes.py --dump results/activations_b1_formats --prototypes results/prototypes_b1_formats.npz --metrics results/prototype_metrics_b1_formats.json --eval-test
```

### E1: Score the original test and every format

**Files:** `actproto/configs/trial.yaml` (modify), `scripts/run_trial.py` (modify), `scripts/run_format_eval.py` (new), `tests/test_trial_config.py` (modify), `tests/test_run_trial.py` (modify), `README.md` (modify), `docs/EXPERIMENT_STATUS.md` (append one pointer)
**Complexity:** M
**Depends on:** T1

`B2fmt` and `B2fmteval` use `results/adapters/b1_formats`, `results/prototypes_b1_formats.npz`, and `results/prototype_metrics_b1_formats.json`. Both route hop 1 only, one replica. `B2fmt` allows 4 steps on the original test. `B2fmteval` allows 12 steps. `--run all` stays B0, B1, B2.

```text
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2fmt
python scripts/run_format_eval.py
```

Unset `ACTPROTO_CUDA_DEVICES` afterward. The pointer includes `B2fmt` verified success and `B2fmteval` verified success plus first-tool accuracy for `sequence`, `header_first`, and `header_last`.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | The installed stack covers training and eval. | |

## File Change Summary

| File | Action (new/modify/delete) |
| --- | --- |
| `actproto/data/order_curriculum.py` | modify |
| `tests/test_order_curriculum.py` | modify |
| `scripts/train_lora.py` | modify |
| `scripts/extract_activations.py` | modify |
| `tests/test_train_lora.py` | modify |
| `actproto/configs/trial.yaml` | modify |
| `scripts/run_trial.py` | modify |
| `scripts/run_format_eval.py` | new |
| `tests/test_trial_config.py` | modify |
| `tests/test_run_trial.py` | modify |
| `README.md` | modify |
| `docs/EXPERIMENT_STATUS.md` | modify |

## Open Questions

- `header_last` is the layout hop 1 can see, because the scored token is the last one. `header_first` is included because that is the other wording. The format split in the pointer says which one the router uses.
