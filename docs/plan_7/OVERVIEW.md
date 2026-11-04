# Plan 7: Order curriculum for a new adapter

## Summary

The train split contains six gold sequences, and the only 3-call sequence is `query_database` then `calculator` then `finish`. This plan builds an extra curriculum where every 2-tool and 3-tool order appears four times, plus four copies of the each-tool-twice chain and four copies of the 12-call chain. Those rows are mixed with the original 288 train tasks. Assistant-only SFT writes `results/adapters/b1_orders`. Prototypes are fit on the original train prompts with that adapter loaded, into new files. Two evals follow: the original 104-task test with a 4-step cap, and one copy of each curriculum order with a 12-step cap. `results/adapters/b1`, `results/adapters/b1_assistant_only`, and `results/prototypes.npz` stay in place. Verified task success stays the headline metric.

## Goals

- `order_tasks(repeats=4)` returns 152 tasks. Every permutation of two and three work tools appears four times. `order-twice-*` and `order-12-*` appear four times each. The finish answer is the `|`-joined tool values and is not written in the prompt.
- `train_lora.py --orders --plan-only` reports 440 rows. Without `--orders`, the row count stays 288 and the default output directory stays `results/adapters/b1`.
- Training with `--orders --output-dir results/adapters/b1_orders` finishes assistant-only. It does not write `results/adapters/b1`.
- `results/activations_b1_orders`, `results/prototypes_b1_orders.npz`, and `results/prototype_metrics_b1_orders.json` are produced from the original train and val prompts. `results/prototypes.npz` is unchanged.
- `B2orders` uses the new adapter and new prototypes, routes hop 1 only, and allows 4 steps on the original test split.
- `scripts/run_order_eval.py` scores `order_tasks(repeats=1)` (38 tasks) as `B2ordereval` with a 12-step cap.
- One pointer in `docs/EXPERIMENT_STATUS.md` records both verified-success numbers and the tool-name match on the 38 order tasks.

## Non-Goals

- Focal loss, class-weighted loss, or resampling from the product of current tool rates.
- Overwriting `results/adapters/b1`, `results/adapters/b1_assistant_only`, `results/prototypes.npz`, `results/prototype_metrics.json`, or `results/activations`.
- Changing `combination_tasks()` or the recorded `B2combo` gold answers.
- Changing verified task success, tool-call F1, or argument correctness.
- Fitting prototypes on the curriculum prompts. Hop 1 of the original test is still a single-tool prompt.

## Background / Context

Project root is `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter is `.venv\Scripts\python`. Train pins the largest GPU inside `train_lora.py`. Eval uses `ACTPROTO_CUDA_DEVICES=1` and unsets it afterward. Do not leave `CUDA_VISIBLE_DEVICES` set in the parent shell.

Train sequences today: `calculator, finish` 96; `query_database, finish` 93; `lookup_record, finish` 51; `finish` 27; `query_database, calculator, finish` 12; `search_docs, finish` 9. `B2apchain` scored 104/104 on that distribution. `B2combo` matched gold tool names on 11/38 scripted orders, and the 12-call task stopped after two calls. Five tool names cannot fill 12 calls at two uses each, so the 12-call sequence uses calculator, search_docs, and query_database three times.

`gold_to_sft_row` already replays tools into tool turns and masks loss onto assistant calls when the chat template carries generation markers. `run_lora_sft` refuses full-sequence loss.

## Features / Tasks

### C1: Curriculum tasks

**Files:** `actproto/data/order_curriculum.py` (new), `tests/test_order_curriculum.py` (new)
**Complexity:** M
**Depends on:** none

Work tools are calculator, search_docs, query_database, and lookup_record. Arguments rotate through three valid variants. Each task prompt lists the calls in order and says to finish with the tool values joined by `|`. The finish string is computed by running the tools: calculator value, first search hit id or `none`, the sales figure when one row matches otherwise the row count, and the lookup price. `order_tasks(1)` has 38 tasks. `order_tasks(4)` has 152. The 12-call names match the `B2combo` 12-call order. Original `generate_tasks()` families do not include `order_curriculum`.

### T1: Train on the mix

**Files:** `scripts/train_lora.py` (modify), `tests/test_train_lora.py` (modify)
**Complexity:** S
**Depends on:** C1

`--orders` appends `order_tasks(4)` to the train split. `--output-dir` still wins. The plan-only payload's `n_rows` is 440 when `--orders` is set and 288 otherwise. The GPU command is `python scripts/train_lora.py --orders --output-dir results/adapters/b1_orders`.

### E1: Prototypes and the two evals

**Files:** `actproto/configs/trial.yaml` (modify), `scripts/run_trial.py` (modify), `scripts/run_order_eval.py` (new), `tests/test_trial_config.py` (modify), `tests/test_run_trial.py` (modify), `README.md` (modify), `docs/EXPERIMENT_STATUS.md` (append one pointer)
**Complexity:** M
**Depends on:** T1

Add `B2orders`: adapter `results/adapters/b1_orders`, prototypes `results/prototypes_b1_orders.npz`, threshold `results/prototype_metrics_b1_orders.json`, `route_first_hop_only: true`, `max_agent_steps: 4`, `replicas: 1`. `--run all` stays B0, B1, B2.

`run_order_eval.py` calls `run_live_trial("B2ordereval", trial, order_tasks(1))`. Add `B2ordereval` with the same adapter and prototypes, `max_agent_steps: 12`.

After training, from `lexsi/`, with `CUDA_VISIBLE_DEVICES` unset:

```text
python scripts/extract_activations.py --split all --adapter results/adapters/b1_orders --out results/activations_b1_orders
python scripts/build_prototypes.py --dump results/activations_b1_orders --prototypes results/prototypes_b1_orders.npz --metrics results/prototype_metrics_b1_orders.json --eval-test
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2orders
python scripts/run_order_eval.py
```

Unset `ACTPROTO_CUDA_DEVICES` after the second eval. The extract uses the original generated tasks, not the curriculum. Append one pointer with `B2orders` verified success and `B2ordereval` verified success plus tool-name matches.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | Training uses the installed stack. | |

## File Change Summary

| File | Action (new/modify/delete) |
| --- | --- |
| `actproto/data/order_curriculum.py` | new |
| `tests/test_order_curriculum.py` | new |
| `scripts/train_lora.py` | modify |
| `tests/test_train_lora.py` | modify |
| `actproto/configs/trial.yaml` | modify |
| `scripts/run_trial.py` | modify |
| `scripts/run_order_eval.py` | new |
| `tests/test_trial_config.py` | modify |
| `tests/test_run_trial.py` | modify |
| `README.md` | modify |
| `docs/EXPERIMENT_STATUS.md` | modify |

## Open Questions

- Training on every short order makes `B2ordereval` an in-distribution check, not a held-out set. The original 104-task run is the regression check. This plan does not hold out a permutation.
