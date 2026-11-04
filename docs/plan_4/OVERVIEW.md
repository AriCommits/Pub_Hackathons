# Plan 4: Query-argument log and the training decision

## Summary

`B2d1r` scores 68/104, and the 36 misses are `query_database` tasks that finish `0`, `NA`, or `None`. The saved eval stores the tool name and the final answer, not the arguments or the tool JSON, so it cannot tell a blank `year` from a wrong `year`. This plan saves those fields on a new run, `B2d1log`, classifies each mismatched query call, and writes the decision: fill the blank fields from the existing regex, or train an assistant-only adapter later. This plan does not train.

## Goals

- Every scored episode row includes `predicted_calls`, `gold_calls`, `tool_results`, and `prompt`.
- `B2d1log` repeats the `B2d1r` episode (route hop 1, model on hop 2, two steps) and writes `results/b0_b2/B2d1log/`. `B2d1r` stays as it is.
- `docs/plan_4/query_arg_report.md` lists each mismatched `query_database` call and counts `missing_field` against `wrong_value`.
- The count picks one next experiment. A majority of `wrong_value` means a later assistant-only retrain. Otherwise the next experiment fills blank fields from `heuristic_args`. The chosen sentence is appended once to `docs/EXPERIMENT_STATUS.md`.

## Non-Goals

- Training any adapter, including assistant-only loss, more epochs, or focal loss.
- Filling blank arguments in the live router. That waits on the report.
- A new chained-router run, hop-2 prototypes, or a third agent step.
- Overwriting `results/adapters/b1`, `results/b0_b2/B2d1r`, or `results/b0_b2/{B0,B1,B2,B2d1}`.

## Background / Context

Project root is `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter is `.venv\Scripts\python`. The eval GPU is `ACTPROTO_CUDA_DEVICES=1`, one replica. `score_episode` in `actproto/evaluation/metrics.py` already parses calls to compute `first_argument_correctness`, then drops the arguments when it writes the row. `heuristic_args` in `actproto/agent/schema_regen.py` recovers `product` and `year` from the user sentence with a regex. `B2d1r` first-tool accuracy is 1.000 and first-argument correctness is 0.625. The 39 first-argument misses are all `query_database`.

## Features / Tasks

### L1: Keep arguments and tool JSON on the eval row

**Files:** `actproto/evaluation/metrics.py` (modify `score_episode`), `tests/test_evaluation.py` (modify)
**Complexity:** S
**Depends on:** none

Add `predicted_calls`, `gold_calls`, `tool_results`, and `prompt` to the dict `score_episode` returns. Calls keep `name` and `arguments`. `tool_results` is the parsed JSON of each `role: tool` message, in order. `prompt` is `case.task`. Existing numeric metrics stay. Averages in `EvalResult.as_dict` stay limited to the current numeric keys, so the new lists are not averaged. The test uses the existing inventory fixture and checks that the successful row stores `{"record_id": "REC-001"}` and the tool content. No model load.

### Q1: Classify mismatched query arguments

**Files:** `scripts/query_arg_report.py` (new), `tests/test_query_arg_report.py` (new)
**Complexity:** S
**Depends on:** L1

The script reads one `eval.json`. A row is in scope when `gold_calls[0].name` is `query_database` and `first_argument_correctness` is 0. For that first call, compare predicted arguments to gold arguments. A gold key absent, null, or `""` is `missing_field` when `heuristic_args("query_database", prompt)` contains the gold value for that key. A blank extra argument, such as `region: ""`, is also `missing_field`: the tool treats that empty string as a filter. A present gold value that differs, or a non-blank extra argument, is `wrong_value`. A row with any `wrong_value` key is `wrong_value`. A row with only missing keys is `missing_field`.

Write `docs/plan_4/query_arg_report.md` with the two counts, one line per row (`case_id`, class, predicted arguments, gold arguments, tool result), and one decision sentence. If `wrong_value` is strictly greater than `missing_field`, the sentence says the next experiment is an assistant-only adapter and this plan does not train it. Otherwise the sentence says the next experiment fills blank query fields from the regex and this plan does not fill them.

The test classifies two inline rows, one missing `year` that the regex recovers and one with `year: 1999`. It does not read `results/` and does not load Tiny Aya.

### E1: Run B2d1log and append the decision

**Files:** `actproto/configs/trial.yaml` (modify), `scripts/run_trial.py` (modify choices), `tests/test_run_trial.py` (modify), `docs/EXPERIMENT_STATUS.md` (append one pointer)
**Complexity:** M
**Depends on:** L1, Q1

Add `B2d1log` with the same adapter, prototypes, threshold, `route_first_hop_only: true`, `max_agent_steps: 2`, and `replicas: 1` as `B2d1r`. Description: argument log of the depth-1 retry. Add it to the `--run` choices. `--run all` stays B0, B1, B2.

After L1 and Q1 are green, run from `lexsi/`:

```text
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2d1log
python scripts/query_arg_report.py
```

Unset `ACTPROTO_CUDA_DEVICES` when the processes exit. Do not pass `--run all`. Append one pointer to `docs/EXPERIMENT_STATUS.md` with the two counts and the decision sentence. Do not rewrite that file.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | Logging and the report use the installed stack. | |

## File Change Summary

| File | Action (new/modify/delete) |
| --- | --- |
| `actproto/evaluation/metrics.py` | modify |
| `tests/test_evaluation.py` | modify |
| `scripts/query_arg_report.py` | new |
| `tests/test_query_arg_report.py` | new |
| `actproto/configs/trial.yaml` | modify |
| `scripts/run_trial.py` | modify |
| `tests/test_run_trial.py` | modify |
| `docs/plan_4/query_arg_report.md` | new |
| `docs/EXPERIMENT_STATUS.md` | modify |

## Open Questions

- If `wrong_value` and `missing_field` tie, the report chooses the fill-blanks experiment. A tie is not enough evidence to spend a training run.
- Tool JSON that fails to parse is stored as `{"raw": "<text>"}` and does not change the argument class.
