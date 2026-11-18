# Plan 3: One-call stop, depth-1 retry, chained routing

## Summary

`B2d1` already names the right first tool on all 104 test tasks. Verified success is 0/104 because 95 finishes are `PARSE_ERROR: multiple tool calls in one turn` and the other 9 are `finish` with an empty answer (Stockholm, Dublin, Athens). This plan stops generation after one closed tool call, allows one extra finish generation when that answer is empty, then runs two new evals on the 5060 Ti. `B2d1r` repeats the depth-1 episode. `B2chain` leaves the router on after the tool result is back in the transcript. Nothing overwrites `B0`, `B1`, `B2`, or `B2d1`.

## Goals

- `generate_text` stops once the new tokens contain one closed `<tool_call>...</tool_call>` or `<|END_OF_TURN_TOKEN|>`. `parse_tool_call` still rejects two calls in one string.
- When the router has selected `finish` and the first generation has an empty answer, a parse error, or a different tool name, generate once more. A non-empty valid `finish` answer from that second call is kept. Any other tool still gets one generation.
- Run `B2d1r`: same shape as `B2d1` (route hop 1 only, inner model on hop 2, at most 2 steps), new result directory.
- Run `B2chain`: route every hop from the current transcript, at most 4 steps, new result directory. Hop 2 is whatever the new vector ranks first after the tool command and the tool result are both in context.

## Non-Goals

- A second or third copy of the tool call pasted into the result text. The assistant turn is already the command, and the following user turn is already `Tool result:\n...`. That pair is the context. Do not duplicate it.
- A used-tool mask or a static top-k list.
- Another LoRA run, assistant-only loss, or a new adapter. That work stays in `docs/plan_1` (L1).
- The `429.0` → `429` finish rewrite and the prose-to-field snap. Those stay in `docs/plan_1` (R1). If `B2d1r` shows them, report the count. Do not add the rewrite in this plan.
- Overwriting `results/adapters/b1`, `results/prototypes.npz`, `results/prototype_metrics.json`, or `results/b0_b2/{B0,B1,B2,B2d1}`.
- Re-running B0, B1, or B2.
- Replacing `parse_tool_call`. Two tool calls in one string stay a `ParseError`.
- Building hop-2 prototypes. `B2chain` uses the existing first-tool prototypes and reports whatever they do.

## Background / Context

Project root is `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter is `.venv\Scripts\python`. GPU for both evals: `ACTPROTO_CUDA_DEVICES=1` (RTX 5060 Ti), one replica. Do not start either eval while another `run_trial.py` is using the GPUs. Do not leave `CUDA_VISIBLE_DEVICES` set in the parent shell.

`docs/plan_2` (branch `wip/pre-plan-2-master`) is the depth-1 plan already implemented on `master` through `e147ea8`. This directory is plan 3 because a checkout of that branch already owns `docs/plan_2/`. The on-disk scanner only sees `docs/plan_1/` and would have printed `docs/plan_2`.

What the finished runs scored:

| Run | Verified success | First tool | First arguments | Steps | Repeats | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | 1/104 | 1.000 | 0.894 | 5.57 | 4.57 | 97.7 s |
| B2d1 | 0/104 | 1.000 | 0.885 | 1.91 | 0 | 31.8 s |

`B2d1` failure reasons: 95 `answer PARSE_ERROR: multiple tool calls in one turn`, and 9 empty `finish` answers on the capital-city tasks. 98/104 already had the gold tool-name sequence. `finish` accepts `""` because `validate_args` only checks that `answer` is a string.

How context is updated today, and how it stays:

`append_tool_turn` in `actproto/agent/trajectory.py` appends two messages after every executed call: an assistant message whose `tool_calls` entry is the name and arguments, and a `tool` message whose content is the JSON result. `messages_for_template` in `actproto/agent/prompts.py` renders that pair for the Cohere template, which has no tool role. The assistant turn becomes one `<tool_call>...</tool_call>` string. The tool turn becomes a user turn `Tool result:\n` plus the JSON. `generate_text` and, after step 0, `make_vector_fn` both call `messages_for_template` on `state.log.messages`. The next language-model turn and the next router vector therefore both see the command once and the result once.

`B2d1` never asks the router for hop 2: `live_runner.py` passes `None` from the vector function when `run_id == "B2d1"` and `state.step > 0`, and `routed_policy` then calls the inner model. `B2chain` must not take that branch. The argument prompt is still appended to the current user turn (the tool-result turn, once one exists) because a second user turn in a row breaks the template. That concatenation stays.

Gold lengths on the test split: 9 of length 1 (`finish` only), 89 of length 2, 6 of length 3. `B2chain` caps at 4 steps so a length-3 trajectory fits (tool, tool, finish) and the identical-success brake has one spare step. It does not restore B2's 6-step cap.

## Features / Tasks

### S1: Stop generation after one closed tool call

**Files:** `actproto/agent/stop.py` (new), `actproto/agent/model.py` (modify `generate_text`), `tests/test_generation_stop.py` (new)
**Complexity:** S
**Depends on:** none

Add `completion_should_stop(text) -> bool`. It is true when the decoded new tokens contain one closed `<tool_call>...</tool_call>` or the token `<|END_OF_TURN_TOKEN|>`. A half-open tag does not stop. `generate_text` passes that check as a `StoppingCriteria` to `model.generate`. `max_new_tokens` stays the hard cap. Do not change `parse_tool_call`. `tests/test_agent_parse.py::test_parse_multiple_json_refuses` stays as it is. The new test covers an empty string, a half-open tag, one closed tag, and an end-of-turn token, and it does not call `load_model`.

### F1: One extra finish generation when the answer is empty

**Files:** `actproto/agent/routed_strategy.py` (modify the `model_fn` inside `routed_policy`), `tests/test_finish_retry.py` (new)
**Complexity:** S
**Depends on:** none

Only when the selected tool is `finish`. The first generation is unchanged. If it parses as `finish` with a non-empty `answer` that `validate_args` accepts, use it. If the answer is empty or whitespace, the parse fails, or the parsed name is not `finish`, call `generate_fn` once more. The second prompt is the same messages plus one line: `The finish answer must be non-empty. Emit one tool call.` If that second call is `finish` with a non-empty valid answer, use it. Otherwise return the existing invalid sentinel and let `regenerate_arguments` fall through to the heuristic. Other tools stay at one generation. Do not read gold answers. Do not add a third generation.

The test uses a fake `generate_fn` that returns `finish` with `""` and then `finish` with `Stockholm`, and a second fake that returns empty twice so the heuristic answer `""` is what gets executed. No model weights.

### T1: First-hop flag and the two new run ids

**Files:** `actproto/configs/trial_schema.py` (modify `RunSpec`), `actproto/configs/trial.yaml` (modify), `actproto/evaluation/live_runner.py` (modify `build_live_project`), `tests/test_trial_config.py` (modify), `tests/test_run_trial.py` (modify)
**Complexity:** S
**Depends on:** none

Add `route_first_hop_only: bool = False` to `RunSpec`. In `build_live_project`, replace the `run_id == "B2d1"` check with `spec.route_first_hop_only`. Set `route_first_hop_only: true` on the existing `B2d1` entry so a future rerun of that id keeps today's behavior.

Add two runs. Do not edit the `B0`, `B1`, or `B2` entries.

- `B2d1r`: same adapter, prototypes, and threshold as `B2d1`. `router: true`, `route_first_hop_only: true`, `max_agent_steps: 2`, `replicas: 1`. Description: depth-1 retry after the one-call stop.
- `B2chain`: same adapter, prototypes, and threshold. `router: true`, `route_first_hop_only: false`, `max_agent_steps: 4`, `replicas: 1`. Description: route every hop from the transcript, including the tool call and the tool result.

Tests load `trial.yaml` and assert the flag, both new ids, and that `B0` through `B2d1` still resolve. A unit test of the vector closure asserts that `route_first_hop_only` yields `None` after step 0 and that `B2chain` does not. No Tiny Aya load.

### E1: Depth-1 retry on the 5060 Ti

**Files:** `docs/EXPERIMENT_STATUS.md` (append one pointer)
**Complexity:** M
**Depends on:** S1, F1, T1

Wait until S1, F1, and T1 are integrated and no `run_trial.py` is on the GPUs. From `lexsi/`, with the venv interpreter:

```text
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2d1r
```

Unset `ACTPROTO_CUDA_DEVICES` in that shell when the process exits. Write `results/b0_b2/B2d1r/`. Do not pass `--run all`. Do not start `B2chain` in this task.

Append one pointer to `docs/EXPERIMENT_STATUS.md`. Report verified task success, first-tool accuracy, first-argument correctness, the count of `multiple tool calls` parse errors, and median latency. Compare first-tool accuracy to the saved B2 rate of 1.000. A drop is a bug in the first hop, whose transcript is still system plus user. Task success may stay low if the single finish string is still wrong. Record that. Do not add a mask, a third step, or the plan 1 answer rewrite here.

### E2: Chained routing on the 5060 Ti

**Files:** `docs/EXPERIMENT_STATUS.md` (append one pointer)
**Complexity:** M
**Depends on:** E1

Start only after `B2d1r` has written `results/b0_b2/B2d1r/eval.json` and that process has exited. Same device and interpreter:

```text
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2chain
```

Unset `ACTPROTO_CUDA_DEVICES` afterward. Write `results/b0_b2/B2chain/`. Leave `B2d1r` untouched.

Append one pointer. Report verified task success, prefix tool choice, first-tool accuracy, repeated calls, mean steps, and median latency. Split the 6 length-3 tasks out: how many hop-2 tool names differ from hop 1, and how many match gold. Hop-2 prototypes were not trained. A weak hop-2 number is a result, not a reason to refit prototypes on the test split or to raise the step cap.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | Stopping criteria and both evals use the installed stack. | |

## File Change Summary

| File | Action (new/modify/delete) |
| --- | --- |
| `actproto/agent/stop.py` | new |
| `actproto/agent/model.py` | modify |
| `tests/test_generation_stop.py` | new |
| `actproto/agent/routed_strategy.py` | modify |
| `tests/test_finish_retry.py` | new |
| `actproto/configs/trial_schema.py` | modify |
| `actproto/configs/trial.yaml` | modify |
| `actproto/evaluation/live_runner.py` | modify |
| `tests/test_trial_config.py` | modify |
| `tests/test_run_trial.py` | modify |
| `docs/EXPERIMENT_STATUS.md` | modify |

## Open Questions

- `B2chain` scores hop 2 with prototypes fit on first-tool activations only. This plan reports that score. It does not dump hop-2 activations or refit.
- If `B2d1r` turns the 95 parse errors into answer mismatches such as `429.0` versus `429`, that is plan 1 R1. Do not start that rewrite from inside E1 or E2.
- The empty-answer retry can still leave the 9 capitals empty if the second generation is also empty. Report that count. Do not pull the city out of the prompt with a new regex.
