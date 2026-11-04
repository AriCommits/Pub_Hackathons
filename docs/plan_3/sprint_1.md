# Sprint 1 — Wave 1: Stop after one call, empty-finish retry, trial ids

**Plan:** `docs/plan_3/OVERVIEW.md` (read for full context)
**Wave:** 1 of 3
**Can run in parallel with:** none — this is the first wave
**Must complete before:** Sprint 2

## Preconditions

- Nothing. S1, F1, and T1 have `Depends on: none` and share no files.
- Project root: `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter: `.venv\Scripts\python`.
- Do not start `run_trial.py` or any GPU work in this wave. CPU only.
- Do not edit `results/b0_b2/{B0,B1,B2,B2d1}` or `results/adapters/b1`.

## What this wave unblocks

- S1, F1, and T1 together unblock **E1** (Sprint 2). E2 waits until E1's eval file exists.

---

## Agents in This Wave

### Agent A: S1 — Stop generation after one closed tool call

**Complexity:** S
**Estimated time:** 1 hour
**Files to modify:**
- `actproto/agent/stop.py` (NEW) — `completion_should_stop(text) -> bool`
- `actproto/agent/model.py` — call it from a `StoppingCriteria` inside `generate_text`
- `tests/test_generation_stop.py` (NEW) — pure string cases, no model load

**Depends on:** none
**Blocks:** E1, E2

**Instructions:**
`generate_text` decodes until `max_new_tokens` (256). `B2d1` then fails 95/104 tasks with `PARSE_ERROR: multiple tool calls in one turn` because `model_policy` turns that parse error into `finish`.

`completion_should_stop` is true when the decoded new tokens contain one closed `<tool_call>...</tool_call>` or the literal `<|END_OF_TURN_TOKEN|>`. A half-open `<tool_call>` does not stop. An empty string does not stop. Wire it as a transformers `StoppingCriteria` on `model.generate` in `generate_text`. Leave `max_new_tokens` as the hard cap. Leave `do_sample=False`.

Do not change `parse_tool_call`. Two calls in one string remain a `ParseError`. `tests/test_agent_parse.py::test_parse_multiple_json_refuses` must stay green. The new test does not import `load_model`.

**Definition of Done:**
- [ ] `completion_should_stop` is true for one closed tool call and for `<|END_OF_TURN_TOKEN|>`, and false for empty text and a half-open tag.
- [ ] `generate_text` passes that predicate as a stopping criterion and still caps `max_new_tokens`.
- [ ] `.venv\Scripts\python -m pytest tests/test_generation_stop.py tests/test_agent_parse.py` passes.
- [ ] No regressions in existing tests for files this agent touched.

### Agent B: F1 — One extra finish generation when the answer is empty

**Complexity:** S
**Estimated time:** 1.5 hours
**Files to modify:**
- `actproto/agent/routed_strategy.py` — second `generate_fn` call only for `finish`
- `tests/test_finish_retry.py` (NEW) — fake generator, no weights

**Depends on:** none
**Blocks:** E1, E2

**Instructions:**
The 9 capital-city failures are `finish` with `answer: ""`. `validate_args` accepts that because it only checks that `answer` is a string. The heuristic in `schema_regen.py` also returns `""` for those prompts. Do not add a city regex and do not read gold.

In the `model_fn` closed over by `routed_policy`, keep a single generation for every tool other than `finish`. For `finish`:

1. Generate once, as today, with `argument_prompt` on the current user turn.
2. If the parse is `finish` and `answer` is non-empty and `validate_args("finish", arguments)` succeeds, return those arguments.
3. Otherwise call `generate_fn` one more time. Add this line to the same user content: `The finish answer must be non-empty. Emit one tool call.`
4. If the second parse is `finish` with a non-empty valid answer, return it.
5. Otherwise return `{"__invalid__": True}` so `regenerate_arguments` falls through to the heuristic.

No third generation. Do not change `schema_regen.py`. Do not change the identical-success fallthrough or the failed-call retry already in `routed_policy`.

Tests: a fake generator returns `finish` `""` then `finish` `Stockholm`, and the executed arguments are `{"answer": "Stockholm"}` with exactly two generations. A second fake returns empty twice, the executed answer is `""`, and `generate_fn` was called twice. A `lookup_record` route still calls `generate_fn` once.

**Definition of Done:**
- [ ] `finish` gets at most two generations; every other selected tool gets one.
- [ ] A non-empty second answer is kept. Two empty answers fall through to the heuristic.
- [ ] `.venv\Scripts\python -m pytest tests/test_finish_retry.py tests/test_arg_prompt.py tests/test_routed_strategy.py` passes.
- [ ] No regressions in existing tests for files this agent touched.

### Agent C: T1 — First-hop flag and the two new run ids

**Complexity:** S
**Estimated time:** 1 hour
**Files to modify:**
- `actproto/configs/trial_schema.py` — `RunSpec.route_first_hop_only: bool = False`
- `actproto/configs/trial.yaml` — flag on `B2d1`, add `B2d1r` and `B2chain`
- `actproto/evaluation/live_runner.py` — honor the flag instead of `run_id == "B2d1"`
- `tests/test_trial_config.py` — the new ids and the untouched B0–B2 entries
- `tests/test_run_trial.py` — vector closure: first-hop-only is `None` after step 0; `B2chain` is not

**Depends on:** none
**Blocks:** E1, E2

**Instructions:**
`build_live_project` currently special-cases `run_id == "B2d1"` and returns no router vector after step 0. Replace that check with `spec.route_first_hop_only`. Set the flag true on the existing `B2d1` entry. Do not change `B0`, `B1`, or `B2`.

Add:

- `B2d1r`: adapter `results/adapters/b1`, `router: true`, prototypes `results/prototypes.npz`, threshold `results/prototype_metrics.json`, `route_first_hop_only: true`, `max_agent_steps: 2`, `replicas: 1`. Description: depth-1 retry after the one-call stop.
- `B2chain`: the same adapter, prototypes, and threshold, `route_first_hop_only: false`, `max_agent_steps: 4`, `replicas: 1`. Description: route every hop from the transcript, including the tool call and the tool result.

`_filter` already drops unknown YAML keys, so the new field must exist on `RunSpec` or the flag will be ignored. Do not run the GPU eval. The vector-closure test must not load Tiny Aya: build the lambda the same way `build_live_project` does, with a fake `vector_fn` if the real encoder cannot be constructed without weights. Asserting the flag on the loaded `RunSpec` is the required config check. A direct test that the `B2d1` / `B2d1r` branch returns `None` when `state.step > 0`, and the `B2chain` branch returns the inner vector, is the required behavior check.

**Definition of Done:**
- [ ] `B2d1` and `B2d1r` set `route_first_hop_only` true. `B2chain` sets it false and caps steps at 4.
- [ ] `B0`, `B1`, and `B2` entries are unchanged.
- [ ] `.venv\Scripts\python -m pytest tests/test_trial_config.py tests/test_run_trial.py` passes.
- [ ] No regressions in existing tests for files this agent touched.

---

## Wave-1 integration

Three agents, three disjoint file sets. Merge order does not matter. After all three land:

```text
.venv\Scripts\python -m pytest tests/test_generation_stop.py tests/test_agent_parse.py tests/test_finish_retry.py tests/test_arg_prompt.py tests/test_routed_strategy.py tests/test_trial_config.py tests/test_run_trial.py
```

Then proceed to `sprint_2.md`. Do not start `B2d1r` until this pytest is green.
