# Sprint 3 — Wave 3: Chained routing on the 5060 Ti

**Plan:** `docs/plan_3/OVERVIEW.md` (read for full context)
**Wave:** 3 of 3
**Can run in parallel with:** none — serial, same GPU as Sprint 2
**Must complete before:** nothing — this is the last wave

## Preconditions

- **Wave 2 complete.** `results/b0_b2/B2d1r/eval.json` exists and that `run_trial.py` process has exited.
- S1's stopping criterion is in `generate_text`. T1's `B2chain` entry has `route_first_hop_only: false` and `max_agent_steps: 4`.
- Project root: `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter: `.venv\Scripts\python`.
- One replica. `ACTPROTO_CUDA_DEVICES=1`.

---

## Agents in This Wave

### Agent A: E2 — Chained routing on the 5060 Ti

**Complexity:** M
**Estimated time:** 2.5 hours (up to 4 steps, a router forward pass on each hop after the first)
**Files to modify:**
- `docs/EXPERIMENT_STATUS.md` — append one pointer after the run. Do not rewrite the file or the E1 pointer.

**Depends on:** E1
**Blocks:** none

**Instructions:**
`B2chain` routes every hop. After each executed call the transcript already contains both pieces: the assistant `<tool_call>` string, and the next user turn `Tool result:\n` plus the JSON result. `make_vector_fn` encodes that transcript when `state.step > 0`. Do not add another copy of the command into the result text. Do not set `route_first_hop_only` on this run.

```text
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2chain
```

Unset `ACTPROTO_CUDA_DEVICES` when the process exits. Write `results/b0_b2/B2chain/`. Leave `B2d1r` and the older runs untouched. Do not refit prototypes. The bank was fit on first-tool activations. Hop 2 using that bank is the measurement.

Append one pointer with:

- verified task success
- prefix tool choice
- first-tool accuracy
- repeated calls
- mean steps
- median latency
- the 6 length-3 tasks: how many hop-2 names differ from hop 1, and how many match gold

A weak hop-2 number is a result. Do not refit on the test split, do not raise the step cap past 4, and do not add a used-tool mask.

**Definition of Done:**
- [ ] `results/b0_b2/B2chain/eval.json` exists and has 104 rows.
- [ ] `results/b0_b2/B2d1r/` and `results/b0_b2/{B0,B1,B2,B2d1}/` were not modified.
- [ ] One pointer is appended, including the length-3 hop-2 split.
- [ ] `ACTPROTO_CUDA_DEVICES` is not left set in the shell that started the run.

---

## Wave-3 integration

Single agent, single GPU run. This wave ends the plan. The two result directories stay side by side: `B2d1r` is the depth-1 retry, `B2chain` is the chained-tool run.
