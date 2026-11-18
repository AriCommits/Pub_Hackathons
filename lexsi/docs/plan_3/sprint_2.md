# Sprint 2 — Wave 2: Depth-1 retry on the 5060 Ti

**Plan:** `docs/plan_3/OVERVIEW.md` (read for full context)
**Wave:** 2 of 3
**Can run in parallel with:** none — serial
**Must complete before:** Sprint 3

## Preconditions

- **Wave 1 complete.** E1 depends on **S1**, **F1**, and **T1**.
- The Wave 1 pytest command in `sprint_1.md` has passed.
- Project root: `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter: `.venv\Scripts\python`.
- No other `run_trial.py` is using the GPUs. One replica. `ACTPROTO_CUDA_DEVICES=1` (`cuda:1` is the RTX 5060 Ti).
- Do not leave `CUDA_VISIBLE_DEVICES` set in the parent shell. The trial process pins the device itself when `ACTPROTO_CUDA_DEVICES` is set.

---

## Agents in This Wave

### Agent A: E1 — Depth-1 retry on the 5060 Ti

**Complexity:** M
**Estimated time:** 1.5 hours (104 tasks, about 32 s each on the last depth-1 run, plus model load)
**Files to modify:**
- `docs/EXPERIMENT_STATUS.md` — append one pointer after the run. Do not rewrite the file.

**Depends on:** S1, F1, T1
**Blocks:** E2

**Instructions:**
This is the retry of `B2d1`: route hop 1, then one inner-model turn, at most 2 steps. The new pieces are the one-call stop and the empty-finish retry. The run id is `B2d1r`. It writes `results/b0_b2/B2d1r/`.

```text
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2d1r
```

Unset `ACTPROTO_CUDA_DEVICES` when the process exits. Do not pass `--run all`. Do not start `B2chain`. Do not overwrite `results/b0_b2/{B0,B1,B2,B2d1}`, `results/adapters/b1`, or `results/prototypes.npz`.

Append one pointer to `docs/EXPERIMENT_STATUS.md` with:

- verified task success
- first-tool accuracy
- first-argument correctness
- count of outcome reasons that contain `multiple tool calls`
- median latency
- the comparison of first-tool accuracy to the saved B2 rate of **1.000**

A first-tool drop is a bug in the first hop. The first hop's transcript is still system plus user, so stop and investigate before calling the run done. Task success may stay low when the single finish string is wrong (`429.0` versus `429`, or a still-empty capital). Record the reason counts. Do not add a mask, a third step, or the plan 1 answer rewrite in this sprint.

**Definition of Done:**
- [ ] `results/b0_b2/B2d1r/eval.json` exists and has 104 rows.
- [ ] `results/b0_b2/{B0,B1,B2,B2d1}/eval.json` were not modified by this run.
- [ ] One pointer is appended to `docs/EXPERIMENT_STATUS.md` with the five numbers and the 1.000 comparison.
- [ ] `ACTPROTO_CUDA_DEVICES` is not left set in the shell that started the run.

---

## Wave-2 integration

Single agent, single GPU run. After `eval.json` is on disk, confirm the protected result files still have their previous timestamps, then proceed to `sprint_3.md`.
