# Sprint 2 — Score live sessions with the prefix version

**Plan:** docs/plan_13/OVERVIEW.md
**Wave:** 2 of 5
**Can run in parallel with:** none — serial
**Must complete before:** Sprint 3, Sprint 5

---

## Agents in This Wave

Sprint 1 is merged. `bounded_prefix_v1` exists, and the study sampler already passes the `default` chat template with 256 new tokens per turn. This wave only switches the live trainer score. It shares `actproto/training/grpo_run.py` and `tests/test_train_grpo.py` with Sprint 3, so those files cannot be edited again until this wave is merged.

Zero GRPO loss is tied rewards, not a solved val set. The 200/200 macro `1.0` is frozen SFT e2 (`results/sft_v2/e2-scored`), not a GRPO result. Beta stays `0.0`. Do not open the sealed test split. Do not put Tiny Aya on the 8 GB RTX 4060 (`5b885e12-623f-268c-f07c-ee33db379969`). The 5060 Ti UUID is `134c34ef-c155-b001-4e3a-b9134b072c21`. Do not train and do not edit `docs/plan_13/OVERVIEW.md`.

Run tests with `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe` from the `lexsi` directory. Never use `C:\Python314\python.exe`. If you commit, `git add` only the files listed below. Do not run `git add .`. Do not push.

### Agent A: G3 — Score the live trainer with the prefix version

**Complexity:** S
**Estimated time:** 2–3 hours
**Files to modify:**
- `actproto/training/grpo_run.py` — `_executed_trace_reward` scores captured live sessions with `bounded_prefix_v1`
- `tests/test_train_grpo.py` — trainer-path tests for the prefix version, a missing session, and no length bonus

**Depends on:** G2
**Blocks:** G6

**Instructions:**
Read `docs/plan_13/OVERVIEW.md` for context. Keep the Sprint 1 sampler changes in `grpo_run.py`: explicit `default` template, 256 new tokens per sampled turn, and no shared episode pool. Do not revert them and do not widen this edit into diagnostics or the tied-batch gate. Those are Sprint 3.

`_executed_trace_reward` scores captured live sessions with `bounded_prefix_v1`. A decoded completion that merely contains `"finish"` stays at reward `0` when no session was captured. The completion text is not the score. Do not add a length term in the trainer wrapper. The reward function returns one scalar per completion, the same scalar the ledger stores, so TRL still group-normalizes one reward.

If a ledger for that completion is already finalized, read that ledger and do not score the session a second time. Sprint 3 is what finalizes the ledger during a resample. This function only has to honor a ledger that is already there. Do not implement the redraw in this sprint.

Leave `verified_success_v2` and `bounded_shaped_v1` numerically unchanged for existing callers. `score_executed_session` may stay on the shaped version; the trainer path is the one that must request `bounded_prefix_v1`. Do not treat `num_generations: 4` as a fix. Do not set beta above 0. Do not open the test split, do not launch `scripts/train_grpo.py`, and do not load Tiny Aya.

**Definition of Done:**
- [ ] A trainer-path test with a live session asserts `bounded_prefix_v1`.
- [ ] A missing session asserts reward `0` for text that contains `"finish"`.
- [ ] No assertion introduces a length bonus, and the wrapper adds none.
- [ ] A finalized ledger is read once and the session is not scored again.
- [ ] Sprint 1 per-turn budget and template wiring still behave as merged.
- [ ] `bounded_shaped_v1` numeric tests still pass.
- [ ] Tests written and passing for the modified files, using `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe -m pytest tests/test_train_grpo.py tests/test_rewards_v2.py`
- [ ] No regressions in existing tests
