# Sprint 5 — Bounded smoke procedure and val comparison

**Plan:** docs/plan_13/OVERVIEW.md
**Wave:** 5 of 5
**Can run in parallel with:** none — serial
**Must complete before:** none

---

## Agents in This Wave

Sprints 1–4 are merged. The prefix config exists at `actproto/configs/grpo_v2_gpu_prefix.yaml` and the old tool-reward runs are already recorded as negative results. This wave writes the procedure for a later smoke and the val comparison. It does not start that job. It is alone because it edits `docs/RL_BACKEND_COMPATIBILITY.md` and `tests/test_train_grpo.py` after Sprint 4.

Zero GRPO loss is tied rewards, not a solved val set. The 200/200 macro `1.0` is frozen SFT e2 in `results/sft_v2/e2-scored`, not the GRPO run. A tied or identical adapter is a negative result. Beta stays `0.0`. Do not open the sealed test split. Do not put Tiny Aya on the 8 GB RTX 4060 (`5b885e12-623f-268c-f07c-ee33db379969`). The 5060 Ti UUID is `134c34ef-c155-b001-4e3a-b9134b072c21`. Do not train, do not launch the smoke, and do not edit `docs/plan_13/OVERVIEW.md`.

Run tests with `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe` from the `lexsi` directory. Never use `C:\Python314\python.exe`. If you commit, `git add` only the files listed below. Do not run `git add .`. Do not push.

### Agent A: G6 — Specify the bounded smoke and the val comparison

**Complexity:** S
**Estimated time:** 2–3 hours
**Files to modify:**
- `docs/RL_BACKEND_COMPATIBILITY.md` — append the smoke and val-comparison procedure; keep the Sprint 4 negative-result record
- `tests/test_train_grpo.py` — reject a comparison payload whose split is `test`; assert the procedure does not run from unit tests

**Depends on:** G1, G3, G4, G5
**Blocks:** none

**Instructions:**
Read `docs/plan_13/OVERVIEW.md` for context. This task writes the procedure. It does not start the job. Default pytest must not launch it. Do not edit `scripts/train_grpo.py`, `scripts/run_experiment_v2.py`, or `scripts/compare_experiments_v2.py`. Those commands are named, not changed.

Keep the Sprint 4 write-up of `grpo_gpu_tools` and `grpo_gpu_tools_r2`: reward mean `-0.1`, standard deviation `0`, gradient norm `0`, clipped completions at 256, terminated length `0`, every group tied, adapter sha256 `2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd`, identical to `results/sft_v2/e2/checkpoint-150`. Do not describe that hash or the 200/200 macro `1.0` as a GRPO win.

The smoke, when a later operator runs it, is one model-bearing process in the Plan 11 CUDA env (`C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-gpu-env`, torch `2.11.0+cu128`, AgentTune 1.1.3, TRL 1.7.1). A child process selects UUID `134c34ef-c155-b001-4e3a-b9134b072c21` and refuses `5b885e12-623f-268c-f07c-ee33db379969`. The parent shell does not keep `CUDA_VISIBLE_DEVICES`. No second model-bearing job runs at the same time. The command is `scripts/train_grpo.py --train` with `actproto/configs/grpo_v2_gpu_prefix.yaml`. The output directory is `results/rl_v2/grpo_gpu_prefix`. If the process OOMs, record the memory error once and stop. Do not move the job to the 4060, do not shrink the per-turn budget back into a shared 256-token pool, and do not raise the group size.

After the smoke, read `step_diagnostics.jsonl`. If every update was skipped or the adapter sha256 still matches e2 (`2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd`), write that as a negative result and include the component and termination breakdown. Do not send that adapter to val as an improvement. A val rescore is allowed only to confirm the checkpoint still matches e2, and only after the training process has released the 5060 Ti. If any update was not skipped, score that checkpoint with `scripts/run_experiment_v2.py` on `--split val`, `--mode primary`, matrix `actproto/configs/sft_matrix.yaml`, `--device-uuid 134c34ef-c155-b001-4e3a-b9134b072c21`. Do not pass `--final-eval` or `--split test`. Compare the val JSON to `results/sft_v2/e2/checkpoint-150` with `scripts/compare_experiments_v2.py`, which accepts val only. A gain below `0.05`, a failed run, or an identical adapter is a negative result. The 200/200 macro `1.0` number is not the GRPO score.

Beta stays `0.0`. Do not describe a KL term against base Tiny Aya as an e2 reference. Do not set `LEXSI_RL_GROUP4_FIT`. The plan file and the unit tests do not launch the smoke.

**Definition of Done:**
- [ ] The doc names the config `actproto/configs/grpo_v2_gpu_prefix.yaml`, UUID `134c34ef-c155-b001-4e3a-b9134b072c21`, output directory `results/rl_v2/grpo_gpu_prefix`, the val-only commands, and the identical-adapter hash.
- [ ] The doc says the parent shell must not keep `CUDA_VISIBLE_DEVICES`, the 4060 is refused, and an OOM is recorded once and stops the job.
- [ ] The doc says a fully skipped update or an adapter that still matches e2 is a negative result, and a val rescore of that adapter is only a match check after the 5060 Ti is released.
- [ ] A test rejects a comparison payload whose split is `test`.
- [ ] Default pytest does not launch the smoke, and this sprint does not start a GPU process.
- [ ] Sprint 4's negative-result record is still in the doc.
- [ ] Tests written and passing for the modified files, using `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe -m pytest tests/test_train_grpo.py`
- [ ] No regressions in existing tests
