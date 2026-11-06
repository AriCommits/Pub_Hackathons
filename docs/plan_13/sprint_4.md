# Sprint 4 — Historical tie record and prefix run config

**Plan:** docs/plan_13/OVERVIEW.md
**Wave:** 4 of 5
**Can run in parallel with:** none — serial
**Must complete before:** Sprint 5

---

## Agents in This Wave

Sprints 1–3 are merged. Template, per-turn budget, prefix scoring, component logs, and the tied-batch gate are in tree. This wave records the old tool-reward runs as negative results and adds a new config that points at a new directory. It is alone because it edits `actproto/training/grpo_run.py`, `actproto/training/rl_config.py`, `tests/test_grpo_config.py`, and `tests/test_train_grpo.py` after Sprint 3, and `docs/RL_BACKEND_COMPATIBILITY.md`, which Sprint 5 also edits.

Zero GRPO loss on `results/rl_v2/grpo_gpu_tools` and `results/rl_v2/grpo_gpu_tools_r2` is tied rewards, not a solved val set. The 200/200 macro `1.0` is `results/sft_v2/e2-scored`, the frozen SFT adapter validation. The saved adapter sha256 `2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd` matches `results/sft_v2/e2/checkpoint-150`. Beta stays `0.0`. Do not open the sealed test split. Do not put Tiny Aya on the 8 GB RTX 4060 (`5b885e12-623f-268c-f07c-ee33db379969`). The 5060 Ti UUID is `134c34ef-c155-b001-4e3a-b9134b072c21`. Do not train, do not delete historical adapters, and do not edit `docs/plan_13/OVERVIEW.md`.

Run tests with `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe` from the `lexsi` directory. Never use `C:\Python314\python.exe`. If you commit, `git add` only the files listed below. Do not run `git add .`. Do not push.

### Agent A: G5 — Record the tied runs as tied, and point the next run at a new directory

**Complexity:** M
**Estimated time:** 4–6 hours
**Files to modify:**
- `actproto/configs/grpo_v2_gpu_prefix.yaml` (NEW) — short prefix-reward smoke config on the 5060 Ti
- `actproto/training/grpo_run.py` — `HISTORICAL_RUNS` and `refuse_historical_output` reject the old tool-reward directories
- `actproto/training/rl_config.py` — loader accepts only per-turn 256, beta 0, and the new output directory for this file
- `tests/test_grpo_config.py` — new YAML loads; other per-turn values and the 4060 UUID fail
- `tests/test_train_grpo.py` — historical directory names are refused; the tools YAML stays on its old output
- `docs/RL_BACKEND_COMPATIBILITY.md` — both tool-reward runs written up as negative results

**Depends on:** G1, G4
**Blocks:** G6

**Instructions:**
Read `docs/plan_13/OVERVIEW.md` for context. Do not edit `actproto/configs/grpo_v2_gpu_tools.yaml`. The tools YAML and its output-directory assertion stay pointed at `results/rl_v2/grpo_gpu_tools`.

Add `actproto/configs/grpo_v2_gpu_prefix.yaml`. The new file keeps beta `0.0`, learning rate `5.0e-6`, `num_generations: 2`, batch 1, gradient accumulation 2, temperature `0.8`, top-p `0.95`, `use_vllm: false`, bf16, gradient checkpointing, the 5060 Ti name and UUID `134c34ef-c155-b001-4e3a-b9134b072c21`, and the e2 baseline manifest. Add `per_turn_new_tokens: 256` and keep `max_completion_length: 256` as the trainer denominator. The loader rejects any other per-turn value. `output_dir` is `results/rl_v2/grpo_gpu_prefix`. `max_steps` for this file is a short smoke bound of no more than 20, so a later operator does not inherit an unbounded run. The loader still rejects any beta other than 0 and still refuses the 4060 UUID `5b885e12-623f-268c-f07c-ee33db379969`.

Add `grpo_gpu_tools` and `grpo_gpu_tools_r2` to `HISTORICAL_RUNS`, next to `grpo_gpu_e2`, `grpo_gpu_smoke`, and `grpo_cpu_smoke`. `refuse_historical_output` must reject those directory names. Do not delete anything under them. Do not reuse those directories as the new output.

In `docs/RL_BACKEND_COMPATIBILITY.md`, record both tool-reward runs as negative results: reward mean `-0.1`, standard deviation `0`, gradient norm `0`, clipped completions at 256, terminated length `0`, every group tied, adapter sha256 `2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd`, identical to `results/sft_v2/e2/checkpoint-150`. State that the macro `1.0` on 200/200 val tasks is `results/sft_v2/e2-scored`, the SFT adapter validation, and that GRPO has not solved the val set. State that increasing the group size does not fix a systematic tie. Leave the beta section at 0, and do not describe a base-model KL as an e2 reference.

Preserve the Sprint 1 sampler, the Sprint 2 prefix score, and the Sprint 3 diagnostics and tied-batch gate. Do not write the smoke procedure; that is Sprint 5, and it will append to the same compatibility doc. Do not launch training. Do not pass `--final-eval` or `--split test`.

**Definition of Done:**
- [ ] `actproto/configs/grpo_v2_gpu_tools.yaml` is unchanged and still points at `results/rl_v2/grpo_gpu_tools`.
- [ ] The new YAML loads only with per-turn 256, beta 0, `num_generations: 2`, and `results/rl_v2/grpo_gpu_prefix`.
- [ ] Any other per-turn value, any beta other than 0, and the 4060 UUID are rejected.
- [ ] `max_steps` in the new file is at most 20.
- [ ] A test rejects training output named `grpo_gpu_tools` or `grpo_gpu_tools_r2`.
- [ ] Nothing under those historical directories is deleted.
- [ ] The compatibility doc states the tied-run metrics, the adapter hash, that 200/200 is SFT e2, and that a larger group does not fix a systematic tie. Beta stays 0.
- [ ] Tests written and passing for the modified files, using `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe -m pytest tests/test_grpo_config.py tests/test_train_grpo.py`
- [ ] No regressions in existing tests
