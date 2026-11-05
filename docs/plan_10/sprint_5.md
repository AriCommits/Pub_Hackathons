# Sprint 5 — Bounded RL study and checkpoint selection

**Plan:** docs/plan_10/OVERVIEW.md
**Wave:** 5 of 7
**Can run in parallel with:** none across waves; agents inside this wave may work in parallel.
**Must complete before:** Sprint 6

---

## Preconditions and shared execution contract

Depends on Plan 9 (`docs/plan_9/OVERVIEW.md`) and its immutable `docs/SFT_BASELINE_MANIFEST.json`. Source tree's current `scripts/train_grpo.py` accepts empty data, prints an import check, and never trains. `actproto/training/rewards.py` supplies disconnected reward helpers. Existing SFT is a valuable initialization/control, not RL. Existing 104/104 system result is not a standalone SFT baseline.

The current [AgentTune repository](https://github.com/Lexsi-Labs/AgentTune) exposes a GRPO factory with tool-rollout support and a Tiny Aya example. Use it as the intended backend, then prove that the pinned version performs the required behavior. Do not copy the example's superficial text-based reward. [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) describes online grouped generation, token accounting and template constraints. These are references, not proof that today's installed stack supports our format. [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) supplies recorded-episode evaluation through Plan 9's tested bridge.

Use a separate `.venv-rl` or documented isolated Linux/WSL environment if compatibility requires it; keep the existing SFT environment working. A Linux move is a contingency after a concrete compatibility failure, not a silent assumption. Record all backend choices and platform tests. Observed training card: RTX 5060 Ti ~16 GB; discover UUID/device mapping again. Never launch concurrent model-bearing jobs. Read the long-running-background-tasks skill before launching or supervising any run. This plan authoring does not install packages or start training.

### Mandatory cross-plan gates

Interface exploration R0 can begin after Plan 9 S0/S6 contracts and locks exist. Rollout/reward implementation can use toy fixtures after those interfaces stabilize. No real optimization until Plan 9 S7 has frozen SFT selection, test hashes, model/template/verifier hashes and baseline provenance. Plan 9 S8 documentation must be integrated before the final RL report. Any change to shared scoring/data semantics requires a versioned amendment and rerunning both controls, not a favorable one-sided comparison.

Primary evaluation uses Plan 9's deterministic router-off, semantic-fallback-off mode and identical task budgets. RL begins from the selected SFT checkpoint; the KL reference, if used, is that same SFT policy, not the unadapted base. Optional routing is a post-training system ablation only. Native-model tools, format conversion and observation serialization must be equivalent across SFT and RL evaluation.

### Initial optimization and budget choices

GRPO only. Smoke: two prompts, two stochastic rollouts each, one or two optimizer updates. Pilot: 32 train tasks stratified by family, 20 updates, validation on a fixed 40-task validation subset. Candidate full run: up to 200 updates, evaluate full validation every 25 updates, save resumable checkpoints at each evaluation, early-stop after three non-improving evaluations. Start learning rate 5e-6, LoRA capacity inherited from SFT, group size 2 (4 only if measured memory allows), temperature .8, top-p .95, gradient checkpointing, 256 generated tokens per call. Validate installed-trainer batch/group divisibility; do not assume `batch=1, group=2` is accepted. Use sequential generation if supported and document its batch semantics. Bound total tokens per episode and label truncation; do not silently reduce evaluation budgets to fit training.

These are initial config hypotheses, adjusted only on train/validation pilots before final run freeze. Pilot may compare 2e-6 and 5e-6; at most two full seed-42 candidates including the selected pilot settings. A nominal KL coefficient .02 is allowed only with a proven frozen SFT reference implementation. If no such reference fits memory, explicitly run/report beta=0 and use a validation drift gate; never label base-only reference as SFT reference. Record sampling logprobs, likelihood recomputation and behavior-policy age; no unrestricted stale rollout replay.

Budget: at most 36 GPU-hours for RL study plus controls, selected after a ten-minute throughput/memory pilot; do not claim a duration before measuring. Reserve final evaluation time before spending the training budget. Minimum experiment is one real seed-42 RL run and both controls; seeds 43/44 and shaped-reward ablation are optional. Three seed runs are required for seed-stability claims; otherwise label findings exploratory. Cut optional experiments rather than verifier checks, save/reload validation, or the fair comparison.

Integrate and verify all prior waves of Plan 10 before starting this wave. External dependencies below must exist. Work in an isolated branch/worktree; modify only owned files. All new outputs go to `results/rl_v2/`. Preserve old artifacts. One GPU-owning job across both plans. Implementation-only tasks do not launch long experiments.

## Agents in This Wave

### Agent A: R5 — Bounded RL pilot and validation-only selection

**Complexity:** L
**Estimated time:** 3–5 engineering hours plus measured GPU runtime (rough estimate).
**Files to modify:**

- `scripts/run_rl_study.py` — NEW; owned by R5.
- `actproto/configs/rl_study.yaml` — NEW; owned by R5.
- `docs/RL_TRAINING_RESULTS.md` — NEW; owned by R5.
- `docs/RL_SELECTED_MANIFEST.json` — NEW; owned by R5.

**Depends on:** R4
**Blocks:** R6

**Instructions:**

Execute the initial optimization schedule and time cap above with run manifests and one GPU lease. Primary reward is sparse verified success. Pilot logs distribution of valid successes, reward variance and truncation by family. If groups have zero variance, diagnose task difficulty using train only and use fixed train-stratified sampling/temperature pilot; do not secretly add shaping or mine final-test errors. Freeze chosen settings after at most two pilot learning rates. Allow at most two full candidates, select checkpoints by validation macro verified success with Plan 9 tie-breaks and regression gate. Do not select by training reward.

Evaluate full validation every 25 updates, save/stop on finite-state failures or the configured patience, preserve every failed candidate. Monitor PID/log heartbeat, loss/reward components, NaNs, KL drift, repeated calls, memory and token budget until terminal status. An interrupted run is resumable or failed, never completed. Repeat selected config on independent seeds only within budget; optional shaped reward gets a separate run ID and same budgets. Record what was omitted.

**Definition of Done:**

- [ ] At least one real trained and reloadable RL candidate with provenance and full validation traces exists, even if worse than SFT. Selection manifest freezes checkpoint, reward version, config, final-test hashes and remaining final-eval budget. SFT baseline and verifier unchanged. No final-test performance used to choose settings or checkpoint.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

## Integration and what this wave unblocks

Follow `docs/plan_10/PARALLEL-WORK.md`. The integrator reviews API compatibility and file ownership, merges completed task commits, and runs the full suite. This unblocks Sprint 6.
