# Sprint 5 — Bounded SFT study and baseline selection

**Plan:** docs/plan_9/OVERVIEW.md
**Wave:** 5 of 6
**Can run in parallel with:** none across waves; agents inside this wave may work in parallel.
**Must complete before:** Sprint 6

---

## Preconditions and shared execution contract

Read `docs/AUDIT_2026-11-04.md` for evidence. Existing tests: 88 passing. Best historical system `B2apchain` scored 104/104 but includes first-hop routing and fallbacks; router-off performance for its assistant-only adapter is unknown. `B2orders` scores 103/104 and `B2fmt` 88/104 on the old development set. Curriculum evaluations reuse all 38/266 evaluation prompts in training. The existing verifier accepts a correct finish-only answer on a task requiring tools.

This plan supersedes unfinished Plan 1 improvements for future work. Retain historical plans as records. Adopt useful requirements (mask tests, validation checkpointing, strict thresholds), replace answer-snapping with declared semantic scoring, and defer step-wise routing until a matched first-hop ablation justifies it. Plans 3–8 remain provenance, not cumulative implementation instructions. Plan 2 is not present in this checkout.

Project root is `C:/Users/arian/Github/Hackathons/lexsi`; Git root is its parent `Hackathons`. Use project `.venv/Scripts/python.exe` for existing tests. GPU identities must be detected by UUID/name at execution time: observed PyTorch cuda:1 is RTX 5060 Ti (~16 GB), cuda:0 is RTX 4060 (8 GB). One model-bearing GPU job at a time; never assume memory will accommodate concurrent workers. Read the long-running-background-tasks skill before launching or monitoring experiments. Existing environment remains untouched while toolkit compatibility is tested in a separate environment. No training is launched by creating this plan.

### Shared experiment contract

Version `tooluse_v2` separates `TaskSpec` (private oracle, split, group, tool dependencies), `PolicyInput` (public prompt, tool schemas, observations only), and `ExecutionTrace` (harness-recorded calls/results, provenance, model spans, stopping reason). Gold answers, expected arguments, split labels, and case identifiers are never passed in policy messages. A hash-addressed manifest ties together data, tool fixtures, verifier, prompt template, decoding, model revision, adapter, and code revision.

Use two task modes: `outcome` allows semantically equivalent solutions; `required_tools` checks declared calls/dependencies from trusted execution in addition to the result. An explicit sequence instruction can require an exact order; an ordinary business question should not be constrained to one arbitrary reference sequence. Record `answer_success`, `execution_compliance`, and `verified_success_v2` separately. Preserve `task_success_v1` only for historical comparison. Parse failure is an event, never a synthetic successful finish.

Target data budget: 1,200 train, 200 validation, 400 final-test tasks, expandable only before freeze if unique executable groups cannot meet quotas. Use 4 semantic strata: single-tool 25%, dependent 2–4 calls 35%, ordered/subset compositions 25%, recoverable failures and no-tool tasks 15%; provide per-stratum counts. Allocate 200 final tasks to new-entity in-distribution evaluation and 200 to held-out composition/prompt-family stress. Keep all layouts/paraphrases/variants of a logical task together. Split entities and argument tuples; additionally hold out composition families for stress. Do not claim long-chain generalization if its examples occur in train.

The 400-task final test is sealed from model selection for both plans. Freeze SFT choice before any final evaluation. Default: perform the single matched final SFT+RL evaluation in Plan 10. If Plan 9 must ship alone, final-test SFT once, record that disclosure, and do not use its failures to tune RL; reserve a second immutable test version before further tuning. Developer/agent access cannot literally be prevented by a local hash: enforce split guards, access logging, and workflow discipline.

Default policy evaluation is deterministic, router off, semantic fallback off, at most 256 generated tokens per call, with task-class budgets fixed in the manifest: 4 total calls for short tasks, 8 for dependent medium tasks, 12 for long diagnostics; finish counts as a call. Reject gold tasks that cannot fit. Training samples only train tasks. All candidates see the same schema/prompt contract. Record emitted tokens, model invocations, executed tool calls, failures, latency, and truncation separately.

Performance decision: primary validation metric is macro-averaged `verified_success_v2` across strata, with micro success also reported. Target improvement is at least 5 percentage points over the legacy assistant-only checkpoint under the same harness, with no more than 2 points loss on single-tool/no-tool strata and no more than 1.5x median latency. These are engineering targets, not promised results. If unmet, retain the stronger candidate and publish a negative result. Use family/group-clustered bootstrap paired confidence intervals; the point-estimate gate alone is not statistical significance. For efficiency claims, require a predeclared 2-point noninferiority margin and report uncertainty, including success-conditioned and unconditional steps.

Integrate and verify all prior waves of Plan 9 before starting this wave. External dependencies below must exist. Work in an isolated branch/worktree; modify only owned files. All new outputs go to `results/sft_v2/`. Preserve old artifacts. One GPU-owning job across both plans. Implementation-only tasks do not launch long experiments.

## Agents in This Wave

### Agent A: S7 — Bounded SFT experiments, error analysis, and checkpoint freeze

**Complexity:** L
**Estimated time:** 4–6 engineering hours plus measured GPU runtime (rough estimate).
**Files to modify:**

- `scripts/run_sft_study.py` — NEW; owned by S7.
- `scripts/compare_experiments_v2.py` — NEW; owned by S7.
- `actproto/evaluation/statistics_v2.py` — NEW; owned by S7.
- `tests/test_statistics_v2.py` — NEW; owned by S7.
- `docs/SFT_V2_RESULTS.md` — NEW; owned by S7.
- `docs/SFT_BASELINE_MANIFEST.json` — NEW; owned by S7.

**Depends on:** S5, S6
**Blocks:** S8

**Instructions:**

First CPU integration gate, then real tokenizer preflight, then a two-update GPU smoke with changed adapter hashes and finite loss, then full runs serially. Use one GPU lease for all train/eval/calibration work. Record PID, heartbeat, logs, output path, time/token budget and expected artifacts; monitor until completion and verify outputs before declaring done. Do not kill unrelated jobs or start a duplicate when a lease is held. Resume only compatible incomplete runs.

Validation study: Ebase pinned base model with no adapter on the same corrected harness; E0 old assistant-only checkpoint on corrected harness, router off; E1 new corrected-mask SFT on original train trajectories; E2 corrected-mask SFT on balanced v2 train; E3 optional bounded train-only failure repair of E2, only if E2 has recoverable train errors and budget remains. Record total examples, tokens, steps and GPU time for every candidate; E1→E2 is not compute-matched and must be described accordingly. Primary seed 42; replicate selected SFT candidate on seeds 43 and 44 only if the time budget permits. No rank sweep or unlimited retries. Use a maximum of 24 total GPU-hours for this study, with completion deadlines recorded at launch; first benchmark ten minutes to estimate feasibility, shorten candidate list before beginning if needed. Runtime is unknown until measured.

Use validation, never final test, to select adapter and optional router mode under the predeclared rule. Evaluate old 104-task set only as regression/development diagnostics. Run first-hop calibration on the selected adapter; report prototype and probe results together. Write per-stratum results, errors, confidence intervals, interventions, failure costs and limitations. Freeze checkpoint, train/val hashes, final-test hash, verifier, template, dependency locks, and primary router-off config in SFT_BASELINE_MANIFEST. The chosen candidate may be E0 if refinements do not help. Freeze also identifies the nominal RL reference policy as this SFT checkpoint.

**Definition of Done:**

- [ ] Trained weights reload and produce a real tool trace; data/model/config hashes and selection reasoning are complete; all failed/aborted runs remain labeled; protected artifact hashes unchanged. SFT performance target is tested, not asserted. Final-test isolation is preserved. Plan 10 starts optimization only from this immutable handoff.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

## Integration and what this wave unblocks

Follow `docs/plan_9/PARALLEL-WORK.md`. The integrator reviews API compatibility and file ownership, merges completed task commits, and runs the full suite. This unblocks Sprint 6.
