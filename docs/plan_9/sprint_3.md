# Sprint 3 — Integrated inference and evaluation

**Plan:** docs/plan_9/OVERVIEW.md
**Wave:** 3 of 6
**Can run in parallel with:** none across waves; agents inside this wave may work in parallel.
**Must complete before:** Sprint 4

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

### Agent A: S4 — One evaluation runner with controlled inference settings

**Complexity:** L
**Estimated time:** 5–8 engineering hours (rough estimate).
**Files to modify:**

- `actproto/evaluation/experiment_runner.py` — NEW; owned by S4.
- `actproto/agent/inference_v2.py` — NEW; owned by S4.
- `actproto/agent/trace_v2.py` — NEW; owned by S4.
- `scripts/run_experiment_v2.py` — NEW; owned by S4.
- `actproto/configs/sft_matrix.yaml` — NEW; owned by S4.
- `tests/test_experiment_runner.py` — NEW; owned by S4.
- `tests/test_trace_v2.py` — NEW; owned by S4.

**Depends on:** S1, S2, S3
**Blocks:** S5, S6

**Instructions:**

Wire manifests, fixtures, verifier, saved checkpoints, and policy trace collection into one runner used by SFT validation and later RL evaluation. Primary mode disables routing, heuristic arguments and answer repair; schema validation and single-call stopping remain common. An optional deployed-system mode enables explicitly named interventions and logs every intervention, original model output, call source and additional generation. Invalid output must remain visible, not become a fabricated valid finish. No oracle values enter prompts or fallback generation. Enforce per-task budgets and versioned task environments. Use the same tokenizer/template contract for every candidate.

Expose `--manifest`, `--split`, `--run-id`, `--matrix`, `--adapter`, `--mode`, `--dry-run`, and an explicit `--final-eval` unlock backed by the frozen selection manifest. Training callbacks are forbidden from loading the test split. Seal raw traces plus aggregate metrics in separate result directories. Hash/run manifests include GPU, dtype, generation parameters, interventions, wall-clock timings, generated tokens and inference-engine versions.

**Definition of Done:**

- [ ] Fake policy runs verify budget caps, multiple calls rejected, full traces, expected failures and private-field exclusion. Exact same fixtures and scoring are used by validation and CLI. Old suite still passes. Known finish-only exploit fails end to end. Final test cannot be selected through a training callback or accidental default.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

## Integration and what this wave unblocks

Follow `docs/plan_9/PARALLEL-WORK.md`. The integrator reviews API compatibility and file ownership, merges completed task commits, and runs the full suite. This unblocks Sprint 4.
