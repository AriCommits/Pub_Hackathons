# Plan 9: Reliable supervised tool-use baseline and honest performance evaluation

## Summary

Refine ActProto-LoRA into a reproducible supervised baseline with execution-grounded verification, leakage-resistant data, correct assistant-token supervision, validation-selected checkpoints, and measured router ablations. Preserve the existing adapters and reports. Train a bounded set of SFT candidates, select using validation only, and export a frozen checkpoint and experiment contract for Plan 10's actual rollout-based RL comparison. Success means trustworthy measurements and a reusable baseline; performance improvement must be demonstrated rather than assumed.

## Goals

- Preserve checkpoint `7d090b5`, audit commit `1f489d6`, existing adapters, prototype banks, and all historical reports. Use new `results/sft_v2/` and `data/benchmarks/tooluse_v2/` paths.
- Make shortcut answers, fabricated observations, skipped required tools, and invalid termination fail the strict tool-use benchmark while permitting legitimate alternative solutions for outcome-only tasks.
- Freeze independently held-out task groups and a joint SFT/RL final-evaluation protocol before model selection. Keep the old 104 tasks as a named development/regression set.
- Train and compare legacy assistant-only SFT, corrected-mask SFT, balanced-curriculum SFT, and a bounded train-failure-repair candidate. Evaluate routing separately.
- Integrate actual AuditKIT evaluation, record dependencies and artifact hashes, and deliver a reloadable SFT checkpoint without requiring a completed RL run.
- Leave a tested interface for rollout collection, verification, policy-token masks, and reproducible comparisons that Plan 10 consumes.

## Non-Goals

- No RL optimization in this plan, no teacher distillation claim, no new hosted LLM provider, and no replacing Tiny Aya with a larger model.
- No per-tool LoRA library, implicit weight-update interpretation, model-agnostic routing claim, large router redesign, or all-hop prototype training. Those are separate research hypotheses.
- No blanket answer rewriting to make old exact-match scores pass. Fix task semantics and explicitly declare legitimate normalization.
- No retuning on the final test, updating historical reports to the new metric, overwriting `b1`, or auto-publishing artifacts.

## Background / Context

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

## Features / Tasks

### S0: Freeze interfaces, provenance, and run ownership
**Files:** `actproto/experiments/contracts.py` (new), `actproto/experiments/__init__.py` (new), `actproto/experiments/manifest.py` (new), `actproto/configs/experiment_v2.yaml` (new), `tests/test_experiment_contracts.py` (new), `docs/EXPERIMENT_CONTRACT.md` (new), `docs/ACTIVE_PLANS.md` (new)
**Complexity:** M
**Depends on:** none
**Estimate:** 3–5 engineering hours

Define versioned TaskSpec, PolicyInput, ExecutionTrace, VerificationResult, run manifest, and immutable artifact identifiers. Specify adapters for existing Task/EventLog objects rather than mutating their meaning. Include task mode, acceptable answer normalization, tool-state dependencies, budgets, group IDs, environment/tool-fixture hash, schema version, completion reason, trace provenance, and generated-token ownership. Keep oracle fields out of PolicyInput serialization. Specify selection thresholds and dataset quotas from the shared contract in configuration before results exist. Implement a shared process-safe GPU lease in the manifest utilities, keyed by device UUID with PID/start-time ownership, heartbeat and conservative stale-owner recovery; never steal a live lease. Artifact writers fail on occupied run directories unless a validated resume token is supplied; resume must match config/data/model hashes. Create docs declaring plans 9 then 10 authoritative.

**Acceptance:** Round trips preserve private/public boundaries; unknown contract versions fail clearly; duplicate run IDs cannot overwrite files; manifest hashes change when fixtures, templates, or verifier change. Snapshot protected artifact hashes without copying large weights.

### S1: Execution-grounded verification and trustworthy metrics
**Files:** `actproto/evaluation/verification_v2.py` (new), `actproto/evaluation/metrics_v2.py` (new), `actproto/evaluation/trace_adapter.py` (new), `tests/test_verification_v2.py` (new), `tests/test_metrics_v2.py` (new)
**Complexity:** L
**Depends on:** S0
**Estimate:** 5–8 engineering hours

Implement a pure verifier over TaskSpec and trusted harness trace. Do not accept model-supplied tool JSON as executed observations. Verify tool names, schema and semantic values, actual tool success or task-declared expected error, finish timing, dependencies, and final outcome. Respect legitimate no-tool tasks and alternative valid trajectories. Numeric tolerance and output formatting belong to the task contract; IDs and sentinel strings remain exact where required. Do not use arbitrary substring acceptance. Keep historical scoring untouched.

Measure exact duplicate `(tool, canonical args, relevant state)` calls and cycles separately from useful repeated use. Count invalid parsing, failed execution, noncompliant calls, model invocations and generated tokens, not just tool names. Excess-call metrics require a valid minimum/reference bound and must be labeled unavailable when no sound bound exists. Capture task compliance, answer accuracy, first-tool and argument metrics without conflating them.

**Acceptance:** Finish-only known answer fails required-tools tasks; forged output, wrong order, incorrect arguments, stale state, wrong numeric units, and truncation fail. Legal repeated database calls, alternative valid ordering, expected errors, and no-tool answers pass. Run adversarial verifier fixtures and cross-check oracle-generated traces; no historical result changes.

### S2: Leakage-resistant, balanced, fully specified curriculum
**Files:** `actproto/data/benchmark_v2.py` (new), `actproto/data/splits_v2.py` (new), `actproto/data/curriculum_v2.py` (new), `actproto/data/fixtures_v2.py` (new), `scripts/build_benchmark_v2.py` (new), `tests/test_benchmark_v2.py` (new), `docs/DATASET_V2.md` (new)
**Complexity:** L
**Depends on:** S0
**Estimate:** 6–9 engineering hours

Generate the contract's target counts with fixture-backed tools, varied products/years/records/arithmetic, meaningful data dependencies, distractor catalogs, subset execution, and bounded recoverable failures. Include explicit output convention in every format, placing it before the execution line when header-last must end on that line. Do not train a parser by merely listing tools in execution order for all tasks; include natural task descriptions requiring selection and observation-dependent arguments. Expand fixture banks in a separate versioned environment rather than changing old global fixtures. Pass environment identifiers into execution.

Compute logical group and normalized prompt/call fingerprints before splitting. Keep semantic duplicates and paraphrases together. Hold out argument tuples/entities for IID-new-entity evaluation, and entire composition or language-template families for stress evaluation; report these axes separately rather than claiming both from random seeds. Construct explicit manifests and private oracles; never rely on `split='train'` from the old curriculum factory. Build a reject log for invalid/ambiguous tasks. Replay every gold trace through the real deterministic tool functions before accepting the dataset. Full S1 verifier validation happens at integration.

Train-only failure repair: later collect 100 fixed train-probe tasks, classify tool/argument/termination errors, add at most 120 verified variants within their train groups, and preserve at least 50% original balanced samples in the repair mix. Validation/test errors may inform reports but never supply repair examples. Avoid oversampling easy calculator/finish tokens. Preserve legacy 288/104/104 exactly.

**Acceptance:** Exact and normalized prompt overlap is zero across splits; logical groups do not cross splits; held-out composition/family constraints pass; all gold tool results replay; header variants state output rules; quota and unique-group counts are reported honestly. Immutable manifest contains seed, split strategy, fixture hash, rejected counts, and per-stratum depth distribution.

### S3: Correct masking and validation-selected SFT checkpoints
**Files:** `actproto/training/sft_v2.py` (new), `actproto/training/sft_masks.py` (new), `actproto/configs/sft_v2.yaml` (new), `scripts/train_sft_v2.py` (new), `tests/test_sft_v2.py` (new), `tests/test_sft_masks.py` (new)
**Complexity:** L
**Depends on:** S0
**Estimate:** 5–8 engineering hours

Add a new SFT entry point; leave historical train_lora defaults untouched until documentation marks them historical. Validate the selected named Tiny Aya template and insert generation spans covering assistant call content plus its actual response/end-of-turn delimiters. Compare rendered bytes before and after adding Jinja markers. Verify real token masks, not the existence of marker strings: user/system/tool and padding labels are -100; assistant calls/finish and intended termination labels are trained. Detect empty masks, wrong alignment, and truncated assistant turns before model training. Exclude policy-invisible private fields.

Start from the pinned base model for newly corrected SFT candidates; preserve the old assistant-only adapter as a control. Default rank 16, alpha 32, dropout .05, lr 2e-4, batch 1, grad accumulation 8, maximum two epochs. Save epoch checkpoints plus training state. Inject a validation callback using the shared evaluation runner after it lands; choose by macro verified success, then argument correctness, then fewer invalid calls, then fewer tokens. Do not select on token accuracy alone. Respect configured sequence length; detect overflow rather than silently dropping finish. Bucket long examples and report excluded counts. Use a real quantization config only when selected and supported; otherwise record bf16, never claim QLoRA from an ignored flag.

**Acceptance:** Tiny mock-model tests establish masking/gradient ownership, callback selection and checkpoint reload; real-tokenizer preflight establishes template equality and token labels before a GPU run. Missing data, marker failure, empty masks, and occupied output paths produce nonzero exit codes. Dry run never loads model weights; real mode must save a trained adapter and completion manifest.

### S4: One evaluation runner with controlled inference settings
**Files:** `actproto/evaluation/experiment_runner.py` (new), `actproto/agent/inference_v2.py` (new), `actproto/agent/trace_v2.py` (new), `scripts/run_experiment_v2.py` (new), `actproto/configs/sft_matrix.yaml` (new), `tests/test_experiment_runner.py` (new), `tests/test_trace_v2.py` (new)
**Complexity:** L
**Depends on:** S1, S2, S3
**Estimate:** 5–8 engineering hours

Wire manifests, fixtures, verifier, saved checkpoints, and policy trace collection into one runner used by SFT validation and later RL evaluation. Primary mode disables routing, heuristic arguments and answer repair; schema validation and single-call stopping remain common. An optional deployed-system mode enables explicitly named interventions and logs every intervention, original model output, call source and additional generation. Invalid output must remain visible, not become a fabricated valid finish. No oracle values enter prompts or fallback generation. Enforce per-task budgets and versioned task environments. Use the same tokenizer/template contract for every candidate.

Expose `--manifest`, `--split`, `--run-id`, `--matrix`, `--adapter`, `--mode`, `--dry-run`, and an explicit `--final-eval` unlock backed by the frozen selection manifest. Training callbacks are forbidden from loading the test split. Seal raw traces plus aggregate metrics in separate result directories. Hash/run manifests include GPU, dtype, generation parameters, interventions, wall-clock timings, generated tokens and inference-engine versions.

**Acceptance:** Fake policy runs verify budget caps, multiple calls rejected, full traces, expected failures and private-field exclusion. Exact same fixtures and scoring are used by validation and CLI. Old suite still passes. Known finish-only exploit fails end to end. Final test cannot be selected through a training callback or accidental default.

### S5: Measure and calibrate optional first-hop routing
**Files:** `actproto/activations/threshold_v2.py` (new), `actproto/activations/router_calibration.py` (new), `scripts/calibrate_router_v2.py` (new), `actproto/configs/router_v2.yaml` (new), `tests/test_router_calibration.py` (new)
**Complexity:** M
**Depends on:** S4
**Estimate:** 3–5 engineering hours

Fit prototypes and a linear-probe control using train activations from the exact selected adapter and task/prompt contract. Tune thresholds only on validation, recording risk/coverage and per-family confusion. Among equally accurate thresholds meeting coverage, prefer stricter margin then score; report coverage tradeoffs explicitly. Hash-bind the bank to adapter, layer, tokenizer and template. Reject mismatched banks rather than loading silently. Default remains first-hop only: no claim about multi-hop routing from first-hop labels.

Compare no router, routed with semantic fallback disabled, and routed deployed-system mode. Keep prototype routing only if validation justifies it under the shared success/latency gate; otherwise retain it as a negative result. Do not fit on test prompts or on validation labels. Parameter calibration belongs after candidate adapter selection in S7; this task implements/tests calibration, not a GPU run.

**Acceptance:** Strict tie fixture selects stricter thresholds; manifest mismatch fails; validation-only calibration and train-only fitting are enforced; failed gate yields router-off selection rather than an undocumented retune. Coverage and fallback frequency are recorded.

### S6: Real AuditKIT evaluation and artifact export
**Files:** `actproto/integrations/__init__.py` (new), `actproto/integrations/auditkit_v2.py` (new), `scripts/check_toolkit_compat.py` (new), `scripts/export_experiment_v2.py` (new), `requirements-auditkit.lock` (new), `tests/test_auditkit_v2.py` (new), `tests/test_export_experiment_v2.py` (new), `docs/TOOLKIT_COMPATIBILITY.md` (new)
**Complexity:** M
**Depends on:** S4
**Estimate:** 4–6 engineering hours

Pin a tested AuditKIT revision and resolve it in an isolated environment. Inspect real signatures; use its supported recorded-episode/custom-predicate evaluation path with the shared verifier, not an import-only shim. Write toolkit version and actual invocation in the run manifest. Compare local and AuditKIT verdicts on successful, failing, alternative-path and adversarial episodes. An explicit `local` backend can support development; submission mode fails if AuditKIT is missing. No silent fallback labeled AuditKIT.

Export from a manifest allowlist: selected adapters and tokenizer/template, prototype/threshold pair if selected, data and fixture manifests, generated curriculum JSONL, traces, metrics, environment lock and model card. Separate public training data from final answers until the joint evaluation is complete. Reject missing files, mismatched hashes, private credentials, caches and unrelated logs. List-only is default; produce a local bundle before any separately authorized HF upload. Document exact toolkit compatibility and the complete data→train→eval commands.

**Acceptance:** Integration test actually calls installed AuditKIT and verifies verdict parity; unit mocks alone do not satisfy this task. Bundle dry run contains every selected dependency and reloads locally; missing artifact or unavailable submission backend fails loudly. Preserve the existing environment and historical exporter behavior.

### S7: Bounded SFT experiments, error analysis, and checkpoint freeze
**Files:** `scripts/run_sft_study.py` (new), `scripts/compare_experiments_v2.py` (new), `actproto/evaluation/statistics_v2.py` (new), `tests/test_statistics_v2.py` (new), `docs/SFT_V2_RESULTS.md` (new), `docs/SFT_BASELINE_MANIFEST.json` (new)
**Complexity:** L
**Depends on:** S5, S6
**Estimate:** 4–6 engineering hours plus measured GPU runtime

First CPU integration gate, then real tokenizer preflight, then a two-update GPU smoke with changed adapter hashes and finite loss, then full runs serially. Use one GPU lease for all train/eval/calibration work. Record PID, heartbeat, logs, output path, time/token budget and expected artifacts; monitor until completion and verify outputs before declaring done. Do not kill unrelated jobs or start a duplicate when a lease is held. Resume only compatible incomplete runs.

Validation study: Ebase pinned base model with no adapter on the same corrected harness; E0 old assistant-only checkpoint on corrected harness, router off; E1 new corrected-mask SFT on original train trajectories; E2 corrected-mask SFT on balanced v2 train; E3 optional bounded train-only failure repair of E2, only if E2 has recoverable train errors and budget remains. Record total examples, tokens, steps and GPU time for every candidate; E1→E2 is not compute-matched and must be described accordingly. Primary seed 42; replicate selected SFT candidate on seeds 43 and 44 only if the time budget permits. No rank sweep or unlimited retries. Use a maximum of 24 total GPU-hours for this study, with completion deadlines recorded at launch; first benchmark ten minutes to estimate feasibility, shorten candidate list before beginning if needed. Runtime is unknown until measured.

Use validation, never final test, to select adapter and optional router mode under the predeclared rule. Evaluate old 104-task set only as regression/development diagnostics. Run first-hop calibration on the selected adapter; report prototype and probe results together. Write per-stratum results, errors, confidence intervals, interventions, failure costs and limitations. Freeze checkpoint, train/val hashes, final-test hash, verifier, template, dependency locks, and primary router-off config in SFT_BASELINE_MANIFEST. The chosen candidate may be E0 if refinements do not help. Freeze also identifies the nominal RL reference policy as this SFT checkpoint.

**Acceptance:** Trained weights reload and produce a real tool trace; data/model/config hashes and selection reasoning are complete; all failed/aborted runs remain labeled; protected artifact hashes unchanged. SFT performance target is tested, not asserted. Final-test isolation is preserved. Plan 10 starts optimization only from this immutable handoff.

### S8: Submission-ready SFT narrative and reproducibility handoff
**Files:** `README.md` (modify), `docs/METHOD.md` (modify), `docs/SUBMISSION.md` (modify), `docs/EXPERIMENT_STATUS.md` (modify), `docs/SFT_REPRODUCTION.md` (new)
**Complexity:** M
**Depends on:** S7
**Estimate:** 2–3 engineering hours

Lead with the actual trained objective and measured validation/development scope. Label historical metrics, contaminated diagnostics and new strict metrics separately. State whether routing helped and what remains unmeasured. Replace stale running-state headers with verified current handoff information; retain historical chronology. Document exact reproducible commands using the new manifest paths, extraction of required splits, environment locks, public bundle contents, and Plan 10's consumption of the frozen checkpoint. Do not claim RL or distillation. Record which Plan 1 ideas were adopted, replaced or deferred. No automatic publishing.

**Acceptance:** A reviewer can identify selected checkpoint, data, verifier, metrics, and next execution step without reading historical transcripts. Command dry runs validate every referenced file/config and avoid historical artifact paths. Handoff identifies final-test access state and pending RL experiments.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| AuditKIT, revision resolved and pinned by S6 | Real recorded-episode evaluation | Required for submission mode; local development explicit |
| Existing torch/transformers/PEFT/TRL/datasets | Supervised training | Existing; record tested exact versions |
| bitsandbytes | Optional quantized loading if memory requires it | Optional; platform compatibility gate, no unverified install requirement |

No new hosted model service. Implement grouped bootstrap with existing NumPy. Upstream interfaces are moving: verify against the pinned install rather than copying assumed signatures. The [AuditKIT repository](https://github.com/Lexsi-Labs/AuditKIT) documents recorded-agent evaluation; the [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer) is the implementation reference to verify during S3.

## File Change Summary

| File | Action | Owner |
| --- | --- | --- |
| `actproto/experiments/contracts.py` | new | S0 |
| `actproto/experiments/__init__.py` | new | S0 |
| `actproto/experiments/manifest.py` | new | S0 |
| `actproto/configs/experiment_v2.yaml` | new | S0 |
| `tests/test_experiment_contracts.py` | new | S0 |
| `docs/EXPERIMENT_CONTRACT.md` | new | S0 |
| `docs/ACTIVE_PLANS.md` | new | S0 |
| `actproto/evaluation/verification_v2.py` | new | S1 |
| `actproto/evaluation/metrics_v2.py` | new | S1 |
| `actproto/evaluation/trace_adapter.py` | new | S1 |
| `tests/test_verification_v2.py` | new | S1 |
| `tests/test_metrics_v2.py` | new | S1 |
| `actproto/data/benchmark_v2.py` | new | S2 |
| `actproto/data/splits_v2.py` | new | S2 |
| `actproto/data/curriculum_v2.py` | new | S2 |
| `actproto/data/fixtures_v2.py` | new | S2 |
| `scripts/build_benchmark_v2.py` | new | S2 |
| `tests/test_benchmark_v2.py` | new | S2 |
| `docs/DATASET_V2.md` | new | S2 |
| `actproto/training/sft_v2.py` | new | S3 |
| `actproto/training/sft_masks.py` | new | S3 |
| `actproto/configs/sft_v2.yaml` | new | S3 |
| `scripts/train_sft_v2.py` | new | S3 |
| `tests/test_sft_v2.py` | new | S3 |
| `tests/test_sft_masks.py` | new | S3 |
| `actproto/evaluation/experiment_runner.py` | new | S4 |
| `actproto/agent/inference_v2.py` | new | S4 |
| `actproto/agent/trace_v2.py` | new | S4 |
| `scripts/run_experiment_v2.py` | new | S4 |
| `actproto/configs/sft_matrix.yaml` | new | S4 |
| `tests/test_experiment_runner.py` | new | S4 |
| `tests/test_trace_v2.py` | new | S4 |
| `actproto/activations/threshold_v2.py` | new | S5 |
| `actproto/activations/router_calibration.py` | new | S5 |
| `scripts/calibrate_router_v2.py` | new | S5 |
| `actproto/configs/router_v2.yaml` | new | S5 |
| `tests/test_router_calibration.py` | new | S5 |
| `actproto/integrations/__init__.py` | new | S6 |
| `actproto/integrations/auditkit_v2.py` | new | S6 |
| `scripts/check_toolkit_compat.py` | new | S6 |
| `scripts/export_experiment_v2.py` | new | S6 |
| `requirements-auditkit.lock` | new | S6 |
| `tests/test_auditkit_v2.py` | new | S6 |
| `tests/test_export_experiment_v2.py` | new | S6 |
| `docs/TOOLKIT_COMPATIBILITY.md` | new | S6 |
| `scripts/run_sft_study.py` | new | S7 |
| `scripts/compare_experiments_v2.py` | new | S7 |
| `actproto/evaluation/statistics_v2.py` | new | S7 |
| `tests/test_statistics_v2.py` | new | S7 |
| `docs/SFT_V2_RESULTS.md` | new | S7 |
| `docs/SFT_BASELINE_MANIFEST.json` | new | S7 |
| `README.md` | modify | S8 |
| `docs/METHOD.md` | modify | S8 |
| `docs/SUBMISSION.md` | modify | S8 |
| `docs/EXPERIMENT_STATUS.md` | modify | S8 |
| `docs/SFT_REPRODUCTION.md` | new | S8 |

## Open Questions

- Dataset quotas are targets: uniqueness and meaningful tool dependencies take priority over repeated synthetic padding. S2 must report actual counts and freeze revisions before experiments.
- Training speed and memory are unknown; use the measured pilot to prioritize Ebase/E0/E1/E2. E3, extra seeds and optional router variants are the first cuts, never verifier checks or the primary matched baseline.
- Tiny Aya template end delimiters must be inspected in the pinned tokenizer. Keep rendered text unchanged except generation markers; if tokenizer behavior defeats that requirement, document and test an explicit template migration for all candidates.
- The actual AuditKIT API may differ from repository prose. Resolve in S6 with a real import/execution test; do not turn it into another compatibility-only claim.
- Performance failure does not block creating an honest baseline. Missing verification, an unreadable checkpoint, or absent provenance does block the RL handoff.

## Execution priority and handoff

Must finish S0–S4, S6, Ebase/E0/E1/E2 from S7, and S8 before treating SFT as established. S5's implementation remains small; GPU router ablations are optional if deadlines tighten. Plan 10 consumes S0–S6 APIs and S7's frozen artifacts; no cross-plan GPU concurrency. Final evaluation is coordinated with Plan 10, not used to iterate this plan.
