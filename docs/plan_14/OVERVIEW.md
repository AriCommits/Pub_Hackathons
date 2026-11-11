# Plan 14: Audit remediation (security + cleanliness), file-clustered

## Summary

Close the Medium security gaps and the highest-ROI cleanliness/misconfig issues from `.opencode/audits/arch/audit_1.md` and `.opencode/audits/cleanliness/audit_1.md` without a big-bang rearchitecture. Work is ordered as **file clusters**: every issue that touches the same files is done in one task so we do not revisit `train_lora.py`, `trial.yaml`, export scripts, or the doc handoff set multiple times. End state: contained path/secret handling shared by HF export and trial resolve; UUID-gated GPU pin for LoRA/eval as well as GRPO; historical GRPO configs archived and fail-closed; live-looking stubs made obviously dead; contaminated evals labeled; one current STATUS handoff. Splitting `grpo_run.py` and relocating the entire legacy stack under `legacy/` are deferred.

## Goals

- HF export uses the same content secret scan as export-v2 (including `resume-token.txt`); list and upload both refuse on hit.
- Trial/CLI path resolution cannot escape the project root; absolute and `../` paths fail closed.
- Resume tokens are mode `0600` and stay out of exportable artifact lists.
- LoRA/eval GPU selection reuses the GRPO UUID helpers (or a thin shared wrapper), not “largest memory.”
- `.gitignore` covers credentials/pickle names; probe save format is honest (or gitignored).
- Only two live GRPO YAMLs remain under `actproto/configs/`; historical e2/tools require `--allow-historical` (or live outside the default load path).
- `train_aligned.py` / legacy `grpo.py` cannot look like successful trainers; non-operative YAML is clearly reference-only.
- Default LoRA/output paths no longer overwrite `results/adapters/b1`; trial export points at the frozen baseline adapter.
- Order/format curriculum evals are labeled diagnostics, not generalization.
- One canonical `docs/STATUS.md` (or rewritten `EXPERIMENT_STATUS.md` lead section) matches live code; stale ACTIVE_PLANS / stub-GRPO claims are corrected or archived.
- Integer schema validation rejects Python `bool`; calculator has a length/node budget.

## Non-Goals

- Splitting `grpo_run.py` / `rl_rollouts.py` into multiple modules (cleanliness god-module item).
- Moving the whole B0–B2 tree under `actproto/legacy/` or deleting the dual stack.
- Replacing host UUID hardcoding with a full multi-machine device registry (only a gitignored local override path is in scope).
- Encrypting resume tokens or integrating an external secrets manager beyond file mode + export bans.
- Changing GRPO reward math, launching GPU jobs, or re-running contaminated curriculum evals as science claims.
- Rewriting all plan_1–13 docs; archive/banner is enough.
- Supply-chain hash-pinning of every lockfile transitive (document authoritative locks only).

## Background / Context

Audits landed on branch `cursor/lexsi-security-cleanliness-audit-8ccb` (PR #6). Security: no Critical; Medium issues are export scanning, path escape, non-UUID GPU pin, gitignore gaps. Cleanliness: Critical operational issues are GRPO YAML sprawl, contradictory docs, contaminated evals; High are stubs, overwrite defaults, hardcoded Windows paths.

Reuse, do not reimplement: `export_experiment_v2` secret regexes; `resolve_output` / `guard_gpu_output` containment ideas; `select_5060_ti` / `rl_device.py`; `HISTORICAL_RUNS` in `grpo_run.py`; frozen adapter identity in `docs/SFT_BASELINE_MANIFEST.json`.

## Features / Tasks

Tasks are **clusters**. Each cluster lists every audit finding it closes. Implement one cluster per PR/commit batch when possible; do not start a second cluster that re-opens the same files until the first is merged.

### C1: Shared secret scan + HF export + resume-token hygiene
**Closes:** Security 1, Security 2 (export side + file mode)  
**Files:** `actproto/utils/secret_scan.py` (new), `scripts/export_experiment_v2.py` (modify — call shared helper), `scripts/export_hf_artifacts.py` (modify), `scripts/train_sft_v2.py` (modify), `scripts/run_experiment_v2.py` (modify), `tests/test_export.py` (modify), `tests/test_export_experiment_v2.py` (modify if present / extend)  
**Complexity:** M  
**Depends on:** none

Extract content + basename secret checks from export-v2 into a shared module. HF export must call the same scanner for list and upload. Ban `resume-token.txt` by name. When writing resume tokens, set `0o600`. Do not relocate the token file in this cluster (optional follow-up); export ban + mode is the acceptance bar.

**Acceptance:** A metrics JSON containing a fake `hf_…` / `api_key` string fails both exporters. `resume-token.txt` is refused. Existing export-v2 tests still pass via the shared helper.

---

### C2: Path containment for trial resolve + CLI writers
**Closes:** Security 4  
**Files:** `actproto/configs/trial_schema.py` (modify), `actproto/utils/paths.py` (new, optional if kept next to trial_schema), `scripts/build_prototypes.py` (modify), `scripts/export_hf_artifacts.py` (modify — same PR as C1 if still open, else follow-up on this file only), `tests/test_trial_config.py` (modify), `tests/test_config.py` (modify as needed)  
**Complexity:** M  
**Depends on:** C1 preferred if `export_hf_artifacts.py` is still dirty; otherwise independent

After resolve, require `path.is_relative_to(project_root)` (or equivalent under 3.10). Reject absolute paths and `..` segments in trial schema and in scripts that take `--prototypes` / `--metrics` / export path args. Mirror the spirit of `resolve_output` without forcing all trial artifacts under `results/rl_v2`.

**Acceptance:** `export.adapter: ../.hf_token` and absolute `/tmp/out` fail. Relative in-tree paths still work. Unit tests cover both.

---

### C3: Device UUID pin + LoRA default output (one pass on `train_lora.py`)
**Closes:** Security 3, Cleanliness 6 (overwrite), Cleanliness 7 (Windows path in study script)  
**Files:** `actproto/utils/devices.py` (modify), `scripts/train_lora.py` (modify), `actproto/training/lora_config.py` (modify), `actproto/configs/sft.yaml` (modify), `actproto/configs/trial.yaml` (modify — paths + export.adapter only; eval-label edits wait for C5), `scripts/run_sft_study.py` (modify), `tests/test_rl_device.py` or new `tests/test_devices.py` (modify/new)  
**Complexity:** M  
**Depends on:** none (parallel with C1)

Replace largest-memory pinning with the same UUID selection path GRPO uses (`select_5060_ti` / nvidia UUID CSV). Fail closed when the expected UUID is absent. Change default LoRA `output_dir` away from `results/adapters/b1` to a non-baseline name (e.g. `results/adapters/lora_run` or require `--output-dir`). Point `trial.yaml` `export.adapter` at the frozen baseline from `SFT_BASELINE_MANIFEST.json` (or documented e2 / assistant-only path). Replace the hardcoded `C:\Users\arian\...` adapter path in `run_sft_study.py` with a repo-relative resolve.

**Acceptance:** With mocked nvidia rows, LoRA pin selects the 5060 Ti UUID ordinal. Missing UUID raises. Default train does not write into `results/adapters/b1`. Study script resolves adapters from repo root.

---

### C4: Gitignore + probe artifact honesty + schema/calculator hardening
**Closes:** Security 5, 7, 8 (np.load flag if touching those files), 9, 12  
**Files:** `.gitignore` (modify), `actproto/activations/linear_probe.py` (modify), `scripts/train_probe.py` (modify if path/extension), `actproto/activations/prototypes.py` (modify), `actproto/activations/store.py` (modify), `actproto/tools/schemas.py` (modify), `actproto/tools/calculator.py` (modify), `tests/test_tools.py` (modify), `tests/test_linear_probe.py` (modify if present)  
**Complexity:** S–M  
**Depends on:** none (parallel with C1–C3)

Add `credentials.json`, `secrets.yaml`, `secrets.json`, `*.pkl`, `*.joblib`, common key patterns to `.gitignore`. Either save probes with real joblib/`skops` or rename extension to `.pkl` and document pickle risk; always gitignore. Pass `allow_pickle=False` on `np.load`. Reject `bool` for integer schema fields (`type(v) is int`). Cap calculator expression length and AST node count.

**Acceptance:** `validate_args(..., {"year": True})` fails. Huge calculator strings error before deep eval. Probe artifacts are ignored by git. `np.load(..., allow_pickle=False)` in store/prototypes.

---

### C5: GRPO config quarantine + historical output guard + stub CLIs
**Closes:** Cleanliness 1, 4 (stubs), 6 (prefix historical), Security 10 (doc alignment deferred to C6 but YAML comments updated here)  
**Files:** `actproto/configs/grpo_v2_gpu_e2.yaml` (move), `actproto/configs/grpo_v2_gpu_tools.yaml` (move), `actproto/configs/archive/` (new), `actproto/configs/experiment_v2.yaml` (move to `configs/reference/` or `docs/`), `actproto/configs/reward_v2.yaml` (move or banner), `scripts/train_grpo.py` (modify), `actproto/training/grpo_run.py` (modify — `HISTORICAL_RUNS` add `grpo_gpu_prefix` only after study dir is renamed in live YAML, or require new output_dir in live study YAML), `actproto/training/grpo.py` (modify — clearer module docstring / rename import surface), `scripts/train_aligned.py` (modify), `scripts/run_eval.py` (modify — banner/exit), `README.md` (modify GRPO section only; full STATUS rewrite in C6)  
**Complexity:** M  
**Depends on:** none (parallel with C1–C4; coordinate README with C6)

Move historical GPU YAMLs to `actproto/configs/archive/`. Live configs: CPU smoke + GPU smoke + one study (`grpo_v2_gpu_prefix.yaml` or rename to `grpo_v2_gpu_study.yaml`). Loader/CLI: `--train` without an explicit study config must not silently default to CPU in a way that looks like GPU train; require `--config` for `--train --live-gpu` and document defaults. Refuse loading archive paths unless `--allow-historical`. Add current study dirname to `HISTORICAL_RUNS` **after** changing the live study `output_dir` to a new name (same commit). `train_aligned.py`: drop `or True`; exit 2 unless `--plan-only`. Legacy `grpo.py` stays raise-only but module docstring states “not the trainer.” `run_eval.py` prints that live eval is `run_trial` / `run_experiment_v2` and exits non-zero without an explicit legacy flag if that matches current behavior goals.

**Acceptance:** `train_aligned.py` without `--plan-only` exits 2. Archive GRPO YAML is not the default. New study output_dir is not in the old historical set until protected. README GRPO bullets list Smoke / Study / Historical only.

---

### C6: Contaminated-eval labels + single STATUS handoff
**Closes:** Cleanliness 2, 3, 11 (README leftovers), Security 10–11 (docs)  
**Files:** `docs/STATUS.md` (new — canonical), `docs/EXPERIMENT_STATUS.md` (modify — point to STATUS or replace lead), `docs/ACTIVE_PLANS.md` (modify), `docs/SUBMISSION.md` (modify), `docs/METHOD.md` (modify only if it still claims GRPO stub), `docs/AUDIT_2026-11-04.md` (modify — stale banner), `docs/RL_BACKEND_COMPATIBILITY.md` (modify — multiple `use_cpu: false` profiles), `docs/plan_14/OVERVIEW.md` (this file), `README.md` (modify — handoff link + GRPO if not finished in C5), `scripts/run_order_eval.py` (modify), `scripts/run_format_eval.py` (modify), `actproto/configs/trial.yaml` (modify — run id comments / names for order/format diagnostics; if C3 already edited this file, do labels in the same C3 PR or a tight follow-up that only touches run metadata)  
**Complexity:** M  
**Depends on:** C5 for README GRPO list; C3 if `trial.yaml` still open

Mark `B2ordereval` / `B2fmteval` (and CLI help) as `diagnostic_train_echo`. Reports must say they are not held-out generalization. Write `docs/STATUS.md` with: frozen SFT baseline pointer, GRPO negative/study status, current commands, “do not start X” only if actually true. ACTIVE_PLANS becomes a pointer to STATUS + plan_14. Banner the Nov 4 audit as historical (GRPO stub claim obsolete). Redact or relativize personal Windows paths in docs touched here.

**Acceptance:** A new agent reading only STATUS + README knows the live train/eval commands and that order/format evals are diagnostics. No top-of-tree doc claims GRPO is unimplemented.

---

### C7 (optional follow-up): Host device override file
**Closes:** remainder of Cleanliness 7 / Security 11  
**Files:** `actproto/configs/local.device.yaml.example` (new), `.gitignore` (modify — `local.device.yaml`), `actproto/training/rl_config.py` (modify), `actproto/utils/devices.py` (modify)  
**Complexity:** S  
**Depends on:** C3, C4

Optional: expected GPU UUID loaded from gitignored `configs/local.device.yaml` with example committed; YAML defaults remain for this host until override exists.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| _(none)_ | Reuse joblib already via sklearn if switching probe dump | optional |

## File Change Summary

| File | Action |
| --- | --- |
| `actproto/utils/secret_scan.py` | new |
| `actproto/utils/paths.py` | new (optional) |
| `actproto/utils/devices.py` | modify |
| `actproto/configs/trial_schema.py` | modify |
| `actproto/configs/trial.yaml` | modify |
| `actproto/configs/sft.yaml` | modify |
| `actproto/configs/grpo_v2_gpu_e2.yaml` | move → archive |
| `actproto/configs/grpo_v2_gpu_tools.yaml` | move → archive |
| `actproto/configs/grpo_v2_gpu_prefix.yaml` | modify (output_dir / name) |
| `actproto/configs/experiment_v2.yaml` | move → reference |
| `actproto/configs/reward_v2.yaml` | move or banner |
| `actproto/configs/archive/` | new |
| `actproto/configs/local.device.yaml.example` | new (C7) |
| `actproto/training/lora_config.py` | modify |
| `actproto/training/grpo_run.py` | modify |
| `actproto/training/grpo.py` | modify |
| `actproto/training/rl_config.py` | modify (C7) |
| `actproto/activations/linear_probe.py` | modify |
| `actproto/activations/prototypes.py` | modify |
| `actproto/activations/store.py` | modify |
| `actproto/tools/schemas.py` | modify |
| `actproto/tools/calculator.py` | modify |
| `scripts/export_hf_artifacts.py` | modify |
| `scripts/export_experiment_v2.py` | modify |
| `scripts/train_sft_v2.py` | modify |
| `scripts/run_experiment_v2.py` | modify |
| `scripts/build_prototypes.py` | modify |
| `scripts/train_lora.py` | modify |
| `scripts/train_grpo.py` | modify |
| `scripts/train_aligned.py` | modify |
| `scripts/run_eval.py` | modify |
| `scripts/run_sft_study.py` | modify |
| `scripts/run_order_eval.py` | modify |
| `scripts/run_format_eval.py` | modify |
| `scripts/train_probe.py` | modify |
| `.gitignore` | modify |
| `README.md` | modify |
| `docs/STATUS.md` | new |
| `docs/EXPERIMENT_STATUS.md` | modify |
| `docs/ACTIVE_PLANS.md` | modify |
| `docs/SUBMISSION.md` | modify |
| `docs/METHOD.md` | modify |
| `docs/AUDIT_2026-11-04.md` | modify |
| `docs/RL_BACKEND_COMPATIBILITY.md` | modify |
| `docs/plan_14/OVERVIEW.md` | new |
| `tests/test_export.py` | modify |
| `tests/test_trial_config.py` | modify |
| `tests/test_tools.py` | modify |
| `tests/test_devices.py` / `test_rl_device.py` | new/modify |

## Execution order (minimize file thrash)

```text
Parallel wave 1:  C1  |  C3  |  C4
Wave 2:           C2  (after C1 if export_hf still open; else parallel)
Wave 3:           C5
Wave 4:           C6  (STATUS + eval labels; trial.yaml only if C3 done)
Optional:         C7
```

If two clusters would both edit the same file, merge into the earlier cluster rather than opening a second PR on that file.

## Open Questions

1. Should the live study YAML be renamed `grpo_v2_gpu_study.yaml`, or keep `grpo_v2_gpu_prefix.yaml` and only change `output_dir`?
2. Frozen export adapter: exact path from `SFT_BASELINE_MANIFEST.json` (e2) vs `b1_assistant_only` — confirm which is submission-canonical before C3 edits `trial.yaml`.
3. Is C7 (local device YAML) wanted now, or is UUID-in-repo acceptable until a second machine appears?
4. Proceed to `parallelize-plan` on this OVERVIEW for sprint breakdown, or implement wave 1 directly?
