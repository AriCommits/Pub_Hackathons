# Security audit — `lexsi/` (ActProto-LoRA)

**Date:** 2026-11-11 (America/Chicago)  
**Scope:** `/lexsi` in `AriCommits/Hackathons`  
**Skill:** `AriCommits/DevUtils` → `skills/security_audit`  
**Focus:** misconfiguration, secret handling, unsafe I/O, tool sandboxing, GPU/config gates

## Summary

No Critical findings. No hardcoded API keys or tokens in the tree. Tool sandboxing, YAML loading, and GRPO GPU fail-closed gates are generally solid. The highest-impact issues are **export/secret scanning gaps**, **path escape in trial/CLI resolution**, **non-GRPO GPU selection by memory instead of UUID**, and **gitignore holes** for common credential/pickle names.

---

## Findings

### 1. Medium — HF export scans filenames only, not content

**Description:** `scripts/export_hf_artifacts.py` refuses upload only when a basename is in `FORBIDDEN_NAMES` (`.env`, `HF_TOKEN`, `.hf_token`, `secrets.yaml`, `credentials.json`) or the suffix is `.env`. It does not scan file contents. By contrast, `scripts/export_experiment_v2.py` uses content regexes, bans `resume-token.txt`, and validates path roles/hashes.

**Exploit / risk:** An HF `--repo` upload can publish secrets embedded in logs, metrics JSON, configs, or misnamed files under `trial.export.*` paths.

**Remediation:** Share `export_experiment_v2`’s content scanner; deny `resume-token.txt`; refuse upload unless a content scan passes (list-only mode included).

---

### 2. Medium — Plaintext resume tokens written into run directories

**Description:** `scripts/train_sft_v2.py` and `scripts/run_experiment_v2.py` write `resume-token.txt` next to run artifacts. Comments warn not to print/commit the file, but the file remains on disk with default permissions. HF export does not forbid this name.

**Exploit / risk:** Anyone who can read the run directory can resume/overwrite that run. Shared disks, backups, or accidental packaging leak that capability.

**Remediation:** Keep the token outside the artifact tree (or encrypt); set mode `0600`; extend HF forbid-list; prefer a secrets store / prompt for resume.

---

### 3. Medium — Non-GRPO GPU pin ignores UUID (misroute)

**Description:** `scripts/train_lora.py` / `actproto/utils/devices.py` select the CUDA device with the largest `total_memory` and set `CUDA_VISIBLE_DEVICES` to that PyTorch index. Docs (`RL_BACKEND_COMPATIBILITY.md`, plan 11) state nvidia-smi vs PyTorch ordinals can differ and require UUID `134c34ef-…` for the 5060 Ti. GRPO paths enforce UUID; LoRA/eval paths do not.

**Exploit / risk:** LoRA/eval can pin the wrong GPU (OOM on the 4060, or contend with a busy 5060), contradicting the fail-closed GRPO policy.

**Remediation:** Reuse the GRPO UUID selection / masking helpers; refuse start unless the visible UUID matches the host profile.

---

### 4. Medium — `TrialConfig.resolve` / CLI outputs allow path escape

**Description:** Absolute paths and `../` relative paths resolve outside the project root (`actproto/configs/trial_schema.py`). Scripts such as `build_prototypes.py` and `export_hf_artifacts.py` write/read those resolved paths without containment. RL paths use `resolve_output` / `guard_gpu_output`; trial/legacy paths do not.

**Exploit / risk:** Untrusted `trial.yaml` or CLI args can read/write/upload outside the project (for example pointing `export.adapter` at a credential file).

**Remediation:** After resolve, require `is_relative_to(project_root)`; reject absolute/`..` paths in the trial schema; sandbox script outputs like the RL guards.

---

### 5. Medium — `.gitignore` gaps for common secret/artifact names

**Description:** `.gitignore` covers `.env*`, `.hf_token`, `HF_TOKEN`, weight extensions, and most of `results/**`, but not `credentials.json`, `secrets.yaml`, `secrets.json`, `*.pkl`, `*.joblib`, `.aws/`, or private key patterns. `train_probe.py` writes `linear_probe.joblib` via raw `pickle.dumps`.

**Exploit / risk:** Accidental commit of credentials or pickle probes.

**Remediation:** Add those patterns; keep secrets in env/secret managers; optional pre-commit secret scan.

---

### 6. Medium — Dependency / supply-chain surface

**Description:** `pyproject.toml` train extras are broad (`torch>=2.1`, `transformers>=4.40`). RL locks pin AgentTune/AuditKIT to git SHAs and also pull large transitive surfaces (`gradio`, `fastapi`, `chromadb`, `boto3`, `cloudpickle`). Unpinned extras can diverge from locks.

**Exploit / risk:** Wide install surface; trust in GitHub + commit availability; accidental install of unlocked versions.

**Remediation:** Train/GRPO only from lockfiles; pin hashes where possible; document which lock is authoritative per interpreter.

---

### 7. Low — Misleading pickle artifact (`.joblib` that is raw pickle)

**Description:** `actproto/activations/linear_probe.py` saves with `pickle.dumps` to a `.joblib` path. No in-tree `pickle.loads` today.

**Exploit / risk:** Future load of an untrusted probe is RCE; the extension implies joblib safety that is not present.

**Remediation:** Use real `joblib.dump`/`skops`, or a safer format; gitignore `*.joblib`/`*.pkl`; load only from allowlisted trusted paths.

---

### 8. Low — `np.load` without explicit `allow_pickle=False`

**Description:** `prototypes.py` / `store.py` call `np.load` without the flag; `router_calibration.py` sets `allow_pickle=False`. With `numpy>=1.24` the default is already `False`.

**Exploit / risk:** Defense-in-depth gap if constraints loosen.

**Remediation:** Always pass `allow_pickle=False`; validate dtypes/shapes after load.

---

### 9. Low — Schema accepts `bool` as integer

**Description:** `isinstance(True, int)` is `True`, so `validate_args("query_database", {"year": True})` succeeds and the static DB treats `year == 1`.

**Exploit / risk:** Type-confusion / oracle mismatch in the tool sandbox (not RCE).

**Remediation:** Reject bools for integer fields (`type(value) is int`).

---

### 10. Low — Docs vs configs: multiple `use_cpu: false` profiles

**Description:** Docs say `grpo_v2_gpu.yaml` is the only file that may set `use_cpu: false`. Four YAML files do (`grpo_v2_gpu`, `_e2`, `_tools`, `_prefix`), all under schema `grpo_v2_gpu` with loader gates.

**Exploit / risk:** Operator confusion / wrong profile selection (integrity more than confidentiality).

**Remediation:** Update docs to match the loader: any `schema_version: grpo_v2_gpu` file may set `use_cpu: false` under the existing gates; list the four profiles.

---

### 11. Low — Machine-local paths / GPU UUIDs in committed docs and configs

**Description:** Docs and configs embed `C:/Users/arian/...` and host GPU UUIDs.

**Exploit / risk:** Username/path disclosure; UUID hardcoding is intentional fail-closed for one host but misleads other machines if copied blindly.

**Remediation:** Redact personal paths; load expected UUID from an untracked host profile.

---

### 12. Low — Calculator expression size unbounded

**Description:** AST allowlist correctly blocks imports/calls; there is no length/node budget before parse.

**Exploit / risk:** Local DoS of the tool loop via huge expressions.

**Remediation:** Cap expression length and AST node count; optionally bound exponents.

---

## Correctly hardened

- No hardcoded API keys/tokens/passwords; `model.yaml` documents HF auth via env/CLI only.
- No SQL string construction: `query_database` / `lookup_record` filter in-memory fixtures.
- Config YAML uses `yaml.safe_load` only.
- Subprocess calls use argv lists (no `shell=True` / `os.system`).
- Calculator AST whitelist; tool registry is a fixed allowlist with `additionalProperties: False`.
- GRPO device gates: `use_cpu: false` only for `grpo_v2_gpu` schema; 4060 UUID rejected; masked child + `CUDA_VISIBLE_DEVICES`; vLLM rejected; historical output dirs refused for e2/tools/smokes.
- Export v2: path traversal, symlink, private-split, hash, and content secret scanning.
- AgentTune provenance: pinned commit + installed-byte verification.
- Public-prompt / oracle leakage checks on rollout encode paths.

## Priority order

1. Align HF export with export-v2 secret scanning; ban `resume-token.txt`.
2. Contain trial/CLI path resolution under the project root.
3. UUID-pin LoRA/eval GPUs the same way as GRPO.
4. Expand `.gitignore` for credentials/pickle artifacts.
5. Doc/profile cleanup for GRPO `use_cpu` and host UUIDs.
