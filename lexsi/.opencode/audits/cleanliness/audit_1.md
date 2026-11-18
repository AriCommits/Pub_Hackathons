# Cleanliness / maintainability audit — `lexsi/`

**Date:** 2026-11-11 (America/Chicago)  
**Scope:** `/lexsi` in `AriCommits/Hackathons`  
**Skill:** `AriCommits/DevUtils` → `skills/code-review` (strict maintainability / cleanliness)  
**Focus:** misconfiguration, live-looking stubs, config sprawl, contradictory docs, overwrite risk

## Summary

The project runs **two parallel stacks** (legacy B0–B2 / `trial.yaml` vs Plan 9–13 / `*_v2`) with live-looking configs and docs that still describe obsolete paths. Operational risk is concentrated in **GRPO YAML sprawl**, **silent/stub CLIs**, **contaminated curriculum evals**, **overwrite-prone defaults**, and **docs that can send an operator into the wrong stack**. Fail-closed GRPO gates are good; discoverability is the main cleanliness failure.

---

## Critical

### 1. Five near-identical GRPO YAMLs, one schema, easy wrong run

**Evidence:** `grpo_v2.yaml` (CPU), `grpo_v2_gpu.yaml` (smoke), `grpo_v2_gpu_e2.yaml` / `grpo_v2_gpu_tools.yaml` (historical; nearly identical), `grpo_v2_gpu_prefix.yaml` (current study). All GPU files share `schema_version: grpo_v2_gpu`. `scripts/train_grpo.py` defaults to the CPU config. README says only `grpo_v2_gpu.yaml` may set `use_cpu: false` while four files do.

**Risk:** Wrong smoke/historical/study profile looks equally “live.”

**Cleaner structure:** One live GPU study YAML + one smoke YAML. Archive e2/tools. Fail load of archive unless `--allow-historical`. Require explicit `--config` for `--train` (no silent CPU default).

---

### 2. Docs that contradict live code

**Evidence:**
- `docs/ACTIVE_PLANS.md` vs completed Plan 9–13 work in tree / `EXPERIMENT_STATUS.md`
- `EXPERIMENT_STATUS.md` opens with a stale “B2 still running / do not start another train” warning, then later records completion and GRPO results; still claims `train_lora.py` exits 2 with a wire-TRL message (false — it calls `run_lora_sft`)
- `docs/SUBMISSION.md` / `METHOD.md` / `AUDIT_2026-11-04.md` still describe GRPO as a non-training stub, while `grpo_run.py` / `train_grpo.py` train
- Large plan-doc corpus (`docs/plan_*`) with duplicated sprint boilerplate

**Risk:** Agents and humans follow the wrong handoff and waste runs or mis-claim status.

**Cleaner structure:** One canonical `STATUS.md`. Archive audits/plans under `docs/archive/`. Align README / SUBMISSION / METHOD with the frozen SFT baseline and the current GRPO command.

---

### 3. Contaminated curriculum evals still look like generalization

**Evidence:** Prior audit: 38/38 order and 266/266 format eval prompts are exact training repeats. `run_order_eval.py` / `run_format_eval.py` and `trial.yaml` still expose `B2ordereval` / `B2fmteval`. Legacy `score_episode` can mark success from finish-answer alone without required tool execution.

**Risk:** Headline metrics / submission claims treat train-echo diagnostics as held-out success.

**Cleaner structure:** Label those runs `diagnostic_train_echo` in CLI/help/reports. Gate submission metrics on `verification_v2` + sealed tooluse_v2 splits only.

---

## High

### 4. Dead / stub entrypoints that look live

| Path | Behavior |
| --- | --- |
| `actproto/training/grpo.py` | Always raises; name looks like the GRPO API |
| `scripts/train_aligned.py` | `return 0 if args.plan_only or True else 2` — always succeeds, never trains |
| `actproto/training/rewards.py` | Only used by `tests/test_rewards.py`; live path is `rewards_v2` |
| `scripts/run_eval.py` | Recorded-mode shim; live eval is `run_trial` / `run_experiment_v2` |
| `actproto/configs/experiment_v2.yaml`, `reward_v2.yaml` | Specs only — no Python loader; editing YAML changes nothing |

**Cleaner structure:** Rename/delete stubs; require `--plan-only` for plan CLIs and exit 2 otherwise; move non-operative YAML to `docs/` or `configs/reference/`.

---

### 5. Duplicate reward / training / eval stacks

Legacy vs current pairs both look current: `rewards`/`rewards_v2`, `grpo`/`grpo_run`, `sft`+`train_lora`/`sft_v2`+`train_sft_v2`, `metrics`/`metrics_v2`, `run_trial`/`run_experiment_v2`. Trainer scores `bounded_prefix_v1` while `score_executed_session` still defaults to `bounded_shaped_v1`.

**Cleaner structure:** Package `legacy/` vs `tooluse_v2/`; one public reward entry; README reproduce path lists one stack.

---

### 6. Overwrite-prone defaults

- `train_lora.py` / `sft.yaml` default to `results/adapters/b1` (can clobber the preserved baseline)
- `HISTORICAL_RUNS` protects e2/smoke/tools but **not** `grpo_gpu_prefix` (current study default dir)
- `trial.yaml` `export.adapter` still points at historical `b1`, not the frozen e2 / assistant-only adapter

**Cleaner structure:** Timestamped/named default output dirs; refuse overwrite of known baselines; add prefix to historical after the study; point export at the frozen manifest adapter.

---

### 7. Machine-hardcoded GPU identity + Windows paths

UUID `134c34ef-…` baked into every GPU YAML + `rl_config.py` / `train_grpo.py`. `scripts/run_sft_study.py` hardcodes `C:\Users\arian\Github\Hackathons\lexsi\results\adapters\b1_assistant_only`.

**Cleaner structure:** Device pin via env / gitignored `configs/local.device.yaml`. Resolve adapters relative to repo root.

---

### 8. God modules / spaghetti dispatch

- `grpo_run.py` ~1084 lines / many mode branches (plan, CPU smoke, GPU smoke, Tiny Aya train, rewards, OOM)
- `rl_rollouts.py` ~946 lines with dual adapters
- `train_grpo.py` nested schema × mode × `--live-gpu` × child-env checks

**Cleaner structure (code judo):** Split into `grpo_plan`, `grpo_smoke_*`, `grpo_train_tiny_aya`, `grpo_rewards`. Flatten CLI with a mode table.

---

## Medium

### 9. Config flag contradictions (`load_in_4bit`)

`default.yaml` / `model.yaml`: `load_in_4bit: false`. `sft.yaml` / `lora_config.py`: `load_in_4bit: true`. `sft_v2.yaml`: `quantization: none`. Status notes bitsandbytes may be absent.

**Fix:** Single source of truth; fail closed if 4-bit requested without bitsandbytes.

### 10. `trial.yaml` run matrix sprawl

15 run IDs; `--run all` only runs B0–B1–B2. No router-off control for the best adapter.

**Fix:** Split legacy vs diagnostic trial files; document that `all` ≠ every key; add matched router-off run.

### 11. Misleading names / comments

`guard_gpu_output` errors say “GPU smoke” but train uses it; README GRPO section mixes smoke/tools/e2/prefix/historical bans in one paragraph.

**Fix:** Rename helpers; split README GRPO into Smoke / Study / Historical.

### 12. Multi-interpreter env not asserted at train time

Three interpreters (supervised, RL CPU, RL GPU) documented, but `train_grpo.py` does not fingerprint TRL/AgentTune before train.

**Fix:** Assert env fingerprint before `--train --live-gpu`.

---

## Low

- Doc/plan repository weight (~57 markdown files under `docs/`).
- Empty package `__init__`s and stretch surface (`steering/`) still listed in layout.
- Status test-count drift vs current `tests/test_*.py` inventory.

---

## Suggested cleaner layout

```text
lexsi/
  actproto/
    legacy/          # B0–B2: generator, trial runner, metrics v1, rewards.py
    tooluse_v2/      # benchmark, verification, rewards, sft, grpo (split modules)
    configs/
      live/          # grpo_smoke.yaml, grpo_study.yaml, sft_v2.yaml, trial.yaml
      archive/       # grpo_*_e2/tools, old sft.yaml
      local/         # gitignored device UUID override
  docs/
    STATUS.md        # only current handoff
    METHOD.md
    archive/         # plans 1–13, AUDIT_*, old EXPERIMENT_STATUS
  scripts/
    supported/       # train_sft_v2, train_grpo, run_experiment_v2, …
    archive/         # train_aligned, run_eval, run_order_eval, …
```

## Highest-ROI hygiene (do these first)

1. Quarantine historical GRPO YAMLs; one live study config; no CPU default for `--train`.
2. Rewrite STATUS / SUBMISSION / METHOD / ACTIVE_PLANS to one timeline; archive the rest.
3. Mark contaminated evals as diagnostics; export only tooluse_v2 + verification_v2 metrics.
4. Split `grpo_run.py`; delete or clearly reject `train_aligned` / legacy `grpo.py` / unused reward v1.
5. Stop defaulting LoRA writes onto `results/adapters/b1`.

## Keep

Fail-closed GRPO gates (`--live-gpu`, masked child, beta=0 enforcement, historical dir refuse for e2/tools/smokes, public-prompt leak check) — these reduce risk. The remaining mess is mostly **discoverability**: too many live-looking paths and contradictory docs.
