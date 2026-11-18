# Sprint 1 — Evidence lock

**Plan:** `lexsi/docs/plan_15/OVERVIEW.md`
**Wave:** 1 of 4
**Can run in parallel with:** nobody
**Must complete before:** Sprints 2, 3, and 4

## Before this wave

`lexsi/submission/` does not exist yet. Plans 1–14 stay untouched. Do not train, do not score, do not open the sealed test, and do not edit `docs/plan_15/`.

## Agents in this wave

### Agent A: F1 — Lock the evidence sheet

**Complexity:** M
**Files to create:**
- `lexsi/submission/FACTS.md`

**Depends on:** none
**Blocks:** D1, W1, M1, R1, and everything after them

**Instructions:**

You are writing the only file later agents are allowed to treat as numeric truth. Read these and quote them. Do not browse the web for a better story.

- `lexsi/docs/STATUS.md`
- `lexsi/docs/RESULTS_2026-11-14.md`
- `lexsi/docs/DATASET_V2.md`
- `lexsi/docs/METHOD.md`
- `lexsi/docs/SFT_BASELINE_MANIFEST.json`
- `lexsi/docs/TOOLKIT_COMPATIBILITY.md`
- `lexsi/docs/RL_BACKEND_COMPATIBILITY.md`
- `lexsi/docs/RL_GPU_ENV.md`
- `lexsi/actproto/configs/model.yaml`
- `lexsi/actproto/configs/sft_v2.yaml`
- `lexsi/actproto/configs/trial.yaml`
- `lexsi/docs/SUBMISSION.md` (historical checklist only)

If `lexsi/results/sft_v2/e2-scored` exists, open the metrics JSON and quote `verified_success_v2` (or the actual key) for the validation split, the task count, and any tool-call, unnecessary-call, or latency figures. If the directory is missing, write `raw scored JSON absent` and quote the 200/200 macro `1.0` sentence from the plan-12 / plan-13 prose and from `RESULTS_2026-11-14.md` as committed prose, not as a file you re-measured.

Frontmatter:

```yaml
---
label: internal-facts
track: T3
audience: internal
status: draft
---
```

The body is one table with columns `claim`, `status`, `value`, `source`. `status` is `measured`, `derived`, or `unknown`.

Rows you must include:

- Base model id, revision `af89d219b53ed9b13b8a4645f9c8028973510324`, dtype bf16, quantization off. Gated: yes.
- Selected adapter `results/sft_v2/e2/checkpoint-150`, sha256 `6125195bc713f8c4d9e89070233b65667a8068f8fc684a1e14a2522a26b8d670`, id `e2`, frozen, selection split `val`.
- Manifest `improvement` value `0.7466666666666666`. Status `measured` as a stored field. A reconstructed before-score is `derived` only if the scored JSON defines the subtraction. Otherwise `unknown`.
- Dataset: seed 42, 1200/200/200/200, environment `tooluse_v2.fixtures.1`, English templates. Stress means a held-out template family, not a human language.
- Leak checks named in `DATASET_V2.md`: group, entity, exact prompt, normalized prompt, call fingerprint.
- Kits. AgentTune 1.1.3 is the GRPO stack when installed. AuditKIT 1.1.3 commit `71316da345784ff220d5ed59337441a59ab31526` has a recorded path. State whether the 200/200 score file names AuditKIT or the local verifier. CuratorKIT, AlignTune, and SafeTune are unused. CircuitKIT is unused unless a committed result file shows a layer sweep changed a reported number.
- GRPO: pre-fix reward −0.1 and `template_mismatch`; post-fix `grpo_gpu_study_fix` reward 1.0 and `finish`; groups tied; not a val improvement. Include the step counts from `RESULTS_2026-11-14.md`.
- Hardware: RTX 5060 Ti 16 GB for the model, RTX 4060 8 GB refused. Hours and dollars: `unknown` unless a committed file states them.
- License strings: copy only a license name you found in a file or on the model card you could actually open. If you cannot open the Cohere card, status is `unknown` and the later card must say the base license was not re-read.
- Author: Arian Moridani, from the Hugging Face article byline. Teammates: none found in-repo. Email: unknown.
- Public URLs that exist today: article `https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a`, dataset `https://huggingface.co/datasets/ArianMoridani/LexsiHackathon` (empty), Space `https://huggingface.co/spaces/ArianMoridani/LexsiHackathon` (default template), GitHub `https://github.com/AriCommits/Pub_Hackathons` (stale `lexsi/` tree, no root README). Model collection: none found. Dataset collection: none found.
- Tracks to claim: T3 problem 1 (reliable tool calling) and T3 problem 6 (database / business questions on fixtures). Problem 2 only if you found an efficiency number; otherwise `unknown` and later writers omit the claim.

Add a short "Do not claim" list: distillation, safety-benchmark gains, multilingual coverage, a live web agent, CuratorKIT filtering, and GRPO as the thing that scored 200/200.

**Definition of done:**

- [ ] `FACTS.md` exists with the frontmatter and the table.
- [ ] Every numeric claim has a source path or is `unknown`.
- [ ] The sealed test is listed as withheld.
- [ ] No other file changed.
