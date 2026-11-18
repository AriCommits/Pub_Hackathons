# Sprint 2 — Article, cards, and the public data slice

**Plan:** `lexsi/docs/plan_15/OVERVIEW.md`
**Wave:** 2 of 4
**Can run in parallel with:** the other three agents in this file
**Must complete before:** Sprint 3 and Sprint 4

## Before this wave

Sprint 1 has committed `lexsi/submission/FACTS.md`. Read it before writing. If a fact is `unknown`, write "unknown" and the FACTS row id. Do not edit `FACTS.md`, `README.md`, `docs/SUBMISSION.md`, or `docs/plan_15/`. Do not create `submission/README.md`, `COVERAGE.md`, `FORM.md`, `hf_article.md`, or `submission/space/`.

Do not train. Do not pass `--final-eval`. Do not put test tasks, oracles, gold answers, or the fixture bank into `submission/data/`.

## Agents in this wave

### Agent A: D1 — Public data slice and datacard

**Complexity:** M
**Files:**
- `lexsi/scripts/curate_submission_dataset.py` (new)
- `lexsi/tests/test_submission_dataset.py` (new)
- `lexsi/submission/data/train_messages.jsonl` (new)
- `lexsi/submission/data/val_prompts.jsonl` (new)
- `lexsi/submission/data/manifest.json` (new)
- `lexsi/submission/DATACARD.md` (new)

**Depends on:** F1
**Blocks:** G1, C1

**Instructions:**

`docs/DATASET_V2.md` is the contract. `data/benchmarks/` and `data/processed/` are gitignored and may be missing. If a built `tooluse_v2` tree exists, read it. If it does not, run `python -m scripts.build_benchmark_v2 --output <a new directory under data/benchmarks> --seed 42` and do not overwrite an existing directory. That build stays gitignored.

The curator takes the build path and an output directory. It writes:

- `train_messages.jsonl` from the public training messages only. Drop task ids other than a train marker, group ids, and any object named like an oracle, gold call, or answer.
- `val_prompts.jsonl` with the validation user prompt and a task-type tag. No gold, no oracle, no test, no stress.
- `manifest.json` with seed, split sizes from the build manifest, counts of the two files, sha256 of each file, fixture id `tooluse_v2.fixtures.1`, and `withheld: ["test", "stress", "oracles", "fixtures", "traces"]`.

`tests/test_submission_dataset.py` builds a few fake rows, including one test row and one row with an oracle, runs the curator, and asserts those rows are absent and that an oracle key cannot appear in any output line.

`DATACARD.md` frontmatter:

```yaml
---
label: datacard
track: T3
audience: public-dataset
status: draft
---
```

Sections: summary, source (synthetic English templates in this repo, not a scraped corpus and not a teacher distillation), languages (English), splits, leak check, what was removed from the public slice and why, license (the name FACTS settled, or "unsettled — see FACTS" if unknown), and how to regenerate. State that stress is a held-out template family.

Run `python -m pytest tests/test_submission_dataset.py` from `lexsi/`.

**Definition of done:**

- [ ] The three data files exist and the test passes.
- [ ] A search of `submission/data` finds no `"oracle"`, no test-split record, and no gold answer field.
- [ ] The datacard label is `datacard`.

### Agent B: W1 — Judge write-up

**Complexity:** L
**Files:**
- `lexsi/submission/ARTICLE.md` (new)

**Depends on:** F1
**Blocks:** P1, H1, S2, G1, C1

**Instructions:**

Write the hackathon write-up in the order below. Use FACTS for every number. The abstract is under 500 words; put the word count in a line directly under it.

Frontmatter `label: judge-writeup`, `track: T3`, `audience: judges`, `status: draft`.

Required order:

1. Title and one-line claim. Verified tool use on Tiny Aya by LoRA SFT. The prototype router and GRPO are reported and are not the headline.
2. Tracks and problem statements. Bullet `T3` problem 1 and problem 6. Add problem 2 only when FACTS status for that efficiency number is `measured`.
3. Abstract.
4. Headline before → after. If the before number is `unknown`, say that in the sentence. Do not invent one from `improvement` unless FACTS marked the reconstruction `derived` and you show the arithmetic.
5. Base model plus method in one line, including the revision and "bf16 LoRA, not 4-bit".
6. "For judges: one minute" box: claim, metric, result, what failed, where the code and data are.
7. Problem statement and approach. Five local tools: `calculator`, `search_docs`, `query_database`, `lookup_record`, `finish`. One JSON call per turn. Verified by running the tools.
8. Why this problem. A small model that has to query a private table and may not invent the cell.
9. Kits. One subsection each. AgentTune: GRPO / rollout path, version from FACTS, and the statement that the selected policy is SFT. AuditKIT: recorded-eval path and whether the headline number came from it. CuratorKIT, AlignTune, SafeTune: not used. CircuitKIT: not used, or the FACTS sentence if a sweep exists.
10. How the repos connect. Prose plus a short mermaid flowchart: templates → splits → SFT e2 → verifier. GRPO is a side branch that did not replace e2.
11. Models and licenses. Base is gated. Submitted weights are the e2 adapter. No teacher. No closed-model baseline unless FACTS has one.
12. Datasets. Point at `DATACARD.md` for the table. English. Leak check. No CuratorKIT filtering. No attack set, no over-refusal set. Those are other tracks; one sentence is enough.
13. Metrics and why. Verified task success, higher is better. Add tool validity, unnecessary calls, and latency only with FACTS numbers, each with a direction.
14. Method. SFT subsection from `sft_v2.yaml` via FACTS (rank 16, alpha 32, dropout 0.05, lr `2e-4`). GRPO subsection labeled "not the selected policy" with the −0.1 cause and the post-fix tie.
15. Results. A before/after table using FACTS rows only. A "what didn't work" subsection for GRPO and, if FACTS has it, for the prototype router versus plain LoRA.
16. Limitations. English only, synthetic fixtures, sealed test not reported here, gated base, GRPO groups tied, hours and cost unknown if FACTS says so.
17. With more credits. What you would measure next (reward diversity on GRPO, a real efficiency cut, a public ungated base if one exists). Expected direction in words, no fake scores.
18. Artifacts. Existing URLs from FACTS, and `TODO` for the model collection, dataset collection, and filled dataset.
19. Documentation pointer to `REPRODUCE.md`.
20. Declaration. Work is this hackathon project. Do not claim a pre-hackathon paper.
21. Author Arian Moridani. Contributions: the project. Teammates: none in-repo. LLM-usage: this article was drafted by an agent from the repo notes and FACTS sheet.

Skip optional sections that FACTS cannot fill, and add one sentence under Limitations listing what you skipped (confidence intervals, cost per 1k, human judge agreement).

**Definition of done:**

- [ ] Every required heading above is present.
- [ ] Abstract word count is printed and is ≤ 500.
- [ ] No number appears that FACTS marks `unknown`.
- [ ] GRPO is not described as the source of the validation score.

### Agent C: M1 — Model card

**Complexity:** S
**Files:**
- `lexsi/submission/MODEL_CARD.md` (new)

**Depends on:** F1
**Blocks:** S2, G1, C1

**Instructions:**

Card for adapter e2 only. Frontmatter `label: model-card`, `audience: public-model`.

Include: developed by Arian Moridani, base `CohereLabs/tiny-aya-global` at the FACTS revision, PEFT LoRA, the FACTS hyperparameters, sha256, language (English tool-calling data; base model is multilingual and this adapter was not trained or scored as a multilingual system), license line taken from FACTS or "base license not re-read" if unknown, direct use (local verified tool calls on the fixture tools), out of scope (live web, security or red-team claims, business decisions without the verifier), training data pointer to `DATACARD.md`, evaluation (the FACTS val headline and the GRPO non-result), and hardware (RTX 5060 Ti, hours unknown if FACTS says so).

Do not edit files under `results/`.

**Definition of done:**

- [ ] The card names e2, the revision, and the sha256.
- [ ] It does not claim the empty PEFT stub's "More Information Needed" placeholders.
- [ ] It does not claim a multilingual fine-tune.

### Agent D: R1 — Reproduction

**Complexity:** M
**Files:**
- `lexsi/submission/REPRODUCE.md` (new)

**Depends on:** F1
**Blocks:** G1, C1

**Instructions:**

Frontmatter `label: reproduction`, `audience: operators`.

Write:

- What each top folder does: `actproto/`, `scripts/`, `tests/`, `docs/`, `submission/`, `results/` (adapters that are tracked versus gitignored run logs).
- Python `>=3.10`, `pip install -e ".[dev]"`, and `pip install -e ".[train]"` for GPU.
- `huggingface-cli login` because the base is gated. No token in the file.
- Dataset: `python -m scripts.build_benchmark_v2 --output <new-dir> --seed 42`, then the curator command you can see in D1's script name `scripts/curate_submission_dataset.py`. If you cannot read that script because the other agent is still writing it, document the module path and the two output filenames from this brief.
- SFT: `python scripts/train_sft_v2.py` pointed at the FACTS config, output not overwriting `results/adapters/b1`. The frozen artifact is `results/sft_v2/e2/checkpoint-150`.
- Eval: `python scripts/run_experiment_v2.py` on `--split val` only. Say the test split is sealed and is not a public reproduce step.
- GRPO: separate CUDA env, `scripts/train_grpo.py`, see `docs/RL_GPU_ENV.md`. State that a successful smoke is not a better policy than e2.
- Hardware: 16 GB card. The 8 GB card is the wrong device.
- Tests: `python -m pytest` from `lexsi/`.

Use repo-relative paths only. No `C:\Users\...`.

**Definition of done:**

- [ ] A reader can find the dataset, train, and val-eval commands.
- [ ] The sealed test is explicitly excluded.
- [ ] No machine-local user path.
