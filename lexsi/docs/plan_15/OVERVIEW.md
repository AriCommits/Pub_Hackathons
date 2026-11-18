# Plan 15: Lexsi submission write-ups and public artifacts

## Summary

The science for ActProto-LoRA already lives in `lexsi/`. The submission surfaces do not. The Hugging Face article is a short post that still says the adapter, dataset, and Space links will be added later. The dataset repo `ArianMoridani/LexsiHackathon` is an empty card (about 2.56 kB, license stub only). The Space of the same name is the default Gradio template. `https://github.com/AriCommits/Pub_Hackathons` has a `lexsi/` tree and no root README, and its `lexsi/README.md` is the older handoff that still treats `docs/EXPERIMENT_STATUS.md` and historical GRPO configs as live. There is no Hugging Face model collection and no dataset collection. Local `docs/SUBMISSION.md` is a checklist, not the judge write-up, and `lexsi/submission/` does not exist.

This plan builds that packet in `lexsi/submission/`, labels every document, curates a public dataset slice under `lexsi/submission/data/`, and then points the GitHub README, the Hugging Face article, the dataset card, a results Space, and the two required collections at the same facts. It does not retrain, rescore the sealed test, or claim a GRPO improvement.

## Goals

- One judge write-up, `lexsi/submission/ARTICLE.md`, that follows the hackathon section list for Track 3 and is honest about what was measured.
- A paste-ready form packet: abstract under 500 words, track and problem-statement bullets, and the links the portal asks for.
- A public data slice plus a dataset card. Train messages and validation prompts may ship. Sealed-test tasks, oracles, and gold answers may not.
- A model card for the frozen SFT e2 adapter (`results/sft_v2/e2/checkpoint-150`, sha256 `6125195bc713f8c4d9e89070233b65667a8068f8fc684a1e14a2522a26b8d670`).
- Reproduction notes a stranger can follow from `Pub_Hackathons` without the private Windows paths in `docs/EXPERIMENT_STATUS.md`.
- `lexsi/README.md` and `docs/SUBMISSION.md` point at the packet and match `docs/STATUS.md`.
- A runbook that says exactly how to update the existing article, dataset, Space, and GitHub repo, and how to open public ungated collections. Uploads wait on a token and on the license stop-gate below.

## Non-Goals

- New training, a new GRPO study, or a validation rescore. Frozen SFT e2 stays the headline. GRPO stays a negative result plus a repaired trainer.
- Opening `tooluse_v2` test, passing `--final-eval`, or publishing test traces.
- Claiming CuratorKIT, AlignTune, or SafeTune. They were not used.
- Claiming Track 1, Track 2a, or Track 2b, or Track 3 problem 3 (agentic distillation). Problem 2 is included only if the facts sheet already has an efficiency number.
- Republishing `CohereLabs/tiny-aya-global` weights, or treating that gated repo as something this project can ungate.
- A live Space that loads Tiny Aya. The Space is a results browser.
- Editing plans 1–14, `grpo_run.py`, reward code, or the sealed benchmark generator.
- Copying `house-edge/`, `poker-bot/`, `Pyramid-Scheme/`, or `data-dashboard/` into the public hackathon repo.

## Background / Context

Official packet (https://hackathon.lexsi.ai/): registered email, abstract under 500 words, a public ungated model collection, a public dataset collection, a write-up (Hugging Face article preferred, GitHub README acceptable), and a GitHub repo whose README covers reproducibility. Track 3 is exempt from the two-repo minimum. The track kits named on the page are AgentTune and AuditKIT.

Track 3 problem statements this project can actually speak to:

1. Reliable tool calling. The verified result is supervised LoRA on gold trajectories (frozen e2). Reinforcement learning on tool rollouts was built and run; it is not the improvement claim.
6. Data and database agents. Tasks are synthetic business questions over a local fixture database, scored on the tool result, not on the assistant's prose.

Problem 2 (efficient trajectories) is optional here and only with a number already written down. Problems 3–5 and 7–9 are out of scope.

What the local tree already says, and what writers must not soften:

- Base model `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`, bf16, `load_in_4bit: false` (`actproto/configs/model.yaml`). The repo is gated. Access is the account that accepted the license, not a public download.
- Headline metric is verified task success. `docs/METHOD.md` and `docs/RESULTS_2026-11-14.md` say frozen SFT e2 is the primary claim. Plans 12 and 13 record macro `verified_success_v2` of 1.0 on 200/200 validation tasks in `results/sft_v2/e2-scored`. That directory is gitignored. If the scored JSON is not on disk, the facts sheet quotes the committed prose and says the raw file is absent. It does not invent a second measurement.
- `docs/SFT_BASELINE_MANIFEST.json` records `improvement: 0.7466666666666666` on the val split. Quote the field. Do not turn it into a before-score unless the scored JSON defines that subtraction.
- Dataset contract is `docs/DATASET_V2.md`: seed 42, 1,200 / 200 / 200 / 200 train, validation, new-entity test, stress test. Stress is a held-out template family on unseen entities, not a held-out human language. Prompts are English. Leak checks cover group, entity, exact and normalized prompts, and call fingerprints.
- `data/processed/` and `data/benchmarks/` are gitignored. The public GitHub tree has no dataset. Curate from a local build, or regenerate with `python -m scripts.build_benchmark_v2` into a new directory. Do not overwrite an existing benchmark directory.
- AgentTune 1.1.3 is the GRPO / rollout stack when the CUDA env is installed (`docs/RL_GPU_ENV.md`, `docs/TOOLKIT_COMPATIBILITY.md`). AuditKIT 1.1.3 commit `71316da345784ff220d5ed59337441a59ab31526` has a recorded-episode path. The write-up must say whether the 200/200 figure was produced by that runner or by the local verifier. A compatibility class is not "we used AuditKIT."
- GRPO history that must stay visible: `grpo_gpu_e2` and the pre-fix studies logged reward −0.1 because sampling ran under `model.train()` plus gradient checkpointing. After `_sample_peft_action` forces `eval()` for `generate`, `results/rl_v2/grpo_gpu_study_fix` finishes with reward 1.0 and `completion_reason=finish`, and groups stay tied because e2 already solves many train prompts (`docs/RESULTS_2026-11-14.md`). That is not a validation gain.
- Adapter cards under `results/adapters/*/README.md` and `results/sft_v2/e2/checkpoint-150/README.md` are the empty PEFT stub ("More Information Needed").
- The article at https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a (published 16 November 2026) describes the method and then stops at "links go here once they are published."
- Today on the machine clock is 18 November 2026. `docs/SUBMISSION.md` still says the deadline was 14 November 2026. Finish the packet anyway. Do not invent a deadline extension.

Label every file in `lexsi/submission/` with this frontmatter, filled in:

```yaml
---
label: judge-writeup | form | datacard | model-card | reproduction | hf-article | space | internal-facts | coverage | publish-runbook
track: T3
audience: judges | portal | public-dataset | public-model | operators | internal
status: draft
---
```

`internal-facts` and `publish-runbook` stay in git so the packet can be regenerated. They are not the document to paste into the portal.

## Features / Tasks

### F1: Lock the evidence sheet
**Files:** `lexsi/submission/FACTS.md` (new)
**Complexity:** M
**Depends on:** none

Read `docs/STATUS.md`, `docs/RESULTS_2026-11-14.md`, `docs/DATASET_V2.md`, `docs/METHOD.md`, `docs/SFT_BASELINE_MANIFEST.json`, `docs/TOOLKIT_COMPATIBILITY.md`, `docs/RL_BACKEND_COMPATIBILITY.md`, `actproto/configs/model.yaml`, `actproto/configs/sft_v2.yaml`, and `actproto/configs/trial.yaml`. If `results/sft_v2/e2-scored` exists, quote the metric names and the val macro from the file. If it does not, say so.

The sheet is a table of claim, source path, and value. Separate three columns: measured, derived, and not measured. Required rows: base model id and revision, dtype, quantization, adapter path and sha256, dataset sizes and seed, leak-check method, languages (English only), stress-axis meaning, val headline, GRPO negative results, which kits ran, which kits did not, GPU identity, whether hours and dollar cost were logged, license strings actually read from the Cohere and Lexsi cards, and the sole author name evidenced by the Hugging Face article byline. Unknown stays unknown.

Label: `internal-facts` / `internal`. No other task may invent a number that this sheet marks unknown.

### D1: Curate the public dataset and write the datacard
**Files:** `lexsi/scripts/curate_submission_dataset.py` (new), `lexsi/tests/test_submission_dataset.py` (new), `lexsi/submission/data/train_messages.jsonl` (new), `lexsi/submission/data/val_prompts.jsonl` (new), `lexsi/submission/data/manifest.json` (new), `lexsi/submission/DATACARD.md` (new)
**Complexity:** M
**Depends on:** F1

Build or locate `tooluse_v2` (see `docs/DATASET_V2.md`). The curator writes only public rows:

- `train_messages.jsonl`: the existing public training messages (prompts, assistant calls, observations). No task ids beyond a train marker, no group ids, no oracle objects.
- `val_prompts.jsonl`: validation user prompts and task-type tags. No gold calls, no answers, no oracle.
- `manifest.json`: counts, seed, split sizes, fixture environment id `tooluse_v2.fixtures.1`, sha256 of each output file, and an explicit `withheld` list (`test`, `stress`, oracles, fixtures, traces).

The test builds a tiny fixture, runs the curator, and fails if any output line contains an oracle key, a gold answer, a test split label, or a stress-test record. The datacard is the Hugging Face dataset README body: license, language, source (synthetic templates, not scraped and not distilled from a teacher), splits, leak check, what was withheld and why, and the label `datacard`.

Do not add `submission/data/` to an ignore rule. Do not copy `data/benchmarks/` into git.

### W1: Judge write-up
**Files:** `lexsi/submission/ARTICLE.md` (new)
**Complexity:** L
**Depends on:** F1

Write the hackathon layout in order. Required sections are filled. Optional sections are included when F1 has a value and omitted, with one clause saying why, when it does not. Track-conditional sections for T1, T2a, and T2b are one line each: not this track.

Cover, using F1's words:

- Title and one-line claim: verified tool use on Tiny Aya by LoRA SFT; the prototype router and GRPO are reported beside it and are not the headline.
- Tracks: T3 problem 1, T3 problem 6, and problem 2 only if F1 has an efficiency metric.
- Abstract under 500 words. Count the words and record the count under the abstract.
- Before → after number, or "not in the committed snapshot" if F1 says the before figure is unknown. Do not convert `improvement` into a baseline unless F1 did.
- Base model and method in one line.
- A "For judges: one minute" box.
- Problem, approach, and why a small local tool agent is the use case.
- Kits: AgentTune and AuditKIT with the features actually exercised; CuratorKIT, AlignTune, SafeTune, and CircuitKIT named as unused or reference-only. CircuitKIT is reference-only unless F1 finds a layer sweep that changed a reported number.
- How the pieces connect, in prose. A mermaid diagram is welcome.
- Models and licenses. The base is gated; the submitted artifact is the LoRA adapter. No teacher model.
- Data: synthetic English, split table, leak check, no CuratorKIT filtering.
- Metrics: verified task success (higher is better), plus tool-call validity, unnecessary calls, and latency where F1 has them. State the direction of each.
- Training setup from `sft_v2.yaml` (rank 16, alpha 32, dropout 0.05, lr `2e-4`, the targets in STATUS / plan 12). GRPO hyperparameters from `docs/RESULTS_2026-11-14.md`, in a separate subsection, labeled as not the selected policy.
- Results table: same model before vs after, using only F1 rows. A "what didn't work" subsection for the prototype-versus-LoRA outcome if F1 has it, and for GRPO ties and the −0.1 sampling bug.
- Limitations, what more compute would try, expected direction without fake numbers, artifacts (links left as the placeholders F1 recorded), reproduction pointer, declaration of work, author (Arian Moridani unless F1 found teammates), and an LLM-usage statement that this write-up was drafted with an agent from the repo's own notes.

Label: `judge-writeup` / `judges`.

### M1: Model card
**Files:** `lexsi/submission/MODEL_CARD.md` (new)
**Complexity:** S
**Depends on:** F1

Replace the empty PEFT stub with a card for adapter e2 only. Include base model, revision, LoRA config, license of the adapter (the base model's license, named from F1, not invented), language (English tool calls; the base model is multilingual and this adapter was not trained as a multilingual system), out-of-scope uses (live web, security claims, ungrounded business decisions), the val headline, the GRPO non-result, and the sha256. Point training data at `DATACARD.md`. Label: `model-card`.

Do not edit `results/**/README.md` in this task. The publish runbook copies this card later.

### R1: Reproduction notes
**Files:** `lexsi/submission/REPRODUCE.md` (new)
**Complexity:** M
**Depends on:** F1

Repo layout, Python 3.10+, `pip install -e ".[dev]"`, the gated-model login, and the commands that regenerate the dataset, train SFT, and score validation. Point GRPO at `scripts/train_grpo.py` and `docs/RL_GPU_ENV.md`, and say the CUDA env is a separate interpreter. State hardware: RTX 5060 Ti 16 GB for the model, RTX 4060 8 GB is too small. Use repo-relative paths. Do not copy `C:\Users\arian\...` into this file. Say the sealed test is not part of the public reproduce path. Label: `reproduction`.

### P1: Portal form packet
**Files:** `lexsi/submission/FORM.md` (new)
**Complexity:** S
**Depends on:** W1

One page the author can paste from. Fields: email (blank, with the instruction to use the registered address; do not guess), abstract copied from ARTICLE.md with the word count repeated, track bullets copied from the article, collection URLs as `TODO` until C1 fills them, article URL, dataset URL, Space URL, GitHub URL `https://github.com/AriCommits/Pub_Hackathons`. Fail the task if the abstract exceeds 500 words. Label: `form`.

### H1: Hugging Face article source
**Files:** `lexsi/submission/hf_article.md` (new)
**Complexity:** S
**Depends on:** W1

The body is ARTICLE.md plus a one-line note that this file is what gets pasted over https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a. Do not rewrite the science. Add the dataset, adapter, Space, and GitHub links as the same TODOs FORM.md uses, so C1 can fill both. Label: `hf-article`.

### S2: Results Space
**Files:** `lexsi/submission/space/README.md` (new), `lexsi/submission/space/app.py` (new), `lexsi/submission/space/requirements.txt` (new), `lexsi/submission/space/results.json` (new)
**Complexity:** M
**Depends on:** W1, M1

A Gradio app with no `torch`, no `@spaces.GPU`, and no token. `results.json` contains only numbers present in FACTS.md. The UI shows the one-line claim, the before/after line, the GRPO non-result, links, and a short sample of a tool-call turn taken from the public train format (a schema example, not a test item). `requirements.txt` is `gradio` only. The Space README has the Hugging Face YAML header (`title`, `emoji`, `colorFrom`, `colorTo`, `sdk: gradio`, `app_file: app.py`) and the label `space`.

### G1: Point the repo README at the packet
**Files:** `lexsi/README.md` (modify), `lexsi/docs/SUBMISSION.md` (modify)
**Complexity:** S
**Depends on:** F1, D1, W1, M1, R1

`lexsi/README.md` keeps setup and the frozen-e2 commands from the current file. Add a "Submission" section that links `submission/ARTICLE.md`, `submission/DATACARD.md`, `submission/MODEL_CARD.md`, and `submission/REPRODUCE.md`. Delete any instruction that presents `grpo_v2_gpu_tools.yaml` or `docs/EXPERIMENT_STATUS.md` as the live handoff. The current README already points at `docs/STATUS.md`; keep that.

`docs/SUBMISSION.md` becomes a short pointer: track, canonical adapter, links to the packet, and the sentence that GRPO is not the selected policy. Remove the 14 November deadline as if it were still an instruction. Keep the historical date in one clause.

Do not edit `submission/ARTICLE.md`.

### C1: Coverage matrix and publish runbook
**Files:** `lexsi/submission/README.md` (new), `lexsi/submission/COVERAGE.md` (new), `lexsi/submission/PUBLISH.md` (new)
**Complexity:** M
**Depends on:** D1, W1, M1, R1, P1, H1, S2, G1

`README.md` is the directory index with each file's label.

`COVERAGE.md` is a row per bullet in the hackathon write-up list. Columns: id, required or optional, track tag, status (`filled`, `omitted-with-reason`, `not-this-track`), and the file plus heading that satisfies it. Walk ARTICLE.md and mark a required bullet `filled` only when the heading exists and is not a placeholder. Track 3 bullets for attack ASR, over-refusal, and safety benchmarks are `not-this-track`.

`PUBLISH.md` is the operator runbook, label `publish-runbook`:

1. License stop-gate. Read the Tiny Aya license named in FACTS.md. Publish `adapter_config.json` and `adapter_model.safetensors` for e2. Do not upload base weights. Do not upload `tokenizer.json` or `tokenizer_config.json` unless that license explicitly allows redistribution of the tokenizer. The local git tree contains those tokenizer files; the public push must not assume they are fine.
2. Create a public ungated model repo (proposed id `ArianMoridani/actproto-lora-e2`) and upload the adapter plus `MODEL_CARD.md` as the repo README. Create a public Hugging Face model collection that contains that repo. If legacy `b1*` adapters are added, they get the same card treatment and the same tokenizer rule, and the article must still say e2 is the selected one.
3. Upload `submission/data/*` to `ArianMoridani/LexsiHackathon` with `DATACARD.md` as the dataset README. Set the dataset public and ungated. Replace the current `fair-noncommercial-research-license` stub only with the license FACTS.md settled. Create a public dataset collection that contains this dataset.
4. Paste `hf_article.md` over the existing blog post. Do not open a second article.
5. Push `submission/space/` to `ArianMoridani/LexsiHackathon` (Space). Do not leave the default `torch` sample in place.
6. Update `AriCommits/Pub_Hackathons`: add a root README that points at `lexsi/`, copy the current `lexsi/` tree including `submission/` and the publishable adapter weights, and exclude other hackathon projects, `.venv`, `data/benchmarks/`, sealed test files, tokens, and tokenizer files if the stop-gate failed.
7. Write the resulting URLs back into `FORM.md` and `hf_article.md` only after the uploads exist. Until then the TODOs stay.

This task writes the runbook. It does not perform the uploads. Uploads need the author's Hugging Face token and GitHub push rights to `AriCommits/Pub_Hackathons`, which this plan does not assume are present.

## New Dependencies

None. The Space lists `gradio` in its own `requirements.txt` and does not add it to `lexsi/pyproject.toml`. Dataset curation uses the stdlib plus the existing benchmark builder.

## File Change Summary

| Path | Action | Task |
| --- | --- | --- |
| `lexsi/submission/FACTS.md` | new | F1 |
| `lexsi/scripts/curate_submission_dataset.py` | new | D1 |
| `lexsi/tests/test_submission_dataset.py` | new | D1 |
| `lexsi/submission/data/train_messages.jsonl` | new | D1 |
| `lexsi/submission/data/val_prompts.jsonl` | new | D1 |
| `lexsi/submission/data/manifest.json` | new | D1 |
| `lexsi/submission/DATACARD.md` | new | D1 |
| `lexsi/submission/ARTICLE.md` | new | W1 |
| `lexsi/submission/MODEL_CARD.md` | new | M1 |
| `lexsi/submission/REPRODUCE.md` | new | R1 |
| `lexsi/submission/FORM.md` | new | P1 |
| `lexsi/submission/hf_article.md` | new | H1 |
| `lexsi/submission/space/README.md` | new | S2 |
| `lexsi/submission/space/app.py` | new | S2 |
| `lexsi/submission/space/requirements.txt` | new | S2 |
| `lexsi/submission/space/results.json` | new | S2 |
| `lexsi/README.md` | modify | G1 |
| `lexsi/docs/SUBMISSION.md` | modify | G1 |
| `lexsi/submission/README.md` | new | C1 |
| `lexsi/submission/COVERAGE.md` | new | C1 |
| `lexsi/submission/PUBLISH.md` | new | C1 |

No task edits `docs/plan_15/`.

## Open Questions

1. The Hugging Face token and push rights for `ArianMoridani` and `AriCommits/Pub_Hackathons` are not available in this plan. C1 writes the commands and stops before upload.
2. `CohereLabs/tiny-aya-global` is gated. The rules ask for public ungated repos in the collection. The adapter repo can be public. The base cannot be ungated from here. Confirm the adapter and tokenizer redistribution terms before step 1 of the runbook. Default is adapter weights only.
3. The empty dataset card says `fair-noncommercial-research-license`. Synthetic template data can carry a different license than the base model. F1 records both licenses; D1 does not pick a new one that contradicts the base-model card if the adapter or any model-generated trace is in the upload. Train messages include assistant tool calls produced by the generator, not by Tiny Aya, so they are template output. Say that in the card.
4. `results/sft_v2/e2-scored` is gitignored. If it is not on the machine that runs F1, the before/after table stays incomplete on purpose.
5. Registered email and any teammate list are not in the repo. FORM.md leaves them blank.
6. Whether the portal is still open after 14 November 2026 is unknown. The packet is still the thing to finish.
