# Sprint 4 — Coverage matrix and publish runbook

**Plan:** `lexsi/docs/plan_15/OVERVIEW.md`
**Wave:** 4 of 4
**Can run in parallel with:** nobody
**Must complete before:** a human upload. This wave does not upload.

## Before this wave

Sprints 1–3 are committed. The submission directory contains FACTS, ARTICLE, DATACARD, MODEL_CARD, REPRODUCE, FORM, hf_article, data, and space. `README.md` and `docs/SUBMISSION.md` already point at the packet. Do not edit those. Do not edit `docs/plan_15/`.

## Agents in this wave

### Agent A: C1 — Index, coverage, and the publish runbook

**Complexity:** M
**Files:**
- `lexsi/submission/README.md` (new)
- `lexsi/submission/COVERAGE.md` (new)
- `lexsi/submission/PUBLISH.md` (new)

**Depends on:** D1, W1, M1, R1, P1, H1, S2, G1
**Blocks:** nothing in this plan

**Instructions:**

`submission/README.md` frontmatter `label: coverage`, `audience: judges`. It is an index: each file, its label, and one sentence. Say that `FACTS.md` and `PUBLISH.md` are internal and that `ARTICLE.md` is the write-up to read.

`COVERAGE.md` frontmatter `label: coverage`, `audience: internal`. One row per bullet in the hackathon write-up list the user supplied (title, tracks, content warning, abstract, before/after, base model, judges box, problem and approach, why, research questions, kits, how repos connect, models and licenses, datasets, metrics, method, results, limitations, more credits, artifacts, documentation, declaration, teammates, and the optional subsections under them).

Columns: `item`, `mark` (`*`, `^`, `"`, `+` as in the guidelines), `track`, `status`, `where`.

`status` is:

- `filled` when ARTICLE.md has a heading and a real sentence
- `omitted-with-reason` when the article says why it is missing
- `not-this-track` for T1, T2a, T2b, attack ASR, over-refusal, and safety-benchmark requirements
- `gap` when a required `*` or `+` item for T3 is still a TODO

A required T3 item left as `gap` must be listed again at the top of the file under "Open gaps". Do not convert a gap into `filled`.

`PUBLISH.md` frontmatter `label: publish-runbook`, `audience: operators`. Steps, in order, for a human with tokens. This agent does not run them.

1. License stop-gate. Read the Tiny Aya license named in FACTS. Upload `adapter_config.json` and `adapter_model.safetensors` for `results/sft_v2/e2/checkpoint-150`. Do not upload base weights. Do not upload `tokenizer.json` or `tokenizer_config.json` unless that license says tokenizer redistribution is allowed. The local tree already contains tokenizer files; that is not permission.
2. Create public ungated model repo `ArianMoridani/actproto-lora-e2` (change the id only if it is taken) and use `MODEL_CARD.md` as its README. Create a public model collection containing that repo. Legacy `b1*` adapters are optional and must not be described as the selected model. Same tokenizer rule.
3. Upload `submission/data/*` to the existing dataset `ArianMoridani/LexsiHackathon`. Set `DATACARD.md` as its README. Public, ungated. Keep the license FACTS settled; do not leave a contradictory stub. Create a public dataset collection containing this dataset.
4. Paste `hf_article.md` over the existing blog. Do not open a second article.
5. Push `submission/space/` onto the existing Space `ArianMoridani/LexsiHackathon`, replacing the default Gradio sample.
6. Update `https://github.com/AriCommits/Pub_Hackathons`: root README pointing at `lexsi/`, current `lexsi/` including `submission/` and the allowed adapter weights. Exclude other projects in the monorepo, virtualenvs, `data/benchmarks/`, sealed test files, `.env`, and tokenizer files if the stop-gate failed.
7. After the URLs exist, replace the `TODO` collection lines in `FORM.md` and `hf_article.md`. Not before.

End the runbook with a checklist the author can tick: dataset viewer shows rows, model repo loads as a PEFT adapter, collection pages are public, article links resolve, Space is not the torch hello-world, GitHub root README exists, abstract still under 500 words.

**Definition of done:**

- [ ] The index lists every file currently in `submission/`.
- [ ] COVERAGE has a row for each guideline bullet and an Open gaps section.
- [ ] PUBLISH.md tells the operator not to upload tokenizer files unless the license allows it.
- [ ] No upload was performed and no secret was written.
