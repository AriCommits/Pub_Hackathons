# Sprint 3 — Form, article mirror, Space, README

**Plan:** `lexsi/docs/plan_15/OVERVIEW.md`
**Wave:** 3 of 4
**Can run in parallel with:** the other agents in this file
**Must complete before:** Sprint 4

## Before this wave

Sprint 2 is committed. These files exist: `submission/FACTS.md`, `submission/ARTICLE.md`, `submission/DATACARD.md`, `submission/MODEL_CARD.md`, `submission/REPRODUCE.md`, and `submission/data/`. Do not edit them. Do not edit `docs/plan_15/`. Do not upload anything.

## Agents in this wave

### Agent A: P1 — Portal form packet

**Complexity:** S
**Files:**
- `lexsi/submission/FORM.md` (new)

**Depends on:** W1
**Blocks:** C1

**Instructions:**

Frontmatter `label: form`, `audience: portal`. This is the page a person pastes into the hackathon form.

Fields, in order:

- Email: blank. One sentence: use the address you or a teammate registered with. Do not guess an address.
- Abstract: copy the abstract from `ARTICLE.md` unchanged. Repeat the word count. If it is over 500, stop and report that; do not silently trim.
- Tracks: the same bullets as the article.
- Hugging Face model collection: `TODO`.
- Hugging Face dataset collection: `TODO`.
- Article: `https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a`
- Dataset repo: `https://huggingface.co/datasets/ArianMoridani/LexsiHackathon` with the note that the upload is still empty until the runbook runs.
- Space: `https://huggingface.co/spaces/ArianMoridani/LexsiHackathon` with the note that it is still the default template until the runbook runs.
- GitHub: `https://github.com/AriCommits/Pub_Hackathons`
- Write-up in repo: `lexsi/submission/ARTICLE.md`

**Definition of done:**

- [ ] Abstract matches ARTICLE.md and is ≤ 500 words.
- [ ] Email is blank.
- [ ] Collection fields are `TODO`, not invented URLs.

### Agent B: H1 — Hugging Face article source

**Complexity:** S
**Files:**
- `lexsi/submission/hf_article.md` (new)

**Depends on:** W1
**Blocks:** C1

**Instructions:**

Frontmatter `label: hf-article`, `audience: judges`. First line after the frontmatter: this file is pasted over the existing article `https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a`. Do not create a second post.

Then the body of `ARTICLE.md` without its frontmatter. Where the article says artifacts are TODO, keep the TODO. Add the four live URLs from the form brief (article, empty dataset, template Space, GitHub) so a reader sees what exists today.

Do not change a number.

**Definition of done:**

- [ ] The body matches ARTICLE.md.
- [ ] The paste target is the existing blog URL.
- [ ] No second-article instruction.

### Agent C: S2 — Results Space

**Complexity:** M
**Files:**
- `lexsi/submission/space/README.md` (new)
- `lexsi/submission/space/app.py` (new)
- `lexsi/submission/space/requirements.txt` (new)
- `lexsi/submission/space/results.json` (new)

**Depends on:** W1, M1
**Blocks:** C1

**Instructions:**

`results.json` is a small object whose numbers are copied from FACTS.md. Omit keys FACTS marks unknown rather than filling them with zero.

`requirements.txt` contains `gradio` and nothing else.

`app.py` loads `results.json` and shows the one-line claim, the before/after line, the GRPO non-result, and the four URLs. Include one made-up-looking example only if it is clearly a schema illustration of a single `<tool_call>` for `query_database`, not a row from the test set. No `import torch`. No `@spaces.GPU`. No token read. No network call at import.

`README.md` starts with Hugging Face Space YAML:

```yaml
---
title: ActProto-LoRA results
emoji: "🔎"
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
label: space
track: T3
audience: public-model
status: draft
---
```

Short body: this Space does not load Tiny Aya. The base model is gated and the interesting artifact is the scored adapter.

**Definition of done:**

- [ ] `python -c "import ast,pathlib; ast.parse(pathlib.Path('submission/space/app.py').read_text())"` succeeds from `lexsi/`.
- [ ] `app.py` does not contain `torch` or `spaces.GPU`.
- [ ] `results.json` has no key whose value you invented.

### Agent D: G1 — README pointers

**Complexity:** S
**Files:**
- `lexsi/README.md` (modify)
- `lexsi/docs/SUBMISSION.md` (modify)

**Depends on:** F1, D1, W1, M1, R1
**Blocks:** C1

**Instructions:**

In `README.md`, add a Submission section linking:

- `submission/ARTICLE.md`
- `submission/DATACARD.md`
- `submission/MODEL_CARD.md`
- `submission/REPRODUCE.md`
- `docs/STATUS.md` (already the handoff; keep it)

Keep the install and frozen-e2 commands. Do not restore the old public README's live `grpo_v2_gpu_tools.yaml` instructions. GRPO stays the optional section that already points at `docs/RL_GPU_ENV.md`, with one added sentence: the selected policy is frozen SFT e2.

Replace `docs/SUBMISSION.md` with a short pointer. Keep one historical clause that an earlier checklist named 14 November 2026. State the track (T3), the canonical adapter path and sha256 from FACTS, that GRPO is not the selected policy, and the links to the four submission docs. Delete instructions that read like the deadline is still a task to race.

Do not edit `submission/ARTICLE.md`.

**Definition of done:**

- [ ] README links the four submission docs.
- [ ] SUBMISSION.md is a pointer, not a second write-up.
- [ ] No Windows user path added.
