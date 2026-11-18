---
label: publish-runbook
track: T3
audience: operators
status: draft
---

# Publish runbook

These steps need a Hugging Face token for `ArianMoridani` and push rights on `AriCommits/Pub_Hackathons`. This file does not run them.

1. License stop-gate. The Tiny Aya card says CC-BY-NC plus Cohere Labs' Acceptable Use Policy, and the base repo is gated (`submission/FACTS.md`). Upload `adapter_config.json` and `adapter_model.safetensors` from `results/sft_v2/e2/checkpoint-150`. Do not upload base weights. Do not upload `tokenizer.json` or `tokenizer_config.json` unless you have confirmed the gate allows redistributing the tokenizer. The local tree already contains tokenizer files. That is not permission. The Git LFS pointer is not the weight file. Smudge the blob before upload.

2. Create a public ungated model repo. Proposed id: `ArianMoridani/actproto-lora-e2`. Change it only if that id is taken. Use `submission/MODEL_CARD.md` as the repo README. Create a public model collection that contains that repo. Legacy `b1*` adapters are optional, follow the same tokenizer rule, and must not be described as the selected model.

3. Upload `submission/data/*` to the existing dataset `ArianMoridani/LexsiHackathon`. Set `submission/DATACARD.md` as its README. Set the dataset public and ungated. Set the license to `cc-by-nc-4.0`. The current stub says `fair-noncommercial-research-license`. Replace it. Create a public dataset collection that contains this dataset.

4. Paste `submission/hf_article.md` over https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a. Do not open a second article.

5. Push `submission/space/` onto the existing Space `ArianMoridani/LexsiHackathon`, replacing the default Gradio sample. The app does not load Tiny Aya.

6. Update https://github.com/AriCommits/Pub_Hackathons. Add a root README that points at `lexsi/`. Copy the current `lexsi/` tree, including `submission/` and the allowed adapter weights. Leave out every other project in the monorepo, virtualenvs, `data/benchmarks/`, sealed test files, `.env`, and tokenizer files if the stop-gate failed.

7. After those URLs exist, replace the `TODO` collection lines in `submission/FORM.md` and `submission/hf_article.md`. Not before.

## Checklist

- [ ] Dataset viewer shows the train and val rows
- [ ] Model repo loads as a PEFT adapter on the pinned Tiny Aya revision
- [ ] Model collection page is public
- [ ] Dataset collection page is public
- [ ] Article links resolve and the old "links go here later" sentence is gone
- [ ] Space is not the torch hello-world sample
- [ ] GitHub root README exists and points at `lexsi/`
- [ ] Abstract is still under 500 words
- [ ] Tokenizer files were not uploaded unless the license check passed
