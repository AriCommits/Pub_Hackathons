---
label: form
track: T3
audience: portal
status: draft
---

# Portal form

Paste these fields into the hackathon form.

## Email

(blank)

Use the address you or a teammate registered with. Do not guess an address.

## Abstract

ActProto-LoRA fine-tunes Cohere Labs Tiny Aya Global so a small local agent can answer business questions by calling tools, and so a task scores as correct only when those tools actually run. The selected model is a LoRA adapter, id e2, checkpoint-150, trained on synthetic English trajectories. Five deterministic tools cover the work: record lookup, a sales-table query, document search, a calculator, and finish. The harness checks the call against the tool schema and checks the final value. An assistant message that only says it is done does not pass.

Two other methods sit beside that adapter and do not replace it. A prototype router reads the residual stream just before the tool name is chosen and abstains unless a validation threshold is met. GRPO on in-process tool rollouts was built and run. Early runs scored every step at reward −0.1 because sampling happened while the model was in train mode. After generation was switched to eval mode, a short study finished episodes at reward 1.0. The groups were often tied, because the supervised adapter already solved many of the training prompts, so the policy did not move. That study is not a validation win.

Project notes record validation macro verified task success of 1.0 on all 200 validation tasks for the frozen adapter. The scored JSON is not in this checkout, and a stored improvement field of about 0.75 is not turned into a before-score here. The public data release is the training messages and the validation prompts. The sealed test, the stress split, oracles, fixtures, and traces stay private. The base weights are gated. The submitted artifact is the adapter.

Word count: 272

## Tracks

- Track 3, problem 1. Reliable tool calling. The verified policy is supervised LoRA on gold tool trajectories. Reinforcement learning on real tool rollouts was implemented and run, and it is not the improvement claim.
- Track 3, problem 6. Data and database agents. The questions are about a local fixture table, a record store, and a document catalog. The reward is the checked tool result.

Track 3 problem 2 (shorter trajectories at the same success rate) is not claimed. This checkout has no efficiency number for it.

## Links

- Hugging Face model collection: TODO
- Hugging Face dataset collection: TODO
- Article: https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a
- Dataset repo: https://huggingface.co/datasets/ArianMoridani/LexsiHackathon (upload still empty until the runbook runs)
- Space: https://huggingface.co/spaces/ArianMoridani/LexsiHackathon (still the default template until the runbook runs)
- GitHub: https://github.com/AriCommits/Pub_Hackathons
- Write-up in repo: `lexsi/submission/ARTICLE.md`
