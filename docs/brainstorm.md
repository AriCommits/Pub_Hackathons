# Skill-delta brainstorm

Hackathon: [Lexsi AI Safety and Alignment Hackathon](https://hackathon.lexsi.ai/tracks)

Window: build Nov 1–14, 2026. Submit by end of day Nov 14. Demo Day teams announced Nov 16. Demo Day Nov 22. Registration closes Nov 5.

Track decision: **Track 3, Agentic Fine-tuning.** Libraries: [AgentTune](https://github.com/Lexsi-Labs/AgentTune) and [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT). Track 3 is exempt from the “three repos” preference. These two are the pipeline.

Models, one of: Tiny Aya Base, Tiny Aya Global, Tiny Aya Fire, Tiny Aya Water, Tiny Aya Earth, Aya Expanse 8B, Aya Expanse 32B. Start on Tiny Aya Global. It is gated: accept the license and `hf auth login` before training. Tiny Aya has no tool-calling chat template. AgentTune already falls back to putting schemas in the system prompt and parsing the action format. Use that path.

Required artifacts: a real pipeline, more than one before-and-after metric, a public repo or demo, a Hugging Face article, and models, adapters, logs, and datasets hosted on Hugging Face.

## The claim

A forward pass already writes a rank-1 change into the weights. That change is the query. Each tool is a stored skill delta, a small LoRA. The next tool is whichever stored delta points the same way as the update the model just wrote. The prompt contains no tool list and no tool descriptions.

After the tool result is appended, the update is read again. The arrow moves. The new nearest neighbor is the next skill. There is no planner. The residual update points at what the model still cannot do.

## Where this came from

First sketch: give an agent a small extension LoRA per tool, and treat the bank like a vector database. Use the weights themselves for cosine similarity, so the agent can find the next adapter from what its weights are for. The hope was that this would be model-agnostic.

The practical cut of that idea is a tool router: one LoRA per tool, pick it with a description embedding or a hidden-state key, ablate a weight fingerprint, ship the winner. That is buildable and it is boring. It is a dispatcher with a retrieval step in front. The weights are the skill, and a separate index does the choosing.

The interesting object is the lookup key. It has to be a change in the weights, produced by the model, compared to other changes in the weights.

## Mechanism

A linear attention layer implements one gradient step on an implicit loss. The update is rank-1: an outer product of the query direction and the context direction.

- von Oswald et al., 2023. *Transformers learn in-context by gradient descent.* ICML.
- The explicit rank-1 formula for any context: *Learning without training: The implicit dynamics of in-context learning.* arXiv:2507.16003.

At request time, do not train. Hook one layer, read the update that forward pass wrote, and cosine it against the same kind of update stored for each tool LoRA.

1. Freeze Tiny Aya. Run it on the user message. Read the rank-1 update at one chosen layer.
2. Cosine that update against every tool delta.
3. Load the nearest LoRA. The model answers, or emits the tool call that LoRA was trained to emit.
4. Append the tool result. Read the update again. The nearest neighbor should change.

Raw parameter cosine is the wrong geometry. Text-to-LoRA (Charakorn et al., ICML 2025, arXiv:2506.06105) measured this: adapters that behave alike are often far apart as matrices, while similarity of the task description tracks performance. Flattening `A` and `B` and taking cosine clusters noise.

Compare one of these instead:

- The induced rank-1 update at the hooked layer.
- The change each LoRA makes to hidden states on a fixed set of probe prompts (a functional fingerprint).

Same cosine. Different object. The fingerprint is the one to try if the raw induced updates are noisy on a pretrained Tiny Aya. The in-context-learning-as-gradient-descent equivalence is approximate outside linear attention, so the fingerprint is the fallback that still lives in behavior space.

Related work to cite and to separate from:

- Task arithmetic (Ilharco et al.): cosine between task vectors predicts whether merging helps or interferes. That is a merge decision, not a selector.
- LoRAuter (arXiv:2601.21795): routes with cosine on task embeddings from a few validation examples, then mixes adapters in output space. Routing on adapter characteristics lost to routing on task representations.
- Task-aware LoRA composition (arXiv:2602.21222): a vector database of training-example embeddings, merge the adapters of the neighbors.
- Text-to-LoRA: a hypernetwork emits a LoRA from a description. Useful as the negative result on raw weight space, and as the thing this project does not do. Generating a new adapter from text is a different project.

## What “model agnostic” can mean

A LoRA is tied to one architecture and one base checkpoint. A Tiny Aya adapter will not load onto Llama, Qwen, or a different Aya size. This contest also requires a Cohere Labs dense model, so a cross-family claim is out of scope.

The portable object is the probe-set fingerprint, not the LoRA file. Stretch experiment, only after the arrow plot works on one model: measure fingerprints on Tiny Aya Global, run the same probes on another Tiny Aya, and see whether the fingerprint still picks the right skill. The LoRA file stays on the model it was trained on. The index is what moves.

The smaller claim, and the one the demo should lead with: the base stays frozen, the prompt has no tool menu, and a new skill appears by dropping a delta into the library. No gradient step on a router.

## Demo

One plot. Project the stored deltas to two dimensions. Draw the request as an arrow. It lands on a tool. Append the observation. Draw the arrow again. It has moved, and the transcript matches the swing. A two-hop task is enough: lookup, then arithmetic, or search, then format.

The metric that matches the plot is tool-selection accuracy with an empty tool prompt. A schema-in-the-prompt baseline has nothing to dispatch on. This method still has the library.

Other metrics, so the submission has a before-and-after table:

- Right tool chosen.
- Arguments valid and correct.
- Task success.
- Steps per task.
- The swing: after step 1, the nearest delta is the correct second tool.

Before: base Tiny Aya, every tool written into the prompt. After: frozen base, empty tool prompt, routed deltas. A short GRPO run is optional and only after supervised LoRAs already separate. AgentTune’s distillation or SFT path is the run that has to finish.

Optional later figure, for the safety judges: estimate a direction from harmful versus harmless prompts, and refuse to load any tool delta that sits inside a cone around it. Same plot. Do this after the swing works. SafeTune and CircuitKIT stay out of the critical path.

## Day-2 kill test

Train four to six skills that are obviously different. Plot their deltas in the geometry you will actually cosine.

If they overlap, the library has no addresses. Stop. Do not train a classifier to hide that.

If they separate, spend the rest of the window on routing, the two-hop swing, AuditKIT numbers, the Hugging Face article, and the adapters.

Skills should have checkable results: arithmetic, a small SQLite lookup, unit conversion, a date or calendar calculation, a strict JSON schema, a retrieval hop. Six is enough. Hold one out until the library is frozen, then drop it in and show the arrow can land on it with no router update.

For a two-tool request, cosine the two chosen deltas before merging. Merge in output space when they agree. Run them as two steps when they conflict. That uses weight-space cosine for the decision task arithmetic actually supports.

## Track 2, if this dies on day 2

Switch only if the deltas do not separate and the goal becomes learning safety. Do one correction project.

- Inference-time refusal steering, or a domain fine-tune that keeps its refusals. SafeTune plus AuditKIT. Metrics: attack success, over-refusal on harmless prompts, task score.
- Skip CircuitKIT on a first safety project. Locating “where safety lives” often ends in a null result.

A red-teaming dataset (CuratorKIT plus AuditKIT) can ship, and it teaches less about how safety is repaired.

## Open questions

- Which single layer is the hook? Pick it by which layer’s induced updates separate the day-2 skills, not by a theory preference.
- Is the stored key the induced update on the tool’s training prompts, or the functional fingerprint on a shared probe set?
- Does the LoRA need to emit a tool call, or is the skill the behavior itself (arithmetic, formatting, SQL text) with no external harness? External tools are easier to score. Pure behavior skills make the “no tool menu” claim cleaner. The hackathon problem statements lean toward real tool calls. A split is reasonable: the delta selects and shapes the call, AgentTune’s harness executes it.
- How many probe prompts does a stable fingerprint need? Start tiny. Grow only if neighbors flip across seeds.
- Does the arrow swing on real two-hop items, or only when the second hop is obvious from the observation string?
- Tiny Aya may be too weak to emit a parseable call even with a LoRA. Aya Expanse 8B is the step up. 32B is the wrong size for the GPU credits and Colab.

## Non-goals

- A general agent platform.
- A trained router network that hides a failed geometry.
- Cosine between a text embedding and a flattened LoRA.
- Cross-architecture LoRA transfer.
- Circuit discovery.
- A multi-day GRPO run that never emits a parseable tool call.

## Rough sequence

1. License, login, one forward pass on Tiny Aya Global, one layer hook, dump an induced update. Confirm the hook is cheap.
2. Four to six skill LoRAs via AgentTune SFT or distillation. Plot the deltas. Kill or continue.
3. Empty-prompt selection on a held-out set. Record the baseline with the tool menu filled in.
4. Two-hop set. Plot the swing. Score whether the second neighbor is right.
5. Hold out one skill. Insert its delta. No router training.
6. AuditKIT on the trajectories. Hugging Face article, adapters, logs, dataset.
7. Only then: second Tiny Aya fingerprint transfer, or the refusal cone.
