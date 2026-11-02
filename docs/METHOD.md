# Method

## Hypothesis

Tiny Aya Global hidden states immediately before a tool call contain a linearly readable signal for *which* tool should run. Mean L2-normalized prototypes plus cosine similarity can route tool choice well enough to improve **verified** multi-step task success, or cut wasteful calls, relative to ordinary LoRA — without harming argument correctness or blowing the latency budget.

That is a hypothesis. B0–B2 exist to falsify it.

## Activation position

Default capture is the residual stream at layer `2 * n_layers // 3` (`activation.layer: auto`) at the last prompt token (`activation.token: prompt_last`), **before** tool-name tokens are sampled. Optional CircuitKIT / layer sweep writes `actproto/configs/layer.yaml` and must not rewrite `model.yaml`.

## Baselines

| ID | System |
| --- | --- |
| B0 | Base Tiny Aya, no adapter, no router |
| B1 | Ordinary LoRA SFT on gold tool-call trajectories |
| B2 | B1 + prototype router with val-chosen abstention |
| Probe | Multinomial logistic regression on the same frozen vectors |

A prototype-only story is weak unless the probe numbers sit next to it. If the probe reads the tool and prototypes do not help the agent, say so.

## Metrics that cannot be gamed by claiming success

AuditKIT-style `FinalStateAssertion` / `AnswerAssertion` on deterministic tool state. Fabricated assistant text does not pass. Also report tool-choice accuracy, argument validity vs correctness, step count, repeats, unnecessary calls, abstention coverage, and latency.

## Negative results

If prototypes classify tools but B2 does not beat B1 on verified success (or only wins by breaking arguments), that is a valid submission. Do not add steering, alignment loss, or GRPO to disguise it.

## Non-goals

Cross-model Procrustes/CCA, live web APIs, prompt-injection defense claims, and treating a detached cosine as a training loss.
