---
label: coverage
track: T3
audience: internal
status: draft
---

# Write-up coverage

Marks follow the hackathon list: `*` required, `^` optional, `"` track-conditional, `+` used when the author opted into hiring review.

## Open gaps

- Hugging Face model collection does not exist. `FORM.md` says TODO.
- Hugging Face dataset collection does not exist. `FORM.md` says TODO.
- The Hub dataset repo is still an empty card. The local slice is `data/`.
- The Hub Space is still the default template. The local app is `space/`.
- `AriCommits/Pub_Hackathons` has not been updated from this packet.
- The validation before-score is unknown. The scored JSON `results/sft_v2/e2-scored` is absent.
- Registered email is blank on purpose.

## Rows

| item | mark | track | status | where |
| --- | --- | --- | --- | --- |
| Title and one-line claim | * | all | filled | `ARTICLE.md` title and first paragraph |
| Tracks and problem statements | * | all | filled | `ARTICLE.md` Tracks. Problems 1 and 6. Problem 2 omitted. |
| Content warning | ^ | all | filled | `ARTICLE.md` Content warning. None. |
| Abstract under 500 words | * | all | filled | `ARTICLE.md` Abstract. 272 words. |
| Headline before to after | * | all | filled | `ARTICLE.md` Headline. Before is unknown on purpose. |
| Base model and method in one line | * | all | filled | `ARTICLE.md` Base model and method |
| For judges: one minute | ^ | all | filled | `ARTICLE.md` For judges |
| Problem statement and approach | * | all | filled | `ARTICLE.md` Problem statement and approach |
| Why this problem | * + | all | filled | `ARTICLE.md` Why this problem |
| Research questions | ^ | all | filled | `ARTICLE.md` Research questions |
| Pre-registered hypotheses | ^ | all | omitted-with-reason | Same section. None were registered. |
| Threat model | ^ + | T2a T3 | filled | `ARTICLE.md` Threat model |
| Related work | ^ | all | filled | `ARTICLE.md` Related work |
| Hackathon repos used | * | all | filled | `ARTICLE.md` Hackathon repos |
| CuratorKIT | " | when used | filled | Named as not used |
| AlignTune | " | when used | filled | Named as not used |
| AgentTune | " | T3 | filled | AgentTune subsection |
| SafeTune | " | when used | filled | Named as not used |
| CircuitKIT | " | when used | filled | Named as not used for a reported number |
| AuditKIT | " | T3 | filled | AuditKIT subsection |
| How the repos connect | * | all | filled | `ARTICLE.md` How the repos connect |
| Models and licenses | * | all | filled | `ARTICLE.md` Models and licenses |
| Base Aya model | * + | all | filled | Same section, plus `MODEL_CARD.md` |
| Teacher model | " | T1 | not-this-track | No teacher. Track 3. |
| Target model attacked | " | T2a | not-this-track | |
| Closed-model baseline | ^ | all | omitted-with-reason | None in the notes |
| Datasets, benchmarks, licenses | * | all | filled | `ARTICLE.md` and `DATACARD.md` |
| Sources of data | * + | all | filled | `DATACARD.md` Source |
| Languages covered | " + | all | filled | English only |
| Train and test splits, leak check | * + | all | filled | `DATACARD.md` Splits and leak check |
| Safety benchmarks | " + | T2a T2b | not-this-track | |
| General capability benchmarks | " + | T1 T3 | omitted-with-reason | No general-capability number in the checkout |
| CuratorKIT filtering | " + | T1 T2 | filled | Not used. Stated in the article and the card. |
| Attack prompt set | " + | T2a | not-this-track | |
| Over-refusal test set | " + | T2b | not-this-track | |
| Tool and task environments | " + | T3 | filled | `DATACARD.md` and the article's tool list |
| Metrics, and why | * | all | filled | `ARTICLE.md` Metrics |
| List of metrics | * + | all | filled | Verified task success, higher is better |
| Metrics that should not get worse | " ^ | all | omitted-with-reason | No paired safety or general score |
| Checking the automatic judge | ^ | all | omitted-with-reason | The checker is the tool result, not an LLM judge |
| Attack success rate | " | T2a | not-this-track | |
| Agentic trajectory metrics | " | T3 | filled | Verified success is reported. F1, extra calls, and latency are named as absent. |
| Method | * | all | filled | `ARTICLE.md` Method |
| Training setup and hyperparameters | * + | all | filled | Supervised LoRA and GRPO subsections |
| Seeds | ^ | all | filled | Seed 42 |
| Ablations | ^ | all | omitted-with-reason | `e1` and `e3` are omitted in the manifest, with no table |
| Results | * | all | filled | `ARTICLE.md` Results |
| Same model before vs after | * | all | filled | Results table. Before cell is unknown. |
| Safety and general side by side | " ^ | other tracks | not-this-track | |
| Confidence intervals | ^ | all | omitted-with-reason | Limitations |
| Findings as named subsections | ^ | all | filled | What didn't work |
| What didn't work | ^ | all | filled | What didn't work |
| Sample inputs and outputs | ^ | all | filled | Sample turn. Schema illustration. |
| Quantized or deployed results | " | T2b | not-this-track | |
| Cost and ROI | ^ | all | omitted-with-reason | Hours and dollars unknown |
| Compute used | + | all | filled | Hardware named. Hours unknown, and the article says so. |
| Cost per 1k replies | + | all | omitted-with-reason | Not measured |
| Limitations | * | all | filled | `ARTICLE.md` Limitations |
| Known weaknesses | * + | all | filled | Same section |
| Dual-use and ethics | " | T2a | not-this-track | |
| Responsible disclosure | " | T2a | not-this-track | |
| With more credits | * | all | filled | `ARTICLE.md` With more credits |
| Planned approach | * + | all | filled | Same section |
| Expected before and after | * + | all | filled | Direction in words, no forecasted score |
| Artifacts | * | all | gap | Local files exist. Hub collection URLs are TODO. See Open gaps. |
| HF datasets collection | * | all | gap | TODO in `FORM.md` |
| HF models collection | " | T1 T2b T3 | gap | TODO in `FORM.md` |
| GitHub repo | * | all | filled | Code is in this repo. The public mirror sync is a runbook step, not done. |
| Demo or Space | ^ | all | filled | `space/` locally. Hub page not updated. |
| Video or slides | ^ | all | omitted-with-reason | None |
| Documentation and reproduction | * | all | filled | `REPRODUCE.md` |
| Environment requirements | ^ | all | filled | `REPRODUCE.md` Environment |
| Repo layout and run instructions | * | all | filled | `REPRODUCE.md` |
| Declaration of outside work | * | all | filled | `ARTICLE.md` Declaration |
| Discussion | ^ | all | filled | What we learned |
| Implications | ^ | all | filled | What we learned |
| Future work | ^ | all | filled | With more credits |
| Teammates tagged | * | all | filled | None in the repo |
| Author contributions | + | all | filled | Author contributions |
| Acknowledgements | ^ | all | filled | Acknowledgements. None beyond the model and the kits. |
| References | ^ | all | filled | References |
| LLM usage statement | ^ | all | filled | LLM usage statement |
| Changelog | ^ | all | omitted-with-reason | First packet |
| Appendix | ^ | all | omitted-with-reason | Not needed |
| Prompts and judge rubric | ^ | all | omitted-with-reason | The checker is exact tool output, not a rubric |
| Extra tables | ^ | all | omitted-with-reason | The results table is in the body |
| Security considerations | " | T2a | not-this-track | |
