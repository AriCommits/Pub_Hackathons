# Sprint 2 — Gated GPU run

**Plan:** docs/plan_1/OVERVIEW.md
**Wave:** 2 of 2
**Can run in parallel with:** none — serial
**Must complete before:** nothing

This wave is one agent and one GPU. It starts only after Sprint 1 has landed and after any live `scripts/run_trial.py` has exited on its own. Do not kill that process. Do not start this wave because the code is ready if `nvidia-smi` shows the 5060 Ti still running B2.

Working directory: `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter: `.venv\Scripts\python`. Device: `ACTPROTO_CUDA_DEVICES=1` (RTX 5060 Ti). One replica. The RTX 4060 is not a target.

---

## Agents in This Wave

### Agent A: G1 — Train the masked adapter and score it

**Complexity:** M
**Estimated time:** 4 hours of wall time once the 5060 Ti is free
**Files to modify:**
- `actproto/configs/trial.yaml` — new run ids and new artifact paths; B0, B1, and B2 entries stay
- `docs/EXPERIMENT_STATUS.md` — append a short pointer; do not rewrite the file

**Depends on:** S1, L1, P1, R1, H1, T1, D1
**Blocks:** none

**Instructions:**
Confirm `results/b0_b2/B2/eval.json` either exists or the B2 process is gone. Then:

1. Run the L1 trainer. Output is `results/adapters/b1_assistant_only/`. Epoch cap is 2. Keep the epoch with higher validation argument correctness. If the log says assistant-only loss failed or the full sequence was trained, stop. Do not overwrite `results/adapters/b1`.
2. Extract step-wise activations into `results/activations_span/`. Rebuild prototypes and the threshold into `results/prototypes_span.npz` and `results/prototype_metrics_span.json`. Refit the probe from that dump. Leave `results/prototypes.npz` and `results/prototype_metrics.json` in place.
3. Add trial runs that point at the new adapter and the new prototype files. Write their reports to new directories under `results/b0_b2/`. Do not overwrite `B0`, `B1`, or `B2`.
4. Score the new adapter alone and with the router on the regenerated test split (n=104). Run `scripts/slice_eval_buckets.py` on the new eval files.
5. Compare argument correctness and task success with B0 (0.269 and 0.000) and B1 (0.024 and 0.010). If tool choice collapses the way B1 did (0.115, step count near 1, answers that are `PARSE_ERROR: multiple tool calls`), the mask is on the wrong span. Fix L1. Do not raise epochs, rank, or the learning rate.
6. Append one status pointer: new adapter path, which epoch was kept, and the new averages. Do not invent a B2 number if that file is still missing.

Pytest for the Sprint 1 files should already be green before this wave starts. Re-run `python -m pytest` once before the GPU job if Sprint 1 touched the tree.

**Definition of Done:**
- [ ] `results/adapters/b1` and `results/b0_b2/{B0,B1,B2}` are unchanged
- [ ] The new adapter directory exists and the training log shows assistant-only loss
- [ ] Validation argument correctness chose epoch 1 or epoch 2, and there is no epoch 3
- [ ] New eval JSON files exist for the masked adapter and the routed run, sliced by bucket
- [ ] `docs/EXPERIMENT_STATUS.md` has a short appended pointer
