# AuditKIT compatibility and v2 export

AuditKIT core 1.1.3, source commit `71316da345784ff220d5ed59337441a59ab31526`, was installed into a separate virtual environment from `requirements-auditkit.lock`. It has zero mandatory runtime dependencies and uses the source-available, noncommercial LSAL-1.2 license in the upstream repository. The tested API is `auditkit.agent_eval.AgentEvalRunner.run(AgentEvalSpec(mode="recorded", ...))` with a versioned `CustomPredicate`. The predicate reconstructs the locally captured `ExecutionTrace` from the recorded episode and calls the shared verifier; AuditKIT's verdict is the outcome. The version identity is the verifier's SHA-256. The runner checks the installed distribution's version and Git commit through `direct_url.json`; other builds fail rather than silently masquerading as the tested version. A model-supplied provenance string cannot authenticate observations; callers must supply traces captured by the local harness.

Create a new isolated environment and check it without loading a model:

```powershell
python -m venv .venv-auditkit
.\.venv-auditkit\Scripts\python.exe -m pip install -r requirements-auditkit.lock
.\.venv-auditkit\Scripts\python.exe -m scripts.check_toolkit_compat --submission --run-manifest results/sft_v2/auditkit-compat.json
```

To rescore an actual S4 train or validation run with the same backend, pass `--dataset-manifest <dataset-manifest.json> --split val --traces <run/traces.jsonl>` together with `--submission --run-manifest <new-output.json>`. The tool requires an exact one-to-one match between the selected split and recorded trace IDs. Final-test rescoring is intentionally not exposed here; the joint final-evaluation unlock belongs to the later handoff.

The last command writes a new manifest with the actual backend, version, revision, invocation, verifier hash and verdict; it fails if the output already exists, the package is missing/mismatched, or submission is requested with local mode. Create the output parent first. `--backend local` is an explicit development-only option. The recorded-mode test compares successful, failed, valid alternative, forged-source and skipped-required-tool traces with the local verifier; the separate environment must run `tests/test_auditkit_v2.py` to exercise the installed backend. The project's normal suite skips that one integration test when AuditKIT is absent.

After the S7 study freezes a selected model and dataset, prepare a `tooluse_v2` export manifest containing `artifacts` entries `{ "role": ..., "path": ..., "sha256": ... }`. Paths are relative to an explicit root. Required roles are `adapter`, `tokenizer`, `template`, `data_manifest`, `fixture_manifest`, `curriculum`, `traces`, `metrics`, `environment_lock`, and `model_card`; adapters require config and weights, tokenizers require config and tokenizer data. A selected router requires both `prototype` and `threshold`. No final-test answers or test traces may appear in public roles. `private_final` is permitted only with `joint_evaluation_complete: true` and explicit `--include-private`; use a separate private destination. Credential-like material, caches, hidden paths, symlinks, unlisted files, hashes that do not match and missing dependencies fail closed. Recognition is heuristic, so a human must review the allowlist before distribution.

```powershell
python -m scripts.export_experiment_v2 --root <artifact-root> --manifest <export-manifest.json>
python -m scripts.export_experiment_v2 --root <artifact-root> --manifest <export-manifest.json> --destination <new-local-bundle>
```

The first command only lists selected paths; the second produces a new local bundle and verifies copied hashes. The bundle contains its export manifest. No hosting or upload is performed. The role and filename checks establish dependency completeness but do not prove a checkpoint can actually be loaded: a real selected checkpoint/tokenizer and frozen experiment manifest do not exist yet. S7 must reload the selected bundle and report that test. The final data→train→evaluate recipe depends on S7's actual selected files and must be recorded in the S8 reproducibility handoff; no unexecuted study or final evaluation is claimed here.
