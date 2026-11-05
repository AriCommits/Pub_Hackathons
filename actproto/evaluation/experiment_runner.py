"""Manifest-checked benchmark execution shared by validation and CLI."""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from actproto.agent.inference_v2 import ModelPolicy
from actproto.agent.trace_v2 import rollout
from actproto.data.fixtures_v2 import FixtureEnvironment
from actproto.evaluation.metrics_v2 import episode_metrics
from actproto.evaluation.verification_v2 import verify
from actproto.experiments.contracts import TaskSpec
from actproto.experiments.manifest import hash_file


def load_data(manifest_path, split, *, final_eval=False, selection=None, callback=False):
    if split not in ('train', 'val', 'test'):
        raise ValueError('unknown split')
    if callback and split != 'val':
        raise ValueError('training callbacks may load validation only')
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 'tooluse_v2':
        raise ValueError('unsupported dataset manifest')
    if split == 'test':
        if not final_eval or not selection:
            raise ValueError('final test requires explicit unlock and frozen selection')
        frozen = json.loads(Path(selection).read_text(encoding='utf-8'))
        if (frozen.get('status') != 'frozen' or frozen.get('dataset_manifest_sha256') != hash_file(path)
                or not frozen.get('selected_adapter_sha256') or not frozen.get('template_sha256')
                or frozen.get('verifier_sha256') != hash_file(Path(__file__).with_name('verification_v2.py'))):
            raise ValueError('invalid frozen selection manifest')
        with (path.parent / 'final-access.jsonl').open('a', encoding='utf-8') as log:
            import datetime
            log.write(json.dumps({'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  'selection_sha256': hash_file(selection)}) + '\n')
    filenames = ['iid.jsonl', 'stress.jsonl'] if split == 'test' else [f'{split}.jsonl']
    for name in [*filenames, 'fixtures.json']:
        file = path.parent / name
        if not file.is_file() or manifest.get('files', {}).get(name) != hash_file(file):
            raise ValueError(f'dataset file hash mismatch: {name}')
    fixture = json.loads((path.parent / 'fixtures.json').read_text(encoding='utf-8'))
    env = FixtureEnvironment(fixture['identifier'], tuple(fixture['records']), tuple(fixture['sales']), tuple(fixture['documents']))
    if env.digest != manifest.get('fixture_hash') or env.identifier != manifest.get('environment_id'):
        raise ValueError('fixture manifest mismatch')
    tasks = []
    for name in filenames:
        for line in (path.parent / name).read_text(encoding='utf-8').splitlines():
            if line.strip():
                task = TaskSpec.from_dict(json.loads(line)['task'])
                if task.split != split:
                    raise ValueError('task split mismatch')
                tasks.append(task)
    if not tasks:
        raise ValueError('empty benchmark split')
    return env, tasks


def evaluate(tasks, env, policy, *, heartbeat=lambda: None, interventions=()):
    records, strata = [], defaultdict(list)
    for task in tasks:
        heartbeat()
        trace, elapsed = rollout(task, env, policy, interventions=interventions)
        result = verify(task, trace)
        metrics = episode_metrics(task, trace, result)
        metrics['latency_s'] = elapsed
        strata[task.stratum].append(metrics)
        records.append({'task_id': task.task_id, 'stratum': task.stratum, 'trace': trace.to_dict(),
                        'verification': result.to_dict(), 'metrics': metrics})
    per_stratum = {name: mean(x['verified_success_v2'] for x in rows) for name, rows in strata.items()}
    if not per_stratum:
        raise ValueError('no tasks')
    all_metrics = [r['metrics'] for r in records]
    argument = [m['dependency_argument_coverage'] for m in all_metrics if m['dependency_argument_coverage'] is not None]
    summary = {'macro_verified_success_v2': mean(per_stratum.values()),
               'micro_verified_success_v2': mean(m['verified_success_v2'] for m in all_metrics),
               'per_stratum': per_stratum, 'argument_correctness': mean(argument) if argument else 1.0,
               'invalid_calls': sum(m['noncompliant_calls'] + m['invalid_parsing'] for m in all_metrics),
               'generated_tokens': sum(m['generated_tokens'] for m in all_metrics), 'task_count': len(records)}
    return records, summary


def validation_evaluator(checkpoint, *, split='val', model, tokenizer):
    """S3 callback: set ACTPROTO_V2_MANIFEST to the frozen dataset manifest."""
    if split != 'val':
        raise ValueError('training callbacks may load validation only')
    manifest = os.environ.get('ACTPROTO_V2_MANIFEST')
    if not manifest:
        raise ValueError('ACTPROTO_V2_MANIFEST is required')
    env, tasks = load_data(manifest, split, callback=True)
    was_training = model.training
    model.eval()
    try:
        return evaluate(tasks, env, ModelPolicy(model, tokenizer))[1]
    finally:
        model.train(was_training)
