"""Train-only prototype/probe fitting, binding checks and matched validation gate."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from actproto.activations.linear_probe import train_probe
from actproto.activations.prototypes import fit_prototypes
from actproto.activations.similarity import cosine_scores, rank_scores
from actproto.activations.threshold_v2 import calibrate
from actproto.experiments.manifest import hash_file, hash_object
from actproto.tools.registry import canonical_tool_name

_HASH_FIELDS = ('adapter_sha256', 'tokenizer_sha256', 'template_sha256',
                'dataset_sha256', 'fixture_sha256', 'schema_sha256')


def validate_binding(binding):
    if not isinstance(binding, dict) or set(binding) != {*_HASH_FIELDS, 'layer', 'schema_version'}:
        raise ValueError('incomplete router binding')
    if binding['schema_version'] != 'tooluse_v2' or not isinstance(binding['layer'], int) or binding['layer'] < 0:
        raise ValueError('invalid router schema or layer')
    for key in _HASH_FIELDS:
        if not isinstance(binding[key], str) or not re.fullmatch(r'[0-9a-f]{64}', binding[key]):
            raise ValueError(f'invalid router binding {key}')
    return binding


def read_activations(directory, split, expected_binding):
    """Read hash-verified first-hop activations and their explicit provenance."""
    if split not in {'train', 'val'}:
        raise ValueError('activation fitting/calibration forbids test')
    root = Path(directory)
    meta = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    validate_binding(expected_binding)
    if meta.get('split') != split or meta.get('binding') != expected_binding:
        raise ValueError('activation split or router binding mismatch')
    if meta.get('vectors_sha256') != hash_file(root / 'vectors.npz') or meta.get('index_sha256') != hash_file(root / 'index.jsonl'):
        raise ValueError('activation files hash mismatch')
    with np.load(root / 'vectors.npz', allow_pickle=False) as data:
        vectors = np.asarray(data['vectors'], dtype=np.float64)
    index = [json.loads(line) for line in (root / 'index.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if vectors.ndim != 2 or len(vectors) != len(index) or not len(index) or not np.isfinite(vectors).all():
        raise ValueError('invalid activation vectors')
    ids = set()
    for row in index:
        if row.get('split') != split or row.get('hop') != 0:
            raise ValueError('only first-hop activations from the declared split are allowed')
        if not all(isinstance(row.get(k), str) and row[k] for k in ('task_id', 'family', 'gold_tool')):
            raise ValueError('activation index missing task metadata')
        if canonical_tool_name(row['gold_tool']) != row['gold_tool']:
            raise ValueError('activation labels must use canonical tool names')
        if row['task_id'] in ids:
            raise ValueError('duplicate activation task')
        ids.add(row['task_id'])
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError('zero activation vectors cannot define cosine similarity')
    return vectors, index


def fit_bank(vectors, index, binding, selection):
    validate_binding(binding)
    if selection.get('selection_split') != 'val' or selection.get('selected_adapter_sha256') != binding['adapter_sha256'] or selection.get('dataset_sha256') != binding['dataset_sha256']:
        raise ValueError('adapter must be selected on matching validation data before fitting')
    if any(row.get('split') != 'train' or row.get('hop') != 0 for row in index):
        raise ValueError('prototype and probe fitting requires train first-hop activations')
    vectors = np.asarray(vectors, dtype=np.float64)
    if (vectors.ndim != 2 or len(vectors) != len(index) or not len(index)
            or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0)):
        raise ValueError('invalid train activations')
    labels = [row['gold_tool'] for row in index]
    if any(canonical_tool_name(label) != label for label in labels):
        raise ValueError('activation labels must use canonical tool names')
    if len(set(labels)) < 2:
        raise ValueError('at least two tools required for the linear probe')
    prototypes = fit_prototypes(vectors, labels)
    probe, encoder = train_probe(vectors, labels)
    payload = {'schema_version': 'tooluse_v2', 'scope': 'first_hop_only', 'binding': binding,
        'fit_split': 'train', 'train_task_ids': sorted(row['task_id'] for row in index),
        'prototypes': {k: v.tolist() for k, v in sorted(prototypes.items())},
        'probe': {'classes': encoder.classes_.tolist(), 'coef': probe.coef_.tolist(),
                  'intercept': probe.intercept_.tolist()},
        'selection_digest': hash_object(selection)}
    payload['digest'] = hash_object(payload)
    return payload


def validate_bank(bank, binding):
    validate_binding(binding)
    data = dict(bank)
    digest = data.pop('digest', None)
    if digest != hash_object(data) or data.get('binding') != binding:
        raise ValueError('router bank content or binding mismatch')
    if data.get('fit_split') != 'train' or data.get('scope') != 'first_hop_only':
        raise ValueError('invalid router bank scope')
    return bank


def load_bank(path, binding):
    """Never expose a persisted bank before verifying its digest and binding."""
    return validate_bank(json.loads(Path(path).read_text(encoding='utf-8')), binding)


def evaluate_bank(bank, binding, vectors, index, *, min_coverage, score_grid, margin_grid):
    validate_bank(bank, binding)
    if any(row.get('split') != 'val' or row.get('hop') != 0 for row in index):
        raise ValueError('router calibration requires validation first-hop activations')
    vectors = np.asarray(vectors, dtype=np.float64)
    if (vectors.ndim != 2 or len(vectors) != len(index) or not len(index)
            or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0)):
        raise ValueError('invalid validation activations')
    if set(bank['train_task_ids']) & {r['task_id'] for r in index}:
        raise ValueError('train and validation task overlap')
    prototypes = {k: np.asarray(v) for k, v in bank['prototypes'].items()}
    scored = []
    for vector, row in zip(vectors, index, strict=True):
        rank = rank_scores(cosine_scores(vector, prototypes))
        scored.append({'gold': row['gold_tool'], 'predicted': rank['top1'],
                       'score': rank['top1_score'], 'margin': rank['margin'], 'family': row['family']})
    thresholds = calibrate(scored, split='val', min_coverage=min_coverage,
                           score_grid=score_grid, margin_grid=margin_grid)
    probe = bank['probe']
    logits = np.asarray(vectors) @ np.asarray(probe['coef']).T + np.asarray(probe['intercept'])
    prediction = ((logits[:, 0] > 0).astype(int) if len(probe['classes']) == 2
                  else logits.argmax(axis=1))
    families = defaultdict(lambda: {'total': 0, 'correct': 0, 'confusion': {}})
    correct = 0
    for i, row in zip(prediction, index, strict=True):
        predicted = probe['classes'][int(i)]
        hit = int(predicted == row['gold_tool'])
        correct += hit
        family = families[row['family']]
        family['total'] += 1
        family['correct'] += hit
        key = row['gold_tool'] + ' -> ' + predicted
        family['confusion'][key] = family['confusion'].get(key, 0) + 1
    return {'binding': binding, 'bank_digest': bank['digest'], 'scope': 'first_hop_only',
            'thresholds': thresholds, 'probe': {'split': 'val', 'accuracy': correct / len(index),
                                                'n': len(index), 'families': dict(families)}}


def compare_modes(modes, *, split, minimum_improvement=.05, max_stratum_loss=.02, max_latency_ratio=1.5):
    """Apply predeclared gate to matched per-task outcomes; missing evidence routes off."""
    if split != 'val':
        raise ValueError('router selection requires validation')
    required = ('router_off', 'router_no_fallback', 'deployed')
    if set(modes) != set(required):
        return {'selected': 'router_off', 'reason': 'missing comparison modes'}
    indexed = {}
    for mode in required:
        rows = modes[mode]
        if not rows or any(not isinstance(r.get('success'), bool) or not isinstance(r.get('latency_s'), (int, float))
                           or not math.isfinite(r['latency_s']) or r['latency_s'] < 0
                           or not isinstance(r.get('fallback_count'), int) or r['fallback_count'] < 0
                           or not r.get('task_id') or not r.get('stratum') or not r.get('guard_class') for r in rows):
            raise ValueError('invalid mode outcome records')
        indexed[mode] = {r['task_id']: r for r in rows}
        if len(indexed[mode]) != len(rows):
            raise ValueError('duplicate task in mode outcomes')
    if any(set(indexed[m]) != set(indexed['router_off']) for m in required):
        raise ValueError('mode comparisons must contain the same tasks')
    baseline = indexed['router_off']
    for mode in required:
        if any((r['stratum'], r['guard_class']) != (baseline[k]['stratum'], baseline[k]['guard_class'])
               for k, r in indexed[mode].items()):
            raise ValueError('task class mismatch across modes')
    reports = {}
    for mode in required:
        rows = list(indexed[mode].values())
        strata = defaultdict(list)
        for row in rows:
            strata[row['stratum']].append(row['success'])
        reports[mode] = {'macro_verified_success_v2': mean(mean(v) for v in strata.values()),
                         'per_stratum': {k: mean(v) for k, v in strata.items()},
                         'median_latency_s': median(r['latency_s'] for r in rows),
                         'fallback_frequency': sum(r['fallback_count'] > 0 for r in rows) / len(rows)}
    if any(r['fallback_count'] for mode in ('router_off', 'router_no_fallback') for r in indexed[mode].values()):
        raise ValueError('fallback-disabled mode contains fallback interventions')
    off, routed = reports['router_off'], reports['router_no_fallback']
    base_latency = off['median_latency_s']
    if base_latency <= 0:
        return {'selected': 'router_off', 'reason': 'baseline latency unavailable', 'modes': reports}
    losses = {}
    for category in ('single_tool', 'no_tool'):
        ids = [k for k, r in baseline.items() if r['guard_class'] == category]
        if not ids:
            return {'selected': 'router_off', 'reason': f'missing {category} guard', 'modes': reports}
        losses[category] = mean(baseline[k]['success'] for k in ids) - mean(indexed['router_no_fallback'][k]['success'] for k in ids)
    passed = (routed['macro_verified_success_v2'] - off['macro_verified_success_v2'] >= minimum_improvement
              and max(losses.values()) <= max_stratum_loss
              and routed['median_latency_s'] <= max_latency_ratio * base_latency)
    return {'selected': 'router_no_fallback' if passed else 'router_off',
            'reason': 'validation gate passed' if passed else 'validation gate failed',
            'guard_losses': losses, 'modes': reports}
