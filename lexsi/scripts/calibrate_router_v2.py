#!/usr/bin/env python3
"""Fit a selected adapter's first-hop router, then calibrate on validation only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from actproto.activations.router_calibration import compare_modes, evaluate_bank, fit_bank, read_activations
from actproto.experiments.manifest import hash_file


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, help='hash-manifested train activation directory')
    parser.add_argument('--val', required=True, help='hash-manifested validation activation directory')
    parser.add_argument('--binding', required=True, help='adapter, tokenizer, template, dataset and layer binding JSON')
    parser.add_argument('--selection', required=True, help='validation-selected adapter JSON')
    parser.add_argument('--config', default=str(ROOT / 'actproto/configs/router_v2.yaml'))
    parser.add_argument('--comparisons', help='optional matched validation mode outcomes JSON')
    parser.add_argument('--output', required=True, help='new immutable local artifact directory')
    args = parser.parse_args(argv)
    binding, selection = read_json(args.binding), read_json(args.selection)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if config.get('schema_version') != 'tooluse_v2' or config.get('scope') != 'first_hop_only':
        raise ValueError('invalid router config')
    if config.get('fit_split') != 'train' or config.get('calibration_split') != 'val':
        raise ValueError('router fitting and calibration must use train and validation respectively')
    train_vectors, train_index = read_activations(args.train, 'train', binding)
    val_vectors, val_index = read_activations(args.val, 'val', binding)
    bank = fit_bank(train_vectors, train_index, binding, selection)
    report = evaluate_bank(bank, binding, val_vectors, val_index,
        min_coverage=config['minimum_coverage'], score_grid=config['score_grid'], margin_grid=config['margin_grid'])
    gate = config['gate']
    if gate.get('selection_split') != 'val':
        raise ValueError('router mode selection must use validation')
    if args.comparisons:
        comparisons = read_json(args.comparisons)
        if comparisons.get('split') != 'val' or comparisons.get('binding') != binding:
            raise ValueError('comparison split or binding mismatch')
        expected_tasks = {row['task_id'] for row in val_index}
        if any({row.get('task_id') for row in rows} != expected_tasks
               for rows in comparisons['modes'].values()):
            raise ValueError('comparison tasks must match the calibration validation tasks')
        decision = compare_modes(comparisons['modes'], split='val',
            minimum_improvement=gate['minimum_macro_improvement'],
            max_stratum_loss=gate['maximum_single_or_no_tool_loss'],
            max_latency_ratio=gate['maximum_median_latency_ratio'])
    else:
        decision = {'selected': 'router_off', 'reason': 'validation mode comparisons not supplied'}
    if report['thresholds']['selected'] is None:
        decision['selected'] = 'router_off'
        decision['reason'] = 'minimum coverage was not met'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in [('bank.json', bank), ('calibration.json', report), ('decision.json', decision)]:
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False), encoding='utf-8')
    provenance = {'schema_version': 'tooluse_v2', 'binding': binding,
        'config_sha256': hash_file(args.config), 'selection_sha256': hash_file(args.selection),
        'train_manifest_sha256': hash_file(Path(args.train) / 'manifest.json'),
        'val_manifest_sha256': hash_file(Path(args.val) / 'manifest.json'),
        'comparisons_sha256': hash_file(args.comparisons) if args.comparisons else None,
        'files': {name: hash_file(output / name) for name in ('bank.json', 'calibration.json', 'decision.json')}}
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'selected': decision['selected']}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(1)
