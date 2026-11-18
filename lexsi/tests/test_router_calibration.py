import json
from copy import deepcopy

import numpy as np
import pytest

from actproto.activations.router_calibration import (compare_modes, evaluate_bank, fit_bank,
    load_bank, read_activations, validate_bank)
from actproto.activations.threshold_v2 import calibrate
from actproto.experiments.manifest import hash_file, hash_object


def binding():
    return {'schema_version': 'tooluse_v2', 'layer': 8, **{key: hash_object(key) for key in
        ('adapter_sha256', 'tokenizer_sha256', 'template_sha256', 'dataset_sha256',
         'fixture_sha256', 'schema_sha256')}}


def examples(split):
    vectors = np.array([[1., 0.], [0., 1.], [1., .1], [.1, 1.]])
    index = [{'split': split, 'hop': 0, 'task_id': f'{split}-{i}', 'family': 'one' if i < 2 else 'two',
              'gold_tool': 'calculator' if i % 2 == 0 else 'query_database'} for i in range(4)]
    return vectors, index


def selected(b):
    return {'selection_split': 'val', 'selected_adapter_sha256': b['adapter_sha256'],
            'dataset_sha256': b['dataset_sha256']}


def test_ties_prefer_margin_then_score_and_keep_curve():
    records = [{'gold': 'calculator', 'predicted': 'calculator', 'score': .99,
                'margin': .98, 'family': 'one'}]
    result = calibrate(records, split='val', min_coverage=1, score_grid=[0., .9], margin_grid=[0., .9])
    assert result['selected']['min_margin'] == .9 and result['selected']['min_score'] == .9
    assert len(result['risk_coverage_curve']) == 4
    assert result['families']['one']['confusion'] == {'calculator -> calculator': 1}
    with pytest.raises(ValueError, match='validation'):
        calibrate(records, split='test', min_coverage=1, score_grid=[0], margin_grid=[0])


def test_fit_and_calibration_split_and_binding_guards():
    b = binding()
    train_vectors, train_index = examples('train')
    val_vectors, val_index = examples('val')
    bank = fit_bank(train_vectors, train_index, b, selected(b))
    report = evaluate_bank(bank, b, val_vectors, val_index, min_coverage=.5,
                           score_grid=[0., .5], margin_grid=[0., .2])
    assert report['scope'] == 'first_hop_only'
    assert report['probe']['accuracy'] == 1.0
    assert report['thresholds']['selected']['coverage'] == 1.0
    with pytest.raises(ValueError, match='binding'):
        validate_bank(bank, dict(b, adapter_sha256=hash_object('another')))
    tampered = deepcopy(bank)
    tampered['probe']['intercept'][0] += .1
    with pytest.raises(ValueError, match='content'):
        validate_bank(tampered, b)
    with pytest.raises(ValueError, match='train'):
        fit_bank(val_vectors, val_index, b, selected(b))
    with pytest.raises(ValueError, match='validation'):
        evaluate_bank(bank, b, train_vectors, train_index, min_coverage=.5, score_grid=[0], margin_grid=[0])
    with pytest.raises(ValueError, match='selected'):
        fit_bank(train_vectors, train_index, b, dict(selected(b), selection_split='test'))
    overlap = deepcopy(val_index)
    overlap[0]['task_id'] = train_index[0]['task_id']
    with pytest.raises(ValueError, match='overlap'):
        evaluate_bank(bank, b, val_vectors, overlap, min_coverage=.5, score_grid=[0], margin_grid=[0])


def test_hashed_activation_manifest_rejects_modifications(tmp_path):
    b = binding()
    vectors, rows = examples('train')
    np.savez(tmp_path / 'vectors.npz', vectors=vectors)
    index_path = tmp_path / 'index.jsonl'
    index_path.write_text('\n'.join(json.dumps(x) for x in rows), encoding='utf-8')
    manifest = {'split': 'train', 'binding': b, 'vectors_sha256': hash_file(tmp_path / 'vectors.npz'),
                'index_sha256': hash_file(index_path)}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert len(read_activations(tmp_path, 'train', b)[1]) == 4
    with pytest.raises(ValueError, match='split'):
        read_activations(tmp_path, 'val', b)
    index_path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        read_activations(tmp_path, 'train', b)


def outcomes():
    off = [{'task_id': str(i), 'stratum': 's' if i < 2 else 'r',
            'guard_class': 'single_tool' if i == 0 else ('no_tool' if i == 1 else 'other'),
            'success': i != 2, 'latency_s': 1., 'fallback_count': 0} for i in range(4)]
    routed = deepcopy(off)
    routed[2]['success'] = True
    deployed = deepcopy(routed)
    deployed[3]['fallback_count'] = 1
    return {'router_off': off, 'router_no_fallback': routed, 'deployed': deployed}


def test_gate_matched_modes_fallback_and_failure_default_off():
    modes = outcomes()
    result = compare_modes(modes, split='val')
    assert result['selected'] == 'router_no_fallback'
    assert result['modes']['deployed']['fallback_frequency'] == .25
    modes['router_no_fallback'][0]['success'] = False
    assert compare_modes(modes, split='val')['selected'] == 'router_off'
    assert compare_modes({}, split='val')['selected'] == 'router_off'
    modes = outcomes()
    modes['router_no_fallback'][0]['fallback_count'] = 1
    with pytest.raises(ValueError, match='fallback'):
        compare_modes(modes, split='val')
    with pytest.raises(ValueError, match='validation'):
        compare_modes(outcomes(), split='test')


def test_cli_end_to_end_and_binding_checked_load(tmp_path):
    from lexsi.scripts.calibrate_router_v2 import main
    b = binding()
    for split in ('train', 'val'):
        root = tmp_path / split
        root.mkdir()
        vectors, rows = examples(split)
        np.savez(root / 'vectors.npz', vectors=vectors)
        (root / 'index.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
        metadata = {'split': split, 'binding': b, 'vectors_sha256': hash_file(root / 'vectors.npz'),
                    'index_sha256': hash_file(root / 'index.jsonl')}
        (root / 'manifest.json').write_text(json.dumps(metadata), encoding='utf-8')
    bind_path, selection_path = tmp_path / 'binding.json', tmp_path / 'selection.json'
    bind_path.write_text(json.dumps(b), encoding='utf-8')
    selection_path.write_text(json.dumps(selected(b)), encoding='utf-8')
    output = tmp_path / 'run'
    args = ['--train', str(tmp_path / 'train'), '--val', str(tmp_path / 'val'),
            '--binding', str(bind_path), '--selection', str(selection_path), '--output', str(output)]
    assert main(args) == 0
    assert load_bank(output / 'bank.json', b)['fit_split'] == 'train'
    assert json.loads((output / 'decision.json').read_text())['selected'] == 'router_off'
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(ValueError, match='binding'):
        load_bank(output / 'bank.json', dict(b, template_sha256=hash_object('other')))
    comparisons = tmp_path / 'comparisons.json'
    comparisons.write_text(json.dumps({'split': 'val', 'binding': b, 'modes': outcomes()}), encoding='utf-8')
    args[-1] = str(tmp_path / 'another-run')
    with pytest.raises(ValueError, match='validation tasks'):
        main([*args, '--comparisons', str(comparisons)])
