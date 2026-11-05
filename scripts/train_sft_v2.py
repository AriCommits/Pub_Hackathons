#!/usr/bin/env python3
"""Preflight or train a manifest-bound v2 supervised adapter."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.experiments.manifest import GpuLease, RunManifest, hash_bytes, hash_file, hash_object, prepare_run
from actproto.training.sft_masks import selected_template
from actproto.training.sft_v2 import preflight, read_rows, train


def load_evaluator(name):
    if not name:
        raise ValueError('real training requires --evaluator module:function')
    module, separator, attribute = name.partition(':')
    if not separator:
        raise ValueError('evaluator must be module:function')
    evaluator = getattr(importlib.import_module(module), attribute)
    if not callable(evaluator):
        raise ValueError('evaluator is not callable')
    return evaluator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='JSONL train conversations')
    parser.add_argument('--config', default=str(ROOT / 'actproto/configs/sft_v2.yaml'))
    parser.add_argument('--manifest', required=True, help='RunManifest JSON')
    parser.add_argument('--output-root', default='results/sft_v2')
    parser.add_argument('--lease-root', default='results/sft_v2/gpu-leases')
    parser.add_argument('--device-uuid')
    parser.add_argument('--evaluator', help='shared validation function as module:function')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    import yaml
    from transformers import AutoTokenizer
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    manifest = RunManifest.from_dict(json.loads(Path(args.manifest).read_text(encoding='utf-8')))
    if manifest.data_hash != hash_file(args.data) or manifest.config_hash != hash_file(args.config):
        raise ValueError('training data or configuration hash mismatch')
    if manifest.model_hash != hash_object({'model_id': config['model_id'], 'revision': config['revision']}):
        raise ValueError('pinned base model hash mismatch')
    if (Path(args.output_root) / manifest.run_id).exists():
        raise FileExistsError('occupied run directory')
    rows = read_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'])
    template = selected_template(tokenizer, config['template_name'])
    if manifest.template_hash != hash_bytes(template.encode('utf-8')):
        raise ValueError('template hash mismatch')
    encoded, report = preflight(tokenizer, rows, config)
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'preflight': report}, indent=2))
        return 0
    evaluator = load_evaluator(args.evaluator)
    if not args.device_uuid:
        raise ValueError('real training requires --device-uuid and a single visible CUDA device')
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('set CUDA_VISIBLE_DEVICES to exactly one GPU before launching')
    detected = str(torch.cuda.get_device_properties(0).uuid).lower()
    if args.device_uuid.lower().removeprefix('gpu-') != detected:
        raise ValueError('device UUID does not match the visible CUDA device')
    with GpuLease(args.lease_root, 'GPU-' + detected) as lease:
        run, token = prepare_run(args.output_root, manifest)
        # The token is secret and needed to resume; never print it or commit the file.
        (run / 'resume-token.txt').write_text(token, encoding='utf-8')
        (run / 'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        outcome = train(tokenizer, encoded, config, run, evaluator, heartbeat=lease.heartbeat)
        (run / 'complete.json').write_text(json.dumps({'manifest_digest': manifest.digest,
            **outcome}, indent=2), encoding='utf-8')
        print(json.dumps({'run': str(run), **outcome}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(1)
