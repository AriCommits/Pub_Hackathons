#!/usr/bin/env python3
"""Run the same v2 evaluation used by SFT checkpoint selection."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.agent.inference_v2 import ModelPolicy
from actproto.evaluation.experiment_runner import evaluate, load_data
from actproto.experiments.manifest import GpuLease, RunManifest, hash_bytes, hash_file, hash_object, prepare_run
from actproto.training.sft_masks import selected_template


def adapter_digest(path):
    if not path:
        return hash_object({'adapter': None})
    root = Path(path)
    if not root.is_dir() or not (root / 'adapter_config.json').is_file():
        raise ValueError('adapter must be a saved adapter directory')
    files = {str(p.relative_to(root)).replace('\\', '/'): hash_file(p) for p in sorted(root.rglob('*')) if p.is_file()}
    return hash_object(files)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    p.add_argument('--run-id', required=True)
    p.add_argument('--matrix', default=str(ROOT / 'actproto/configs/sft_matrix.yaml'))
    p.add_argument('--adapter')
    p.add_argument('--mode', choices=('primary', 'deployed'), default='primary')
    p.add_argument('--final-eval', action='store_true')
    p.add_argument('--selection', help='frozen selection manifest for final evaluation')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output-root', default='results/sft_v2')
    p.add_argument('--lease-root', default='results/sft_v2/gpu-leases')
    p.add_argument('--device-uuid')
    a = p.parse_args(argv)
    import yaml
    matrix = yaml.safe_load(Path(a.matrix).read_text(encoding='utf-8'))
    settings = matrix[a.mode]
    interventions = settings.get('interventions', [])
    if a.mode == 'primary' and interventions:
        raise ValueError('primary mode cannot enable interventions')
    if any(name != 'retry_parse' for name in interventions):
        raise ValueError('unsupported intervention')
    if (settings.get('router') is not False or settings.get('semantic_fallback') is not False
            or settings.get('do_sample') is not False or settings.get('max_new_tokens') != 256):
        raise ValueError('matrix violates primary inference contract')
    digest = adapter_digest(a.adapter)
    if a.split == 'test':
        if not a.selection or not a.final_eval:
            raise ValueError('final test requires --final-eval and --selection')
        selected = json.loads(Path(a.selection).read_text(encoding='utf-8'))
        if selected.get('selected_adapter_sha256') != digest:
            raise ValueError('adapter does not match frozen selection')
    env, tasks = load_data(a.manifest, a.split, final_eval=a.final_eval, selection=a.selection)
    if a.dry_run:
        print(json.dumps({'dry_run': True, 'split': a.split, 'tasks': len(tasks), 'fixture_hash': env.digest,
                          'adapter_sha256': digest}))
        return 0
    if not a.device_uuid:
        raise ValueError('real evaluation requires --device-uuid and one visible GPU')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 1:
        raise ValueError('set CUDA_VISIBLE_DEVICES to exactly one GPU')
    detected = str(torch.cuda.get_device_properties(0).uuid).lower()
    if a.device_uuid.lower().removeprefix('gpu-') != detected:
        raise ValueError('device UUID mismatch')
    config = yaml.safe_load((ROOT / 'actproto/configs/sft_v2.yaml').read_text(encoding='utf-8'))
    with GpuLease(a.lease_root, 'GPU-' + detected) as lease:
        tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'])
        template = selected_template(tokenizer, config['template_name'])
        if a.split == 'test' and selected.get('template_sha256') != hash_bytes(template.encode()):
            raise ValueError('tokenizer template does not match frozen selection')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        manifest = RunManifest(a.run_id, hash_file(a.matrix), hash_file(a.manifest),
            hash_object({'model_id': config['model_id'], 'revision': config['revision'], 'adapter_sha256': digest}), env.digest,
            hash_bytes(template.encode()), hash_file(ROOT / 'actproto/evaluation/verification_v2.py'), revision)
        run, token = prepare_run(a.output_root, manifest)
        (run / 'resume-token.txt').write_text(token, encoding='utf-8')
        model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['revision'],
                                                     dtype=torch.bfloat16, device_map={'': 0})
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
        model.eval()
        metadata = {'split': a.split, 'model_id': config['model_id'], 'model_revision': config['revision'],
                    'adapter_sha256': digest, 'gpu_uuid': detected, 'dtype': 'bfloat16',
                    'generation': settings, 'interventions': interventions, 'torch': torch.__version__,
                    'transformers': importlib.metadata.version('transformers')}
        (run / 'inference.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        records, summary = evaluate(tasks, env, ModelPolicy(model, tokenizer), heartbeat=lease.heartbeat, interventions=interventions)
        with (run / 'traces.jsonl').open('x', encoding='utf-8') as file:
            for record in records:
                file.write(json.dumps(record, sort_keys=True) + '\n')
        summary['trace_sha256'] = hash_file(run / 'traces.jsonl')
        (run / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'run': str(run), **summary}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(1)
