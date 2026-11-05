"""Collect diagnostic fixture-policy rollouts; this command never trains a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actproto.data.fixtures_v2 import build_environment
from actproto.training.rl_dataset import load_scoped_train_manifest
from actproto.training.rl_rollouts import (PolicySample, TrustedRolloutAdapter, append_abandoned_group, append_batch_event, append_new_rollouts, collect_episode,
                                             read_rollouts, rollout_id, rollout_seed, task_digest)


def backend_fixture_smoke(target: Path) -> dict:
    """Exercise real installed trainer with fake CPU decisions; never optimize."""
    import tempfile
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from actproto.experiments.contracts import TaskSpec
    from actproto.integrations.agenttune_backend import create_trainer

    tokenizer = AutoTokenizer.from_pretrained('CohereLabs/tiny-aya-global', local_files_only=True)
    base = build_environment(1)
    task = TaskSpec('smoke-calculator', 'smoke-group', 'train', 'What is 1+1?',
        'outcome', {'answer': '2'}, [], {'type': 'exact'}, 3, base.identifier, base.digest)
    # Build a separately manifest-verified synthetic train-only scope.
    from actproto.experiments.manifest import hash_file
    with tempfile.TemporaryDirectory() as data_dir:
        root = Path(data_dir)
        fixture = {'identifier': base.identifier, 'records': base.records,
                   'sales': base.sales, 'documents': base.documents}
        (root / 'fixtures.json').write_text(json.dumps(fixture), encoding='utf-8')
        example = {'task': task.to_dict(), 'entity': base.records[0]['id'], 'axis': 'train'}
        (root / 'train.jsonl').write_text(json.dumps(example) + '\n', encoding='utf-8')
        manifest = {'schema_version': 'tooluse_v2', 'environment_id': base.identifier,
                    'fixture_hash': base.digest,
                    'files': {name: hash_file(root / name) for name in ('fixtures.json', 'train.jsonl')}}
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        scoped, loaded_tasks = load_scoped_train_manifest(path)
        if loaded_tasks != [task]:
            raise ValueError('synthetic training fixture task mismatch')
    records = []
    def sink(row):
        records.append(row)
        return append_new_rollouts(target, [row])
    def policy_factory(seed, trainer):
        def generate(ids, budget):
            prompt = tokenizer.decode(ids)
            action = ({'name': 'finish', 'arguments': {'answer': '2'}} if 'Tool result:' in prompt else
                      {'name': 'calculator', 'arguments': {'expression': '1+1'}})
            text = json.dumps(action, separators=(',', ':'))
            sampled = tokenizer.encode(text, add_special_tokens=False)
            return PolicySample(text, sampled, [-0.5] * len(sampled))
        return generate
    adapter = TrustedRolloutAdapter(tokenizer, [task], policy_factory, lambda: scoped,
        run_id='r1-backend-fixture-v1', policy_hash=hashlib.sha256(b'r1-fixed-fixture-policy-v1').hexdigest(),
        record_sink=sink, sampling_mode='synthetic_fixture',
        batch_event_sink=lambda event: append_batch_event(target.with_suffix('.batch.jsonl'), event),
        abandonment_sink=lambda event: append_abandoned_group(target.with_suffix('.resolution.jsonl'), event))
    torch.manual_seed(42)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_layer=1, n_head=1, n_embd=8,
        n_positions=1024, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id))
    with tempfile.TemporaryDirectory() as directory:
        wrapper = create_trainer(model=model, processing_class=tokenizer,
            reward_funcs=lambda completions, **kwargs: [float(i % 2) for i in range(len(completions))],
            train_dataset=Dataset.from_dict({'prompt': [task.prompt, task.prompt]}),
            rollout_func=adapter, output_dir=directory, num_generations=2,
            per_device_train_batch_size=2, gradient_accumulation_steps=1,
            bf16=False, fp16=False, report_to='none', use_cpu=True, beta=0.0)
        wrapper.setup_data()
        wrapper.setup_trainer()
        trainer = wrapper.trainer
        out = trainer._generate_and_score_completions([{'prompt': task.prompt}, {'prompt': task.prompt}])
        trainer_type = type(trainer).__module__ + '.' + type(trainer).__name__
        if trainer_type != 'trl.trainer.grpo_trainer.GRPOTrainer':
            raise ValueError('unexpected backend trainer')
        if out['tool_mask'].shape != out['completion_ids'].shape:
            raise ValueError('environment mask shape mismatch')
    if len(records) != 2 or not all(
        row['trace']['schema_version'] == 'tooluse_v2' and row['completion_reason'] == 'finish'
        and [event.get('tool') for event in row['trace']['events']] == ['calculator', 'finish']
        for row in records):
        raise ValueError('backend fixture failed to execute both v2 tool episodes')
    return {'mode': 'backend_fixture_collect_only', 'trainer': trainer_type,
            'episodes': len(records), 'output': str(target), 'optimizer_steps': 0,
            'policy_logprobs': 'synthetic fixture values'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', help='Plan 9 benchmark manifest with sealed train split')
    parser.add_argument('--output', help='JSONL under results/rl_v2')
    parser.add_argument('--run-id')
    parser.add_argument('--group-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backend-fixture-smoke', action='store_true')
    parser.add_argument('--fixture-policy', action='store_true',
                        help='use the fixed CPU diagnostic policy; no model inference')
    args = parser.parse_args(argv)
    if args.backend_fixture_smoke:
        if args.output is None:
            parser.error('--output is required')
    elif not args.fixture_policy or not args.manifest or not args.output or not args.run_id:
        parser.error('fixture collection requires --fixture-policy, --manifest, --output, and --run-id')
    if args.group_size < 2:
        parser.error('--group-size must be at least 2')
    target = Path(args.output).resolve()
    root = (Path(__file__).resolve().parents[1] / 'results' / 'rl_v2').resolve()
    if not target.is_relative_to(root) or target == root:
        parser.error('--output must be a file under results/rl_v2')
    if args.backend_fixture_smoke:
        print(json.dumps(backend_fixture_smoke(target)))
        return 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('CohereLabs/tiny-aya-global', local_files_only=True)
    scoped, tasks = load_scoped_train_manifest(args.manifest)
    complete = read_rollouts(target)
    fixture_hash = hashlib.sha256(b'fixture-calculator-1-plus-1-then-finish-2-v1').hexdigest()
    def policy_factory(seed):
        calls = 0
        def generate(prompt_ids, token_budget):
            nonlocal calls
            calls += 1
            value = ({'name': 'calculator', 'arguments': {'expression': '1+1'}} if calls == 1 else
                     {'name': 'finish', 'arguments': {'answer': '2'}})
            text = json.dumps(value, separators=(',', ':'))
            return PolicySample(text, tokenizer.encode(text, add_special_tokens=False))
        return generate
    count = 0
    for task in tasks:
        for index in range(args.group_size):
            key = rollout_id(args.run_id, task.task_id, 0, index)
            if key in complete:
                old = complete[key]
                if (old.get('sampling_mode') != 'synthetic_fixture' or old.get('policy_hash') != fixture_hash or old.get('task_hash') != task_digest(task)
                        or old.get('seed') != rollout_seed(args.run_id, task.task_id, 0, index, args.seed)):
                    raise ValueError('existing rollout provenance does not match requested run')
                continue
            row = collect_episode(task, run_id=args.run_id, group_index=0, sample_index=index,
                seed=args.seed, policy_hash=fixture_hash, tokenizer=tokenizer,
                policy_factory=policy_factory,
                environment_factory=lambda: scoped, fixture_only=True)
            row['sampling_mode'] = 'synthetic_fixture'
            count += append_new_rollouts(target, [row])
    print(json.dumps({'mode': 'fixture_collect_only', 'new_rollouts': count, 'output': str(target)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
