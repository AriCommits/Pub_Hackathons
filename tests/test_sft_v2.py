import json

import pytest

from actproto.training.sft_v2 import preflight, read_rows, select_checkpoint, selection_key, train
from test_sft_masks import ToyTokenizer


def row(split='train'):
    return {'split': split, 'messages': [{'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': json.dumps({'name': 'finish', 'arguments': {'answer': 'x'}})}]}


def config():
    return {'epochs': 2, 'max_length': 10000, 'template_name': 'default',
            'dtype': 'bfloat16', 'quantization': 'none'}


def test_train_only_rows_and_preflight(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text(json.dumps(row()), encoding='utf-8')
    rows = read_rows(path)
    encoded, report = preflight(ToyTokenizer(), rows, config())
    assert report['accepted_rows'] == 1 and report['excluded_rows'] == 0
    assert any(x != -100 for x in encoded[0]['labels'])
    path.write_text(json.dumps(row('test')), encoding='utf-8')
    with pytest.raises(ValueError, match='train'):
        read_rows(path)
    with pytest.raises(ValueError, match='no trainable'):
        preflight(ToyTokenizer(), rows, dict(config(), max_length=1))


def test_selection_validation_only_uses_predeclared_tiebreak():
    def candidate(name, success, arguments, invalid, tokens, split='val'):
        return {'checkpoint': name, 'split': split, 'metrics':
            {'macro_verified_success_v2': success, 'argument_correctness': arguments,
             'invalid_calls': invalid, 'generated_tokens': tokens}}
    a = candidate('a', .5, .9, 3, 20)
    b = candidate('b', .5, .9, 2, 30)
    assert select_checkpoint([a, b]) is b
    with pytest.raises(ValueError, match='validation'):
        select_checkpoint([candidate('bad', 1, 1, 0, 1, split='test')])
    with pytest.raises(ValueError, match='finite'):
        selection_key(candidate('bad', float('nan'), 1, 0, 1)['metrics'])


def test_mock_trainer_saves_and_reloads_selected_checkpoint(tmp_path, monkeypatch):
    import sys
    import types
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    class Model:
        def __init__(self):
            self.config = types.SimpleNamespace(use_cache=True)
            self.weight = 0

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Model()

    class Trainer:
        def __init__(self, model, args, train_dataset, data_collator, callbacks):
            self.model, self.args, self.callback = model, args, callbacks[0]
            self.state = types.SimpleNamespace(global_step=0)
            assert data_collator(train_dataset[:1])['labels'].shape[0] == 1

        def train(self):
            for step in (1, 2):
                self.model.weight += 1
                self.state.global_step = step
                checkpoint = tmp_path / f'checkpoint-{step}'
                checkpoint.mkdir()
                (checkpoint / 'adapter_config.json').write_text(json.dumps({'weight': self.model.weight}))
                self.callback.on_save(self.args, self.state, None, model=self.model)
            return types.SimpleNamespace(training_loss=0.5)

        def save_state(self):
            (tmp_path / 'trainer_state.json').write_text('{}')

    class TrainingArguments:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        AutoModelForCausalLM=AutoModel, Trainer=Trainer, TrainerCallback=object,
        TrainingArguments=TrainingArguments))
    monkeypatch.setitem(sys.modules, 'peft', types.SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs, get_peft_model=lambda model, config: model))
    cfg = dict(config(), model_id='model', revision='revision', rank=16, alpha=32,
               dropout=.05, learning_rate=.0002, batch_size=1,
               gradient_accumulation_steps=8, seed=42)
    tokenizer = ToyTokenizer()
    tokenizer.pad_token_id = 0
    encoded, _ = preflight(tokenizer, [row()], cfg)
    def evaluate(checkpoint, *, split, model, tokenizer):
        assert split == 'val'
        return {'macro_verified_success_v2': model.weight / 10, 'argument_correctness': 1,
                'invalid_calls': 0, 'generated_tokens': 1}
    result = train(tokenizer, encoded, cfg, tmp_path, evaluate)
    saved = json.loads((tmp_path / 'checkpoint-2' / 'adapter_config.json').read_text())
    assert result['selected']['checkpoint'].endswith('checkpoint-2')
    assert saved['weight'] == 2 and (tmp_path / 'trainer_state.json').exists()


def test_cli_dry_run_preflights_without_loading_model(tmp_path, monkeypatch):
    import sys
    import types
    import yaml
    from actproto.experiments.manifest import RunManifest, hash_bytes, hash_file, hash_object
    from actproto.training.sft_masks import _BRANCH
    from scripts.train_sft_v2 import main
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return ToyTokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    cfg = dict(config(), model_id='model', revision='pinned')
    config_path, data_path = tmp_path / 'config.yaml', tmp_path / 'data.jsonl'
    config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    data_path.write_text(json.dumps(row()), encoding='utf-8')
    manifest = RunManifest('dry', hash_file(config_path), hash_file(data_path),
        hash_object({'model_id': 'model', 'revision': 'pinned'}), hash_object('fixture'),
        hash_bytes(_BRANCH.encode()), hash_object('verifier'), 'abc')
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding='utf-8')
    output = tmp_path / 'runs'
    args = ['--data', str(data_path), '--config', str(config_path), '--manifest', str(manifest_path),
            '--output-root', str(output), '--dry-run']
    assert main(args) == 0
    assert not output.exists()
    (output / 'dry').mkdir(parents=True)
    with pytest.raises(FileExistsError, match='occupied'):
        main(args)
