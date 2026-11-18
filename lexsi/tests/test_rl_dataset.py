import json
from dataclasses import replace

import pytest
from actproto.experiments.contracts import TaskSpec
from actproto.training.rl_dataset import load_train_tasks


def task():
    return TaskSpec('t', 'g', 'train', 'question', 'outcome', {'answer': 'private'}, [],
                    {'type': 'exact'}, 2, 'env', 'hash')


def test_load_train_tasks_and_exclude_other_splits(tmp_path):
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps({'task': task().to_dict()}) + '\n', encoding='utf8')
    assert load_train_tasks(path) == [task()]
    path.write_text(json.dumps(replace(task(), split='test').to_dict()) + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='non-train'):
        load_train_tasks(path)


def test_reject_empty_and_duplicate_tasks(tmp_path):
    path = tmp_path / 'tasks.jsonl'
    path.write_text('', encoding='utf8')
    with pytest.raises(ValueError, match='empty'):
        load_train_tasks(path)
    row = json.dumps(task().to_dict())
    path.write_text(row + '\n' + row + '\n', encoding='utf8')
    with pytest.raises(ValueError, match='duplicate'):
        load_train_tasks(path)
