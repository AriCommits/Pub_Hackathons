from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TinyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x) + x


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, d: int = 8, n: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TinyBlock(d) for _ in range(n)])
        self.lm_head = nn.Linear(d, vocab)
        self.n_layers = n

    def forward(self, input_ids=None, **kwargs):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return SimpleNamespace(last_hidden_state=x, logits=self.lm_head(x))


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    dest = tmp_path / "results"
    dest.mkdir()
    return dest


@pytest.fixture
def tiny_lm() -> TinyLM:
    torch.manual_seed(0)
    model = TinyLM()
    model.eval()
    return model
