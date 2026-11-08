import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "test/benchmark/qwen38"))
    return importlib.import_module("score_reference")


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        prior = kwargs["past_key_values"] or ()
        ids = tuple(kwargs["input_ids"][0].tolist())
        positions = kwargs["position_ids"][0].tolist()
        assert positions == list(range(len(prior), len(prior) + len(ids)))
        assert kwargs["attention_mask"].tolist() == [[1] * (len(prior) + len(ids))]
        assert kwargs["use_cache"] and kwargs["logits_to_keep"] == 1
        self.calls.append(kwargs)
        history = prior + ids
        return SimpleNamespace(past_key_values=history, logits=torch.tensor([[sum(history)]]))


@pytest.mark.parametrize("chunk_size,expected_calls", [(None, 1), (1, 9), (4, 3), (16, 1)])
def test_prefill_chunking_preserves_history_and_next_decode_position(scorer, chunk_size, expected_calls):
    model = RecordingModel()
    inputs = torch.arange(9).unsqueeze(0)
    output = scorer.forward_reference(model, inputs, None, 9, chunk_size)
    assert len(model.calls) == expected_calls
    assert output.past_key_values == tuple(range(9))
    assert output.logits.item() == 36
    output = scorer.forward_reference(model, torch.tensor([[17]]), output.past_key_values, 10, chunk_size)
    assert output.past_key_values == (*range(9), 17)
    assert output.logits.item() == 53


@pytest.mark.parametrize("count,length,chunk_size", [(0, 0, None), (2, 1, 1), (2, 2, 0)])
def test_invalid_forward_lengths_are_rejected_before_call(scorer, count, length, chunk_size):
    model = RecordingModel()
    with pytest.raises(ValueError, match="valid lengths"):
        scorer.forward_reference(model, torch.zeros((1, count)), None, length, chunk_size)
    assert not model.calls
