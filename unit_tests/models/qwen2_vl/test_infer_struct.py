from types import SimpleNamespace

import pytest
import torch

from lightllm.core.runner import batch_metadata as batch_metadata
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo


def _patch_base_position_ids(monkeypatch):
    def init_positions(self):
        self.position_ids = torch.tensor([9, 19], dtype=torch.int32)
        self.b_q_seq_len = torch.ones(2, dtype=torch.int32)

    monkeypatch.setattr(batch_metadata, "prepare_sequence_metadata", init_positions)


def _make_model():
    return SimpleNamespace(
        config={"rope_scaling": {}},
        _cos_cached=torch.arange(32, dtype=torch.float32).view(32, 1),
        _sin_cached=torch.arange(32, dtype=torch.float32).view(32, 1),
    )


def test_normal_prompt_prefill_builds_multimodal_positions(monkeypatch):
    _patch_base_position_ids(monkeypatch)
    expected_position_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    monkeypatch.setattr(
        Qwen2VLInferStateInfo,
        "get_mrope_position",
        lambda self, multimodal_params: expected_position_ids,
    )

    infer_state = Qwen2VLInferStateInfo()
    infer_state.is_prefill = True
    infer_state.b_position_delta = None
    infer_state.multimodal_params = [{"images": [], "audios": []}] * 2

    Qwen2VLInferStateInfo.bind_metadata(_make_model())(infer_state)

    assert torch.equal(infer_state.position_ids, expected_position_ids)


def test_normal_decode_applies_position_delta(monkeypatch):
    _patch_base_position_ids(monkeypatch)

    infer_state = Qwen2VLInferStateInfo()
    infer_state.is_prefill = False
    infer_state.b_position_delta = torch.tensor([3, 5], dtype=torch.int32)
    infer_state.multimodal_params = [{"images": [], "audios": []}] * 2

    Qwen2VLInferStateInfo.bind_metadata(_make_model())(infer_state)

    expected_position_ids = torch.tensor([[12, 24]] * 3, dtype=torch.int32)
    assert torch.equal(infer_state.position_ids, expected_position_ids)


def test_prefill_rejects_position_delta(monkeypatch):
    _patch_base_position_ids(monkeypatch)

    infer_state = Qwen2VLInferStateInfo()
    infer_state.is_prefill = True
    infer_state.b_position_delta = torch.tensor([3, 5], dtype=torch.int32)
    infer_state.multimodal_params = [{"images": [], "audios": []}] * 2

    with pytest.raises(AssertionError, match="prefill must not provide b_position_delta"):
        Qwen2VLInferStateInfo.bind_metadata(_make_model())(infer_state)
