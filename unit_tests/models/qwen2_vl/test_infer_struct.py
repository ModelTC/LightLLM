from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo


def _patch_base_position_ids(monkeypatch):
    def init_positions(self, model):
        self.position_ids = torch.tensor([9, 19], dtype=torch.int32)
        self.b_q_seq_len = torch.ones(2, dtype=torch.int32)

    monkeypatch.setattr(InferStateInfo, "init_some_extra_state", init_positions)


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

    infer_state.init_some_extra_state(_make_model())

    assert torch.equal(infer_state.position_ids, expected_position_ids)


def test_normal_decode_applies_position_delta(monkeypatch):
    _patch_base_position_ids(monkeypatch)

    infer_state = Qwen2VLInferStateInfo()
    infer_state.is_prefill = False
    infer_state.b_position_delta = torch.tensor([3, 5], dtype=torch.int32)
    infer_state.multimodal_params = [{"images": [], "audios": []}] * 2

    infer_state.init_some_extra_state(_make_model())

    expected_position_ids = torch.tensor([[12, 24]] * 3, dtype=torch.int32)
    assert torch.equal(infer_state.position_ids, expected_position_ids)


def test_draft_commit_prefill_uses_existing_position_delta(monkeypatch):
    _patch_base_position_ids(monkeypatch)

    infer_state = Qwen2VLInferStateInfo()
    infer_state.is_prefill = True
    infer_state.b_position_delta = torch.tensor([3, 5], dtype=torch.int32)
    infer_state.multimodal_params = [{"images": [], "audios": []}] * 2

    infer_state.init_some_extra_state(_make_model())

    expected_position_ids = torch.tensor([[12, 24]] * 3, dtype=torch.int32)
    assert torch.equal(infer_state.position_ids, expected_position_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_draft_commit_prefill_model_input_accepts_position_delta():
    model_input = ModelInput(
        batch_size=2,
        total_token_num=2,
        max_q_seq_len=1,
        max_kv_seq_len=8,
        input_ids=torch.tensor([1, 2], dtype=torch.int64),
        b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
        b_mtp_index=torch.zeros(2, dtype=torch.int32),
        b_seq_len=torch.tensor([4, 6], dtype=torch.int32),
        b_position_delta=torch.tensor([3, 5], dtype=torch.int32),
        mem_indexes_cpu=torch.tensor([10, 11], dtype=torch.int32),
        is_prefill=True,
        multimodal_params=[{"images": [], "audios": []}] * 2,
    )

    model_input.to_cuda()

    assert model_input.b_position_delta.is_cuda
