from types import SimpleNamespace

import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo


def test_draft_commit_prefill_uses_existing_position_delta(monkeypatch):
    def init_single_token_prefill(self, model):
        self.position_ids = torch.tensor([9, 19], dtype=torch.int32)
        self.b_q_seq_len = torch.ones(2, dtype=torch.int32)

    monkeypatch.setattr(InferStateInfo, "init_some_extra_state", init_single_token_prefill)

    infer_state = Qwen2VLInferStateInfo()
    infer_state.is_prefill = True
    infer_state.b_position_delta = torch.tensor([3, 5], dtype=torch.int32)
    infer_state.multimodal_params = [{"images": [], "audios": []}] * 2
    model = SimpleNamespace(
        config={"rope_scaling": {}},
        _cos_cached=torch.arange(32, dtype=torch.float32).view(32, 1),
        _sin_cached=torch.arange(32, dtype=torch.float32).view(32, 1),
    )

    infer_state.init_some_extra_state(model)

    expected_position_ids = torch.tensor([[12, 24]] * 3, dtype=torch.int32)
    assert torch.equal(infer_state.position_ids, expected_position_ids)
