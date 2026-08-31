from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import (
    Deepseek2TransformerLayerInfer,
)


@pytest.mark.parametrize("is_prefill", [False, True])
def test_tp_moe_propagates_prefill_stage(is_prefill):
    captured = {}

    class Gate:
        data_type_ = torch.float32

        @staticmethod
        def mm(hidden_states):
            return torch.zeros((hidden_states.shape[0], 4))

    class Experts:
        @staticmethod
        def experts(*args, **kwargs):
            captured.update(kwargs)

    layer = Deepseek2TransformerLayerInfer.__new__(Deepseek2TransformerLayerInfer)
    layer.embed_dim_ = 4
    layer.n_shared_experts = None
    layer.num_experts_per_tok = 2
    layer.norm_topk_prob = True
    layer.n_group = 1
    layer.topk_group = 1
    layer_weight = SimpleNamespace(
        moe_gate=Gate(),
        experts=Experts(),
        num_fused_shared_experts=0,
    )
    infer_state = SimpleNamespace(is_prefill=is_prefill)

    output = layer._moe_ffn_tp(torch.zeros((3, 4)), infer_state, layer_weight)

    assert output.shape == (3, 4)
    assert captured["is_prefill"] is is_prefill
    assert captured["infer_state"] is infer_state
