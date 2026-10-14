from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_infer import base_layer_infer
from lightllm.common.kv_cache_mem_manager import Qwen3NextMemManager
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer


@pytest.mark.parametrize("prefill,capture", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("mix_mode", [False, True])
def test_linear_attention_preserves_tp_and_sp_token_layout(monkeypatch, prefill, capture, mix_mode):
    layer = Qwen3NextTransformerLayerInfer.__new__(Qwen3NextTransformerLayerInfer)
    layer.is_linear_attention_layer = True
    layer.tp_world_size_ = 2
    layer.layer_num_ = 0
    layer.alloc_tensor = lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
    monkeypatch.setattr(base_layer_infer, "get_env_start_args", lambda: SimpleNamespace(enable_tpsp_mix_mode=mix_mode))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)

    full_input = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    rank_input = full_input[:2] if mix_mode else full_input
    calls = []

    def gather(output, input, **kwargs):
        calls.append("gather")
        torch.testing.assert_close(input, full_input[:2])
        output.copy_(full_input)

    def reduce_scatter(output, input, **kwargs):
        calls.append("reduce_scatter")
        torch.testing.assert_close(input, full_input + 1)
        output.copy_(input[:2] * 2)

    def all_reduce(input, **kwargs):
        calls.append("all_reduce")
        input.mul_(2)

    monkeypatch.setattr(base_layer_infer, "all_gather_into_tensor", gather)
    monkeypatch.setattr(base_layer_infer, "reduce_scatter_tensor", reduce_scatter)
    monkeypatch.setattr(base_layer_infer, "all_reduce", all_reduce)

    def project(input, weight):
        calls.append("project")
        torch.testing.assert_close(input, full_input)
        return input

    def attention(**kwargs):
        calls.append("attention")
        control = kwargs["att_control"]
        params = control.linear_att_prefill_dict if prefill else control.linear_att_decode_dict
        torch.testing.assert_close(params["mixed_qkvzba"], full_input)
        return full_input + 1, None

    def captured_attention(input, state, weight):
        calls.append("captured_attention")
        torch.testing.assert_close(input, full_input)
        return input + 1, None

    layer._linear_in_proj = project
    layer._linear_post = lambda output, z, weight: output
    layer._linear_prefill_cuda_graph_wrapper = captured_attention
    state = SimpleNamespace(
        mem_manager=Qwen3NextMemManager.__new__(Qwen3NextMemManager),
        dist_group=object(),
        prefill_att_state1=SimpleNamespace(prefill_att=attention),
        decode_att_state1=SimpleNamespace(decode_att=attention),
    )
    forward = layer.context_attention_forward if prefill else layer.token_attention_forward
    output = forward(rank_input, state, object())
    torch.testing.assert_close(output, (rank_input + 1) * 2)
    assert calls == (
        (["gather"] if mix_mode else [])
        + ["project", "captured_attention" if capture else "attention"]
        + ["reduce_scatter" if mix_mode else "all_reduce"]
    )
