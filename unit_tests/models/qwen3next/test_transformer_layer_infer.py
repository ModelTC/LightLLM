from types import SimpleNamespace

import torch

from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextTransformerLayerInfer,
)


def test_linear_post_passes_strided_gate_without_materializing():
    tokens, heads, head_dim = 3, 4, 8
    packed_dim = heads * head_dim + 16
    packed = torch.randn((tokens, packed_dim))
    z = packed[:, 8 : 8 + heads * head_dim].view(tokens, heads, head_dim)
    core_attn_out = torch.randn((1, tokens, heads, head_dim))
    calls = []

    def linear_norm(input, gate_value, eps):
        calls.append((input, gate_value, eps))
        return input

    layer_weight = SimpleNamespace(
        linear_norm=linear_norm,
        linear_out_proj=SimpleNamespace(mm=lambda input: input),
    )
    layer_infer = object.__new__(Qwen3NextTransformerLayerInfer)
    layer_infer.eps_ = 1e-6

    output = layer_infer._linear_post(core_attn_out, z, layer_weight)

    assert len(calls) == 1
    assert calls[0][0].shape == (tokens * heads, head_dim)
    assert calls[0][1] is z
    assert not calls[0][1].is_contiguous()
    assert output.shape == (tokens, heads * head_dim)
