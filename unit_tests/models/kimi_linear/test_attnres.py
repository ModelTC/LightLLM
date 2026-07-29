from types import SimpleNamespace

import pytest
import torch

from lightllm.models.kimi_linear.attnres import (
    BlockAttnResConfig,
    BlockAttnResState,
    block_attnres_mix,
    normalize_attnres_query_weight,
)
from lightllm.models.kimi_linear.layer_infer.transformer_layer_infer import (
    KimiLinearTransformerLayerInfer,
)


def _stack_reference(sources, query, norm_weight, eps):
    values = torch.stack(sources)
    keys = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    keys = keys * norm_weight
    logits = torch.einsum("d,std->st", query, keys)
    return torch.einsum("st,std->td", logits.softmax(dim=0), values)


def test_block_attnres_matches_kimi_k3_reference():
    torch.manual_seed(7)
    sources = [torch.randn(5, 8) for _ in range(4)]
    query = torch.randn(8)
    norm_weight = torch.randn(8)

    actual = block_attnres_mix(sources, query, norm_weight, eps=1e-6)
    expected = _stack_reference(sources, query, norm_weight, eps=1e-6)

    torch.testing.assert_close(actual, expected)


def test_zero_query_is_uniform_average():
    sources = [torch.full((2, 4), value) for value in (1.0, 3.0, 8.0)]
    actual = block_attnres_mix(sources, torch.zeros(4), torch.ones(4), eps=1e-6)
    expected = sum(sources) / len(sources)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(8,), (1, 8), (8, 1)])
def test_attnres_projection_accepts_checkpoint_vector_shapes(shape):
    weight = torch.arange(8).reshape(shape)
    normalized = normalize_attnres_query_weight(weight, hidden_size=8)

    assert normalized.shape == (8,)
    torch.testing.assert_close(normalized, torch.arange(8))


def test_attnres_config_uses_released_kimi_k3_field():
    config = BlockAttnResConfig.from_network_config({"attn_res_block_size": 12})

    assert config.block_size == 12
    assert BlockAttnResConfig.from_network_config({}) is None


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_attnres_config_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="attn_res_block_size"):
        BlockAttnResConfig.from_network_config({"attn_res_block_size": block_size})


def test_state_starts_a_new_residual_block_at_layer_boundary():
    embedding = torch.full((1, 3), 10.0)
    state = BlockAttnResState.from_embedding(embedding, block_size=2)

    state.begin_layer(0)
    assert state.prefix_sum is None
    torch.testing.assert_close(state.block_residuals[0], embedding)

    state.add_sublayer_output(torch.full_like(embedding, 1.0))
    state.add_sublayer_output(torch.full_like(embedding, 2.0))
    torch.testing.assert_close(state.prefix_sum, torch.full_like(embedding, 3.0))

    state.begin_layer(1)
    assert len(state.block_residuals) == 1
    state.begin_layer(2)
    assert state.prefix_sum is None
    torch.testing.assert_close(state.block_residuals[1], torch.full_like(embedding, 3.0))


def _make_attnres_layer(layer_num, is_last_layer, attn_output, mlp_output, config, captures, calls):
    layer = object.__new__(KimiLinearTransformerLayerInfer)
    layer.layer_num_ = layer_num
    layer.embed_dim_ = 1
    layer.eps_ = 1e-6
    layer.attnres_config = config
    layer.is_last_layer = is_last_layer
    layer._att_norm = lambda input, infer_state, layer_weight: captures.append(input.clone()) or input
    layer._ffn_norm = lambda input, infer_state, layer_weight: captures.append(input.clone()) or input
    layer.context_attention_forward = lambda input, infer_state, layer_weight: (
        calls.append("context") or torch.full_like(input, attn_output)
    )
    layer.token_attention_forward = lambda input, infer_state, layer_weight: (
        calls.append("token") or torch.full_like(input, attn_output)
    )
    layer._ffn = lambda input, infer_state, layer_weight: torch.full_like(input, mlp_output)
    return layer


def _make_attnres_weight(include_final=False):
    vector = SimpleNamespace(weight=torch.ones(1))
    query = SimpleNamespace(weight=torch.zeros(1))
    weight = SimpleNamespace(
        attnres_attn_query=query,
        attnres_attn_norm=vector,
        attnres_mlp_query=query,
        attnres_mlp_norm=vector,
    )
    if include_final:
        weight.attnres_final_query = query
        weight.attnres_final_norm = vector
    return weight


@pytest.mark.parametrize(
    ("forward_name", "expected_call"),
    [("context_forward", "context"), ("token_forward", "token")],
)
def test_layer_forward_matches_released_attnres_block_semantics(forward_name, expected_call):
    config = BlockAttnResConfig.from_network_config({"attn_res_block_size": 2})
    captures = []
    calls = []
    layer0 = _make_attnres_layer(0, False, 1.0, 2.0, config, captures, calls)
    layer1 = _make_attnres_layer(1, True, 4.0, 5.0, config, captures, calls)
    infer_state = SimpleNamespace(attnres_state=None)

    hidden = getattr(layer0, forward_name)(torch.tensor([[10.0]]), infer_state, _make_attnres_weight())
    output = getattr(layer1, forward_name)(hidden, infer_state, _make_attnres_weight(include_final=True))

    expected_inputs = [10.0, 5.5, 6.5, 8.5]
    for actual, expected in zip(captures, expected_inputs):
        torch.testing.assert_close(actual, torch.tensor([[expected]]))
    torch.testing.assert_close(output, torch.tensor([[11.0]]))
    assert calls == [expected_call, expected_call]
    assert infer_state.attnres_state is None
