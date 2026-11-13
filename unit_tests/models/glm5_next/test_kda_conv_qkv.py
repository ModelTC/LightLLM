from unit_tests.common.attention.helpers import run_attention
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase
from lightllm.common.attention.api.inputs import KDAInputs, KDAResources
from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn
from lightllm.common.attention.backends._shared.linear.causal_conv1d_qkv import causal_conv1d_qkv
from lightllm.common.attention.backends._shared.linear.l2norm import USE_DEFAULT_FLA_NORM, l2norm_fwd
from lightllm.common.attention.backends.kda.triton import backend as backend_module
from lightllm.common.attention.backends.kda.triton.kernels.kda import chunk_kda_with_fused_gate
from unit_tests.common.attention.helpers import make_backend, make_batch, make_context, make_layer_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0) or USE_DEFAULT_FLA_NORM,
    reason="requires Hopper and the LightLLM L2 norm",
)


def reference(x, weight, states, starts, slots, warm, heads):
    mixed = causal_conv1d_fn(
        x.clone().T,
        weight,
        query_start_loc=starts,
        cache_indices=slots,
        has_initial_state=warm,
        conv_states=states,
    ).T
    q, k, v = [part.view(1, x.shape[0], heads, 128) for part in mixed.chunk(3, dim=-1)]
    return l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v.contiguous()


@pytest.mark.parametrize("heads", [1, 2, 8, 64])
@pytest.mark.parametrize("state_width", [3, 4, 7])
@pytest.mark.parametrize("lengths", [(2049, 2, 1024), (257, 1, 1), (1, 2, 3), (0, 1, 0, 2), (1,)])
def test_conv_norm_preserves_outputs_and_history(heads, state_width, lengths):
    torch.manual_seed(42)
    channels = heads * 128 * 3
    opts = dict(device="cuda", dtype=torch.bfloat16)
    projection = torch.randn(sum(lengths), channels + heads + 256, **opts)
    original_projection = projection.clone()
    x = projection[:, :channels]
    weight = torch.randn(channels, 4, **opts) * 0.125
    states = torch.randn(5, channels, state_width, **opts)
    expected_states = states.clone()
    starts = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    slots = torch.tensor([1, 2, -1, 3][: len(lengths)], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([0, 7, 0, 1][: len(lengths)], device="cuda", dtype=torch.int32)
    expected = reference(x, weight, expected_states[..., :3], starts, slots, cache_seqlens > 0, heads)
    actual = causal_conv1d_qkv(x, weight, states[..., :3], starts, slots, cache_seqlens, max(lengths))
    for a, b in zip(actual, expected):
        assert a.is_contiguous()
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert len({a.untyped_storage().data_ptr() for a in actual}) == 3
    assert torch.equal(states, expected_states)
    assert torch.equal(projection, original_projection)


@pytest.mark.parametrize("lengths", [(3, 2, 4), (7, 1, 1)])
def test_conv_norm_cuda_graph_changing_boundaries(lengths):
    torch.manual_seed(42)
    heads, channels = 2, 3 * 2 * 128
    opts = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(9, channels + 258, **opts)[:, :channels]
    weight = torch.randn(channels, 4, **opts) * 0.125
    initial = torch.randn(5, channels, 4, **opts)
    states = initial.clone()
    initial_boundaries = [0, lengths[0], lengths[0] + lengths[1], sum(lengths)]
    starts = torch.tensor(initial_boundaries, device="cuda", dtype=torch.int32)
    slots = torch.tensor([1, 2, -1], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([4, 0, 0], device="cuda", dtype=torch.int32)
    causal_conv1d_qkv(x, weight, states[..., :3], starts, slots, cache_seqlens, max(lengths))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = causal_conv1d_qkv(x, weight, states[..., :3], starts, slots, cache_seqlens, max(lengths))
    for boundaries, cached in ((initial_boundaries, [4, 0, 0]), ([0, 4, 8, 9], [0, 7, 0])):
        starts.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
        cache_seqlens.copy_(torch.tensor(cached, device="cuda", dtype=torch.int32))
        states.copy_(initial)
        expected_states = initial.clone()
        expected = reference(x, weight, expected_states[..., :3], starts, slots, cache_seqlens > 0, heads)
        graph.replay()
        for a, b in zip(actual, expected):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        assert torch.equal(states, expected_states)


@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.parametrize("state_width", [3, 7])
def test_conv_norm_chained_prefills(heads, state_width):
    torch.manual_seed(42)
    channels = 3 * heads * 128
    opts = dict(device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, **opts) * 0.125
    states = torch.randn(5, channels, state_width, **opts)
    expected_states = states.clone()
    slots = torch.tensor([1, 2, -1, 3], device="cuda", dtype=torch.int32)
    cached = torch.tensor([0, 7, 0, 1], device="cuda", dtype=torch.int32)
    for lengths in [(1, 2, 0, 3), (0, 1, 2, 0), (7, 0, 1, 2), (2, 1, 0, 1)]:
        x = torch.randn(sum(lengths), channels + heads + 256, **opts)[:, :channels]
        starts = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        expected = reference(x, weight, expected_states[..., :3], starts, slots, cached > 0, heads)
        actual = causal_conv1d_qkv(x, weight, states[..., :3], starts, slots, cached, max(lengths))
        for a, b in zip(actual, expected):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        assert torch.equal(states, expected_states)
        cached += torch.tensor(lengths, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("lengths", [(129, 2), (129, 1, 1), (1,)])
def test_backend_fused_conv_output_and_state(lengths):
    torch.manual_seed(42)
    heads, channels, tokens = 2, 3 * 2 * 128, sum(lengths)
    opts = dict(device="cuda", dtype=torch.bfloat16)
    projection = torch.randn(tokens, channels + heads + 256, **opts)
    weight = torch.randn(channels, 4, **opts) * 0.125
    conv_states = torch.randn(4, channels, 4, **opts)
    ssm_states = torch.randn(4, heads, 128, 128, device="cuda") * 0.01
    linear_state = SimpleNamespace(get_layer_view=lambda **kwargs: (conv_states, ssm_states))
    infer_state = type("State", (), {})()
    infer_state.att_state_manager = SimpleNamespace(get_state=lambda **kwargs: linear_state)
    infer_state.att_state_role = "main"
    infer_state.b1_cu_q_seq_len = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    infer_state.b_ready_cache_len = torch.arange(len(lengths), device="cuda", dtype=torch.int32) % 2 * 7
    infer_state.max_q_seq_len = max(lengths)
    backend = make_backend(
        backend_module.TritonKDADecodeBackend,
        head_dim=128,
        tp_num_heads=heads,
        tp_hidden_size=heads * 128,
        conv_kernel_size=4,
        lower_bound=-5.0,
    )
    batch = make_batch(
        lengths,
        phase=AttentionPhase.PREFILL,
        device="cuda",
        token_num=tokens,
        max_q_seq_len=max(lengths),
        b1_cu_q_seq_len=infer_state.b1_cu_q_seq_len,
        b_ready_cache_len=infer_state.b_ready_cache_len,
    )
    att = make_context(backend, batch)
    att.b_conv_buffer_idx = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
    att.b_ssm_buffer_idx = att.b_conv_buffer_idx
    layer_weight = SimpleNamespace(
        get_merged_kda_conv_weight=lambda: weight,
        linear_A_log=SimpleNamespace(weight=torch.zeros(heads, device="cuda")),
        linear_dt_bias=SimpleNamespace(weight=torch.zeros(heads * 128, device="cuda")),
    )
    inputs = KDAInputs(
        mixed_qkv=projection[:, :channels],
        raw_gate=torch.randn(tokens, heads * 128, **opts) * 0.1,
        raw_beta=torch.randn(tokens, heads, **opts),
    )
    attention = make_layer_attention(
        backend,
        att,
        KDAResources(
            conv_states, ssm_states, weight, layer_weight.linear_A_log.weight, layer_weight.linear_dt_bias.weight
        ),
    )
    expected_conv, expected_ssm = conv_states.clone(), ssm_states.clone()
    q, k, v = reference(
        projection[:, :channels],
        weight,
        expected_conv[..., :3],
        infer_state.b1_cu_q_seq_len,
        att.b_conv_buffer_idx,
        infer_state.b_ready_cache_len > 0,
        heads,
    )
    expected, final_state = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v,
        raw_g=inputs.raw_gate.view(1, tokens, heads, 128),
        beta=inputs.raw_beta.float().sigmoid().view(1, tokens, heads),
        A_log=layer_weight.linear_A_log.weight,
        g_bias=layer_weight.linear_dt_bias.weight,
        initial_state=ssm_states[att.b_ssm_buffer_idx].contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=infer_state.b1_cu_q_seq_len,
        safe_gate=True,
        lower_bound=-5.0,
    )
    expected_ssm[att.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype)
    actual = run_attention(attention, att, inputs)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(conv_states, expected_conv)
    assert torch.equal(ssm_states, expected_ssm)
