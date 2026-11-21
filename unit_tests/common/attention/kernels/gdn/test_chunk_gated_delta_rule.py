import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention.kernels.gdn.third_party_fla.ops.chunk import chunk_gated_delta_rule
from lightllm.common.attention.kernels.gdn.third_party_fla.ops.kda import chunk_kda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("key_dim,value_dim", [(64, 64), (128, 80)])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("channel_gate", [False, True])
def test_varlen_prefill_matches_recurrent_reference(key_dim, value_dim, normalize, channel_gate):
    torch.manual_seed(7)
    lengths = [1, 17, 65]
    total, heads = sum(lengths), 2
    q = F.normalize(torch.randn(1, total, heads, key_dim, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
    v = torch.randn(1, total, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    gate_shape = (1, total, heads, key_dim) if channel_gate else (1, total, heads)
    g = -torch.rand(gate_shape, device="cuda") * 0.1
    beta = torch.rand(1, total, heads, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(len(lengths), heads, key_dim, value_dim, device="cuda") * 0.1
    boundaries = [0, 1, 18, total]
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int64, device="cuda")

    operation = chunk_kda if channel_gate else chunk_gated_delta_rule
    actual, final = operation(
        q,
        k,
        v.clone(),
        g,
        beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=normalize,
    )
    q_ref = F.normalize(q.float(), dim=-1).bfloat16().float() if normalize else q.float()
    k_ref = F.normalize(k.float(), dim=-1).bfloat16().float() if normalize else k.float()
    expected = torch.empty_like(actual, dtype=torch.float32)
    states = []
    for sequence, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = initial[sequence].clone()
        for token in range(start, end):
            key = k_ref[0, token]
            decay = g[0, token].exp() if channel_gate else g[0, token].exp()[:, None]
            state = state * decay[:, :, None]
            residual = v[0, token].float() - torch.einsum("hk,hkv->hv", key, state)
            residual *= beta[0, token].float()[:, None]
            state += key[:, :, None] * residual[:, None, :]
            expected[0, token] = torch.einsum("hk,hkv->hv", q_ref[0, token], state) * key_dim ** -0.5
        states.append(state)
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(final, torch.stack(states), atol=8e-3, rtol=2e-2)
