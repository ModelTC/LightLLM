"""Native recurrent attention against the original CPU reference."""

from unittest.mock import patch

import pytest
import torch

from lightllm.backends.metal.kernels import fused_recurrent_gated_delta_rule


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires MPS and torch.mps.compile_shader",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("lengths", [(128,), (0, 17, 5), (256,), (0, 0, 0)])
def test_prefill_reads_device_boundaries_and_preserves_initial_state(dtype, normalize, lengths):
    generator = torch.Generator().manual_seed(93501)
    tokens, batch, q_heads, v_heads = sum(lengths), len(lengths), 8, 16
    q = torch.randn(1, tokens, q_heads, 128, dtype=dtype, generator=generator) * 0.1
    k = torch.randn(q.shape, dtype=dtype, generator=generator) * 0.1
    v = torch.randn(1, tokens, v_heads, 128, dtype=dtype, generator=generator) * 0.1
    g = -torch.rand(1, tokens, v_heads, generator=generator) * 0.1
    beta = torch.rand(1, tokens, v_heads, generator=generator)
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    state_storage = torch.randn(2 * batch + 1, v_heads, 128, 128, generator=generator) * 0.01
    state = state_storage[1::2]
    kwargs = dict(
        g=g,
        beta=beta,
        initial_state=state,
        inplace_final_state=False,
        cu_seqlens=boundaries,
        use_qk_l2norm_in_kernel=normalize,
    )
    expected, expected_state = fused_recurrent_gated_delta_rule(q, k, v, **kwargs)

    def device_strided(host):
        storage = torch.empty(*host.shape[:-1], host.shape[-1] * 2 + 1, dtype=host.dtype, device="mps")
        view = storage[..., 1::2]
        view.copy_(host)
        return view

    device_q, device_k, device_v = (device_strided(t) for t in (q, k, v))
    device_state = state_storage.to("mps")
    gpu_kwargs = {name: value.to("mps") if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}
    gpu_kwargs["initial_state"] = device_state[1::2]
    out = device_strided(torch.zeros_like(v))
    with torch.inference_mode(), patch.object(torch.Tensor, "tolist", side_effect=AssertionError("CPU metadata list")):
        actual, actual_state = fused_recurrent_gated_delta_rule(device_q, device_k, device_v, out=out, **gpu_kwargs)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, atol=0.002 if dtype == torch.float16 else 1e-5, rtol=0.003)
    torch.testing.assert_close(actual_state.cpu(), expected_state, atol=3e-4, rtol=0.003)
    torch.testing.assert_close(device_state.cpu(), state_storage, atol=0, rtol=0)


@pytest.mark.parametrize("alias", [False, True])
def test_indexed_decode_still_matches_cpu(alias):
    generator = torch.Generator().manual_seed(93502)
    batch, q_heads, v_heads = 4, 8, 16
    q = torch.randn(batch, 1, q_heads, 128, dtype=torch.float16, generator=generator)
    k = torch.randn(q.shape, dtype=q.dtype, generator=generator)
    v = torch.randn(batch, 1, v_heads, 128, dtype=q.dtype, generator=generator)
    state = torch.randn(6, v_heads, 128, 128, generator=generator) * 0.01
    kwargs = dict(
        g=-torch.rand(batch, 1, v_heads, generator=generator) * 0.1,
        beta=torch.rand(batch, 1, v_heads, generator=generator),
        inplace_final_state=True,
        ssm_state_indices=torch.tensor([2, 2, 1, 2] if alias else [4, 1, 0, 2], dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
    )
    expected, expected_state = fused_recurrent_gated_delta_rule(q, k, v, initial_state=state.clone(), **kwargs)
    gpu_kwargs = {name: value.to("mps") if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}
    actual, actual_state = fused_recurrent_gated_delta_rule(
        q.to("mps"), k.to("mps"), v.to("mps"), initial_state=state.to("mps"), **gpu_kwargs
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.003)
    torch.testing.assert_close(actual_state.cpu(), expected_state, atol=3e-4, rtol=0.003)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_single_token_decode_in_varlen_prefill_preserves_recurrent_state(dtype):
    """The shared mixed scheduler can send decode rows through prefill GDN."""
    generator = torch.Generator().manual_seed(93503)
    lengths = [13, 1, 7, 1]
    tokens, heads = sum(lengths), 16
    q = torch.randn(1, tokens, 4, 128, generator=generator, dtype=dtype) * 0.1
    k = torch.randn(q.shape, generator=generator, dtype=dtype) * 0.1
    v = torch.randn(1, tokens, heads, 128, generator=generator, dtype=dtype) * 0.1
    g = -torch.rand(1, tokens, heads, generator=generator) * 0.2
    beta = torch.rand(1, tokens, heads, generator=generator)
    initial = torch.randn(len(lengths), heads, 128, 128, generator=generator) * 0.1
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    tensors = [x.to("mps") for x in (q, k, v, g, beta, initial)]
    dq, dk, dv, dg, db, state = tensors
    with torch.inference_mode():
        mixed, final = fused_recurrent_gated_delta_rule(
            dq,
            dk,
            dv,
            dg,
            db,
            initial_state=state,
            inplace_final_state=False,
            cu_seqlens=boundaries.to("mps"),
            use_qk_l2norm_in_kernel=True,
        )
        for row, start in [(1, 13), (3, 21)]:
            decoded, decoded_state = fused_recurrent_gated_delta_rule(
                dq[:, start : start + 1],
                dk[:, start : start + 1],
                dv[:, start : start + 1],
                dg[:, start : start + 1],
                db[:, start : start + 1],
                initial_state=state.clone(),
                inplace_final_state=True,
                ssm_state_indices=torch.tensor([row], dtype=torch.int32, device="mps"),
                use_qk_l2norm_in_kernel=True,
            )
            torch.testing.assert_close(mixed[:, start : start + 1].cpu(), decoded.cpu(), atol=2e-4, rtol=2e-3)
            torch.testing.assert_close(final[row].cpu(), decoded_state[row].cpu(), atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(state.cpu(), initial, atol=0, rtol=0)
