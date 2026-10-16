"""Preserve gate rounding and exact counter updates in fused Metal operations."""

import pytest
import torch

from lightllm.backends.metal import kernels
from lightllm.backends.metal.elementwise import sigmoid_product, silu_product


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires MPS and torch.mps.compile_shader",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["blocked", "interleaved"])
@pytest.mark.parametrize("batch,width", [(1, 3584), (16, 257), (128, 3584)])
def test_silu_product_preserves_intermediate_rounding(dtype, layout, batch, width):
    generator = torch.Generator().manual_seed(61003)
    host = torch.randn(batch + 1, 2 * width + 5, dtype=dtype, generator=generator)
    view = host[1:, 1 : 2 * width + 1]
    expected = torch.empty(batch, width, dtype=dtype)
    kernels.silu_and_mul_fwd(view, expected, layout=layout)
    gpu = host.to("mps")
    output_storage = torch.empty(batch + 1, width, dtype=dtype, device="mps")
    output = output_storage[1:]
    kernels.silu_and_mul_fwd(gpu[1:, 1 : 2 * width + 1], output, layout=layout)
    torch.testing.assert_close(output.cpu(), expected, atol=0.002 if dtype == torch.float16 else 1e-6, rtol=0.003)
    torch.testing.assert_close(gpu.cpu(), host, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("gate_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("broadcast", [False, True])
def test_sigmoid_product_preserves_gate_dtype_and_other_storage(dtype, gate_dtype, broadcast):
    generator = torch.Generator().manual_seed(61004)
    host = torch.randn(17, 515, dtype=dtype, generator=generator)
    gate_storage = torch.randn(17, 3 if broadcast else 515, dtype=gate_dtype, generator=generator)
    expected = host.clone()
    gate = gate_storage[:, 1::2]
    kernels.sigmoid_mul_(expected[:, 1::2], gate)
    gpu, gpu_gate = host.to("mps"), gate_storage.to("mps")
    kernels.sigmoid_mul_(gpu[:, 1::2], gpu_gate[:, 1::2])
    torch.testing.assert_close(gpu.cpu(), expected, atol=0.002 if dtype == torch.float16 else 1e-6, rtol=0.003)
    torch.testing.assert_close(gpu_gate.cpu(), gate_storage, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("use_mask", [False, True])
def test_counter_duplicate_slots_masks_invalid_tokens_and_strides(dtype, use_mask):
    generator = torch.Generator().manual_seed(61005)
    requests = torch.tensor([2, 2, 2, -1, 1, -2, 2, 2, 2], dtype=dtype)
    tokens = torch.tensor([1, 1, 1, 256, -1, 8, 256, 999, 1], dtype=torch.int64)
    mask = torch.tensor([True, False, True, True, True, False, True, True, True]) if use_mask else None
    host = torch.randint(0, 100, (7, 515), dtype=torch.int32, generator=generator)
    expected = host.clone()
    kernels.update_req_to_token_id_counter(requests, tokens, expected[1:, 1::2], mask)
    gpu = host.to("mps")

    def strided_device(value):
        storage = torch.empty(2 * value.numel() + 1, dtype=value.dtype, device="mps")
        storage[1::2].copy_(value)
        return storage[1::2]

    kernels.update_req_to_token_id_counter(
        strided_device(requests),
        strided_device(tokens),
        gpu[1:, 1::2],
        strided_device(mask) if mask is not None else None,
    )
    torch.testing.assert_close(gpu.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("tanh_gelu", [False, True])
def test_nonstandard_activation_retains_fallback(monkeypatch, tanh_gelu):
    monkeypatch.setattr(kernels, "ffn_use_tanh_approximate_gelu", lambda: tanh_gelu)
    x = torch.randn(4, 512, dtype=torch.float16)
    args = {} if tanh_gelu else dict(limit=2.0, alpha=1.7)
    expected, actual = torch.empty(4, 256, dtype=x.dtype), torch.empty(4, 256, dtype=x.dtype, device="mps")
    kernels.silu_and_mul_fwd(x, expected, **args)
    kernels.silu_and_mul_fwd(x.to("mps"), actual, **args)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.002, rtol=0.003)


def test_aliased_products_retain_existing_temporary_storage():
    host = torch.randn(1, 512, dtype=torch.float16)
    gpu = host.to("mps")
    assert not silu_product(gpu, gpu[:, :256], "blocked")
    kernels.silu_and_mul_fwd(host, host[:, :256])
    kernels.silu_and_mul_fwd(gpu, gpu[:, :256])
    torch.testing.assert_close(gpu.cpu(), host, atol=0.002, rtol=0.003)
    assert not sigmoid_product(gpu, gpu)
    kernels.sigmoid_mul_(host, host)
    kernels.sigmoid_mul_(gpu, gpu)
    torch.testing.assert_close(gpu.cpu(), host, atol=0.002, rtol=0.003)
