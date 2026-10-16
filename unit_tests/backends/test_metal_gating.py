"""FP32 prefill gates, original beta rounding, strided storage and fallback."""
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from lightllm.backends.metal.gating import gating
from lightllm.backends.metal.kernels import fused_gdn_gating

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")


def oracle(A, a, b, dt, beta=1.0, threshold=20.0):
    return (
        -A.float().exp() * F.softplus(a.float() + dt.float(), beta=beta, threshold=threshold),
        b.float().sigmoid().to(b.dtype).float(),
    )


@pytest.mark.parametrize("rows", [0, 1, 112, 256])
@pytest.mark.parametrize("heads", [1, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("weight_type", [torch.float16, torch.float32])
def test_gating_strided_inputs_match_cpu_and_original_gpu(rows, heads, dtype, weight_type):
    gen = torch.Generator().manual_seed(91212)
    packed = torch.randn(rows, heads * 4 + 7, dtype=dtype, generator=gen) * 3
    weights = torch.randn(2, heads + 7, dtype=weight_type, generator=gen) * 0.2
    a, b = packed[:, 1 : 2 * heads + 1 : 2], packed[:, 2 * heads + 1 : 4 * heads + 1 : 2]
    A, dt = weights[0, 3 : heads + 3], weights[1, 3 : heads + 3]
    device_p, device_w = packed.to("mps"), weights.to("mps")
    da, db = device_p[:, 1 : 2 * heads + 1 : 2], device_p[:, 2 * heads + 1 : 4 * heads + 1 : 2]
    dA, ddt = device_w[0, 3 : heads + 3], device_w[1, 3 : heads + 3]
    with torch.inference_mode():
        actual = gating(dA, da, db, ddt)
        gpu = oracle(dA, da, db, ddt)
    assert actual is not None
    for x, expected in zip(actual, oracle(A, a, b, dt)):
        torch.testing.assert_close(x.cpu(), expected, atol=2e-6, rtol=0.003 if dtype == torch.float16 else 3e-6)
    torch.testing.assert_close(actual[0], gpu[0], atol=2e-6, rtol=3e-6)
    torch.testing.assert_close(actual[1], gpu[1], atol=0, rtol=0)
    torch.testing.assert_close(device_p.cpu(), packed, atol=0, rtol=0)
    torch.testing.assert_close(device_w.cpu(), weights, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gating_softplus_boundaries_and_extremes(dtype):
    a = torch.linspace(-100, 100, 4096, dtype=torch.float32).reshape(-1, 16).to(dtype)
    a[0, :8] = torch.tensor([-100, -20, -2.7725887, -2.772, 0, 19.999, 20, 100], dtype=dtype)
    b = -a
    A = torch.linspace(-20, 20, 16)
    dt = torch.zeros(16)
    with torch.inference_mode():
        actual = gating(*(t.to("mps") for t in (A, a, b, dt)))
    for x, expected in zip(actual, oracle(A, a, b, dt)):
        torch.testing.assert_close(x.cpu(), expected, atol=2e-6, rtol=0.003 if dtype == torch.float16 else 3e-6)


def test_gating_preserves_broad_torch_fallback():
    A = torch.zeros(16, device="mps")
    a = torch.randn(4, 16, device="mps", dtype=torch.float16)
    before = a.cpu()
    assert gating(A.cpu(), a, a, A) is None
    assert gating(A, a.cpu(), a.cpu(), A) is None
    assert gating(A, a, a, A, beta=2) is None
    assert gating(A, a, a, A, threshold=5) is None
    assert gating(A, a.unsqueeze(0), a.unsqueeze(0), A) is None
    assert gating(A, a.requires_grad_(), a, A) is None
    a.requires_grad_(False)
    assert gating(A.repeat(2)[::2], a, a, A) is None
    with torch.inference_mode(), patch("lightllm.backends.metal.gating.gating", return_value=None) as native:
        actual = fused_gdn_gating(A, a, a, A, beta=2, threshold=5)
        assert native.call_count == 1
    for x, expected in zip(actual, oracle(A, a, a, A, beta=2, threshold=5)):
        torch.testing.assert_close(x, expected, atol=0, rtol=0)
    torch.testing.assert_close(a.cpu(), before, atol=0, rtol=0)


def test_gating_selected_through_existing_operator_interface():
    A = torch.zeros(16, device="mps")
    a = torch.randn(112, 16, device="mps", dtype=torch.float16)
    with torch.inference_mode(), patch("lightllm.backends.metal.gating.gating", wraps=gating) as native:
        actual = fused_gdn_gating(A, a, a, A)
        assert native.call_count == 1
    for x, expected in zip(actual, oracle(A, a, a, A)):
        torch.testing.assert_close(x, expected, atol=2e-6, rtol=0.003)
