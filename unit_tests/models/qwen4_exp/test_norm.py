import pytest
import torch

from lightllm.models.qwen4_exp.layer_weights.norm_weight import Qwen4ExpGatedNormWeight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_native_gdn_norm_rounding(dtype, activation):
    torch.manual_seed(837)
    native = Qwen4ExpGatedNormWeight(128, "linear.norm.weight", dtype, activation)
    weight = 1 + torch.randn(128, dtype=dtype) * 0.1
    native.load_hf_weights({"linear.norm.weight": weight})
    values = torch.randn(19, 128, device="cuda", dtype=dtype)
    gate = torch.randn_like(values) * 3
    normalized = values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
    weighted = native.weight * normalized.to(dtype)
    gate_activation = gate.float().sigmoid()
    if activation == "silu":
        gate_activation *= gate.float()
    expected = (weighted * gate_activation).to(dtype)
    out = torch.empty_like(values)
    actual = native(values, gate, 1e-6, out=out)
    assert actual is out
    tolerance = 2e-6 if dtype == torch.float32 else torch.finfo(dtype).eps * 2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_gdn_norm_accepts_strided_gate(activation):
    from lightllm.models.qwen4_exp.triton_kernel.gated_norm import gated_norm

    torch.manual_seed(123)
    tokens, heads, dim = 5, 4, 128
    values = torch.randn(tokens * heads, dim, device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(tokens, heads * dim + 256, device="cuda", dtype=torch.bfloat16)
    gate = packed[:, 128 : 128 + heads * dim].view(tokens, heads, dim)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    actual = gated_norm(values, gate, weight, 1e-6, activation)
    normalized = values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = normalized * weight.float()
    gate = gate.reshape(-1, dim).float()
    gate = gate.sigmoid() if activation == "sigmoid" else torch.nn.functional.silu(gate)
    expected = (expected * gate).to(values.dtype)
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
