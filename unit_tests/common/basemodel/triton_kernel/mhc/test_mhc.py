# SPDX-License-Identifier: Apache-2.0

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import triton

from lightllm.common.basemodel.triton_kernel.mhc import hc_post, hc_pre_norm
from lightllm.common.basemodel.triton_kernel.mhc import pre_norm
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _hc_pre_reference(
    x,
    fn,
    scale,
    base,
    streams,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_multiplier=2.0,
    *,
    normalize_before_projection=False,
):
    """Unfused FP32 reference; returns the merged input before sublayer RMSNorm."""
    assert x.ndim == 2 and x.shape[-1] % streams == 0
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    residual = x.view(tokens, streams, hidden)

    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + rms_eps)
    if normalize_before_projection:
        mixes = F.linear(x_fp32 * inv_rms, fn)
    else:
        mixes = F.linear(x_fp32, fn) * inv_rms

    pre_raw = mixes[:, :streams]
    post_raw = mixes[:, streams : 2 * streams]
    residual_raw = mixes[:, 2 * streams :].view(tokens, streams, streams)

    pre = torch.sigmoid(pre_raw * scale[0] + base[:streams]) + hc_eps
    post = post_multiplier * torch.sigmoid(post_raw * scale[1] + base[streams : 2 * streams])
    residual_mix = (residual_raw * scale[2] + base[2 * streams :].view(streams, streams)).softmax(dim=-1)
    residual_mix = residual_mix + hc_eps
    residual_mix = residual_mix / (residual_mix.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        residual_mix = residual_mix / (residual_mix.sum(dim=-1, keepdim=True) + hc_eps)
        residual_mix = residual_mix / (residual_mix.sum(dim=-2, keepdim=True) + hc_eps)

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(x.dtype)
    return layer_input, residual_mix, post


def _hc_post_reference(layer_output, residual, residual_mix, post_mix, streams, *, round_residual=False):
    """Unfused FP32 reference; residual_mix maps input streams to output streams."""
    tokens, hidden = layer_output.shape
    residual_3d = residual.view(tokens, streams, hidden)
    mixed_residual = (residual_mix.unsqueeze(-1) * residual_3d.float().unsqueeze(2)).sum(dim=1)
    if round_residual:
        mixed_residual = mixed_residual.to(layer_output.dtype).float()
    expanded = post_mix.unsqueeze(-1) * layer_output.float().unsqueeze(1)
    if round_residual:
        expanded = expanded.to(layer_output.dtype).float()
    out = expanded + mixed_residual
    return out.to(layer_output.dtype).reshape(tokens, streams * hidden)


@pytest.fixture(autouse=True)
def setup():
    torch.manual_seed(1525)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))


@pytest.mark.parametrize("tokens", [3, 19, 3073])
@pytest.mark.parametrize("disable_deepgemm", [False, True], ids=["deepgemm", "torch"])
def test_mhc_matches_reference(tokens, disable_deepgemm, monkeypatch):
    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", disable_deepgemm)
    if disable_deepgemm:
        monkeypatch.setitem(sys.modules, "deep_gemm", None)
    streams, hidden = 4, 4096
    x = torch.randn(tokens, streams * hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, streams * hidden, device="cuda") * 0.005
    scale = torch.randn(3, device="cuda")
    base = torch.randn(24, device="cuda")
    norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    expected = _hc_pre_reference(x, fn, scale, base, streams, 1e-5, 1e-6, 20)
    actual = hc_pre_norm(x, fn, scale, base, norm, streams, 1e-5, 1e-5, 1e-6, 20)
    torch.testing.assert_close(actual[0], rmsnorm_forward(expected[0], norm, 1e-5), atol=0.04, rtol=0.03)
    for a, b in zip(actual[1:], expected[1:]):
        # DeepGEMM's mHC projection uses TF32 inputs and FP32 accumulation.
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)
    layer_out = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        hc_post(layer_out, x, *actual[1:], streams),
        _hc_post_reference(layer_out, x, *actual[1:], streams),
        atol=0.04,
        rtol=0.02,
    )


@pytest.mark.parametrize("missing", ["module", "kernel"])
def test_mhc_reports_missing_deepgemm(monkeypatch, missing):
    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", False)
    monkeypatch.setitem(sys.modules, "deep_gemm", None if missing == "module" else ModuleType("deep_gemm"))
    x = torch.empty(1, 16, dtype=torch.bfloat16)
    fn = torch.empty(24, 16)
    scale = torch.empty(3)
    base = torch.empty(24)
    norm = torch.empty(4, dtype=torch.bfloat16)

    with pytest.raises(ImportError, match="deep_gemm"):
        hc_pre_norm(x, fn, scale, base, norm, 4, 1e-5, 1e-5, 1e-6, 20)


@pytest.mark.parametrize("tokens,hidden,prefix", [(1, 4096, "attn"), (19, 780, "ffn"), (3073, 4096, "attn")])
@pytest.mark.parametrize("seed", [1525, 2026])
def test_glm_mhc_caller_matches_normalize_first_reference(tokens, hidden, prefix, seed, monkeypatch):
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer

    torch.manual_seed(seed)
    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    streams = 4
    x = torch.randn(tokens, streams * hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, streams * hidden, device="cuda") * 0.005
    scale = torch.randn(3, device="cuda")
    base = torch.randn(24, device="cuda")
    norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    merged, expected_mix, expected_post = _hc_pre_reference(
        x, fn, scale, base, streams, 1e-5, 1e-6, 20, normalize_before_projection=True
    )
    merged = merged.float()
    expected_input = (merged * torch.rsqrt(merged.square().mean(-1, keepdim=True) + 1e-5) * norm.float()).to(x.dtype)
    expected_mix, expected_post = expected_mix.to(x.dtype), expected_post.to(x.dtype)

    layer = SimpleNamespace(mhc_streams=streams, eps_=1e-5, hc_eps=1e-6, hc_sinkhorn_iters=20)
    weights = SimpleNamespace(
        **{
            f"hc_{prefix}_{name}": SimpleNamespace(weight=value)
            for name, value in (("fn", fn), ("scale", scale), ("base", base))
        }
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    actual_input, actual_mix, actual_post = Glm5NextTransformerLayerInfer._hc_pre(
        layer, x, weights, prefix, SimpleNamespace(weight=norm)
    )
    assert torch.backends.cuda.matmul.allow_tf32
    assert actual_mix.dtype == actual_post.dtype == x.dtype
    # FP32 reduction differences can cross the merged BF16 midpoint; RMSNorm
    # rounds to BF16 again. Check FP32 projection separately at strict tolerance.
    torch.testing.assert_close(actual_input, expected_input, atol=0.004, rtol=2 * torch.finfo(x.dtype).eps)
    torch.testing.assert_close(actual_mix, expected_mix, atol=1e-5, rtol=0.008)
    torch.testing.assert_close(actual_post, expected_post, atol=1e-5, rtol=0.008)
    layer_out = torch.randn(tokens, hidden, device="cuda", dtype=x.dtype)
    torch.testing.assert_close(
        hc_post(layer_out, x, actual_mix, actual_post, streams, glm5_numerics=True),
        _hc_post_reference(layer_out, x, actual_mix, actual_post, streams, round_residual=True),
        atol=0.004,
        rtol=0.008,
    )


@pytest.mark.parametrize("tokens,hidden", [(1, 4096), (19, 780), (3073, 4096)])
def test_glm_mhc_projection_retains_fp32_inputs(tokens, hidden, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    x = torch.randn(tokens, 4 * hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 4 * hidden, device="cuda") * 0.005
    x_fp32 = x.float()
    expected = F.linear(x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + 1e-5), fn)

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    partials = pre_norm._hc_normalized_projection(x, fn, 1e-5)
    assert torch.backends.cuda.matmul.allow_tf32
    torch.testing.assert_close(partials.sum(0), expected, atol=2e-5, rtol=2e-5)


def test_glm_mhc_normalizes_before_cancelling_large_products(monkeypatch):
    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", False)
    monkeypatch.setitem(sys.modules, "deep_gemm", None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # Project-first overflows; normalize-first has finite cancelling products.
    x = torch.full((1, 4 * 17), 2.0**32, device="cuda", dtype=torch.bfloat16)
    fn = torch.zeros(24, x.shape[-1], device="cuda")
    fn[:, 0], fn[:, 1] = 2.0**110, -(2.0**110)
    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + 1e-5)
    expected = F.linear(x_fp32 * inv_rms, fn)
    old_order = F.linear(x_fp32, fn) * inv_rms
    assert torch.isfinite(expected).all()
    assert not torch.isfinite(old_order).all()
    torch.testing.assert_close(pre_norm._hc_normalized_projection(x, fn, 1e-5).sum(0), expected, atol=0, rtol=0)


def test_glm_mhc_rounds_post_coefficients_before_mixing():
    x = torch.full((1, 4 * 17), 1 / 256, device="cuda", dtype=torch.bfloat16)
    fn = torch.zeros(24, x.shape[-1], device="cuda")
    scale = torch.zeros(3, device="cuda")
    base = torch.zeros(24, device="cuda")
    base[4:8] = math.log(1.003 / 0.997)
    norm = torch.ones(17, device="cuda", dtype=x.dtype)
    _, mix, post = hc_pre_norm(x, fn, scale, base, norm, 4, 1e-5, 1e-5, 1e-6, 20, glm5_numerics=True)
    assert mix.dtype == post.dtype == x.dtype
    torch.testing.assert_close(post, torch.ones_like(post), atol=0, rtol=0)
    torch.testing.assert_close(mix, torch.full_like(mix, 0.25), atol=0, rtol=0)
    layer_out = torch.ones(1, 17, device="cuda", dtype=x.dtype)
    actual = hc_post(layer_out, x, mix, post, 4, glm5_numerics=True)
    torch.testing.assert_close(actual, torch.ones_like(actual), atol=0, rtol=0)
    _, unrounded_mix, unrounded_post = _hc_pre_reference(
        x, fn, scale, base, 4, 1e-5, 1e-6, 20, normalize_before_projection=True
    )
    old_result = _hc_post_reference(layer_out, x, unrounded_mix, unrounded_post, 4)
    assert (old_result != actual).all()


def test_glm_mhc_post_rounds_residual_like_training():
    x = torch.zeros(1, 4, 17, device="cuda", dtype=torch.bfloat16)
    x[:, 0], x[:, 1] = 1, 1 / 256
    mix = torch.full((1, 4, 4), 0.25, device="cuda", dtype=x.dtype)
    post = torch.ones(1, 4, device="cuda", dtype=x.dtype)
    layer_out = torch.full((1, 17), 3 / 512, device="cuda", dtype=x.dtype)
    # The native training bmm rounds 0.2509765625 to 0.25 before expansion.
    mixed = torch.bmm(mix.transpose(1, 2), x)
    expected = torch.addcmul(mixed.float(), layer_out[:, None].float(), post[:, :, None].float()).to(x.dtype)
    actual = hc_post(layer_out, x.flatten(1), mix, post, 4, glm5_numerics=True)
    torch.testing.assert_close(actual, expected.flatten(1), atol=0, rtol=0)
    assert (hc_post(layer_out, x.flatten(1), mix, post, 4) != actual).all()


def test_glm_mhc_keeps_pre_coefficients_in_fp32():
    x = torch.zeros(1, 4, 17, device="cuda", dtype=torch.bfloat16)
    x[:, 0], x[:, 1] = 1, -1
    x = x.flatten(1)
    fn = torch.zeros(24, x.shape[-1], device="cuda")
    scale = torch.zeros(3, device="cuda")
    base = torch.zeros(24, device="cuda")
    # Both coefficients round to BF16 0.5, which would erase this input entirely.
    pre_values = torch.tensor([0.5005, 0.5], device="cuda") - 1e-6
    base[:2] = torch.logit(pre_values)
    norm = torch.ones(17, device="cuda", dtype=x.dtype)
    actual, _, _ = hc_pre_norm(x, fn, scale, base, norm, 4, 1e-5, 1e-5, 1e-6, 20, glm5_numerics=True)
    merged, _, _ = _hc_pre_reference(x, fn, scale, base, 4, 1e-5, 1e-6, 20, normalize_before_projection=True)
    merged = merged.float()
    expected = (merged * torch.rsqrt(merged.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)
    assert (expected > 0.1).all()
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.008)


def test_glm_mhc_rounds_post_product_before_residual_add():
    x = torch.full((1, 4 * 17), -1.015625, device="cuda", dtype=torch.bfloat16)
    layer = torch.full((1, 17), 1.0078125, device="cuda", dtype=x.dtype)
    mix = torch.eye(4, device="cuda", dtype=x.dtype).unsqueeze(0)
    post = torch.full((1, 4), 1.0078125, device="cuda", dtype=x.dtype)
    expected = _hc_post_reference(layer, x, mix, post, 4, round_residual=True)
    actual = hc_post(layer, x, mix, post, 4, glm5_numerics=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual) == 0
    assert torch.count_nonzero(hc_post(layer, x, mix, post, 4)) > 0
