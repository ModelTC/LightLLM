# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
import triton

from lightllm.common.basemodel.triton_kernel.mhc import hc_post, hc_pre_norm
from lightllm.common.basemodel.triton_kernel.mhc import pre_norm
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _hc_pre_reference(x, fn, scale, base, streams, rms_eps, hc_eps, sinkhorn_iters, post_multiplier=2.0):
    """Unfused FP32 reference; returns the merged input before sublayer RMSNorm."""
    assert x.ndim == 2 and x.shape[-1] % streams == 0
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    residual = x.view(tokens, streams, hidden)

    x_fp32 = x.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + rms_eps)
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


def _hc_post_reference(layer_output, residual, residual_mix, post_mix, streams):
    """Unfused FP32 reference; residual_mix maps input streams to output streams."""
    tokens, hidden = layer_output.shape
    residual_3d = residual.view(tokens, streams, hidden)
    mixed_residual = (residual_mix.unsqueeze(-1) * residual_3d.float().unsqueeze(2)).sum(dim=1)
    out = post_mix.unsqueeze(-1) * layer_output.float().unsqueeze(1) + mixed_residual
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
