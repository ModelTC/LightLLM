# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
import triton

from lightllm.common.basemodel.triton_kernel.mhc import hc_head, hc_post, hc_post_pre_norm, hc_pre_norm
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


@pytest.mark.parametrize("tokens,hidden", [(1, 4096), (8, 4096), (16, 6144), (17, 4096), (129, 4096), (3073, 4096)])
@pytest.mark.parametrize("disable_deepgemm", [False, True], ids=["fused", "torch"])
def test_hc_post_pre_norm_matches_reference(tokens, hidden, disable_deepgemm, monkeypatch):
    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", disable_deepgemm)
    streams = 4
    residual = torch.randn(tokens, streams * hidden, device="cuda", dtype=torch.bfloat16)
    layer_output = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    # Deliberately asymmetric mixing checks input/output stream orientation.
    residual_mix = torch.randn(tokens, streams, streams, device="cuda").softmax(-1)
    post_mix = torch.rand(tokens, streams, device="cuda") * 2
    fn = torch.randn(24, streams * hidden, device="cuda") * 0.005
    scale = torch.randn(3, device="cuda")
    base = torch.randn(24, device="cuda")
    norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    residual_before = residual.clone()
    expected_residual = _hc_post_reference(layer_output, residual, residual_mix, post_mix, streams)
    expected_pre = _hc_pre_reference(expected_residual, fn, scale, base, streams, 1e-6, 1e-6, 20)
    actual = hc_post_pre_norm(
        layer_output, residual, residual_mix, post_mix, fn, scale, base, norm, streams, 1e-6, 1e-5, 1e-6, 20
    )
    torch.testing.assert_close(residual, residual_before, atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected_residual, atol=0.04, rtol=0.02)
    torch.testing.assert_close(actual[0], rmsnorm_forward(expected_pre[0], norm, 1e-5), atol=0.04, rtol=0.03)
    for a, b in zip(actual[2:], expected_pre[1:]):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("tokens,hidden", [(1, 4096), (9, 6144), (129, 4096), (3073, 4096)])
@pytest.mark.parametrize("zeros", [False, True])
def test_hc_head_learned_gates_and_preallocated_output(tokens, hidden, zeros):
    streams = 4
    residual = torch.randn(tokens, streams * hidden, device="cuda", dtype=torch.bfloat16)
    if zeros:
        residual.zero_()
    fn = torch.randn(streams, streams * hidden, device="cuda") * 0.005
    scale = torch.randn(1, device="cuda")
    base = torch.randn(streams, device="cuda")
    inv_rms = torch.rsqrt(residual.float().square().mean(-1, keepdim=True) + 1e-5)
    gates = torch.sigmoid(F.linear(residual.float(), fn) * inv_rms * scale + base) + 1e-6
    expected = (residual.view(tokens, streams, hidden).float() * gates.unsqueeze(-1)).sum(1).bfloat16()
    output = torch.full((tokens, hidden), float("nan"), device="cuda", dtype=torch.bfloat16)
    actual = hc_head(residual, fn, scale, base, streams, 1e-5, 1e-6, out=output)
    assert actual is output
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)


@pytest.mark.parametrize("tokens", [1, 16, 17, 129])
def test_deepseek_v4_deferred_state_and_cuda_graph_replay(tokens):
    from lightllm.models.deepseek_v4.layer_infer import hyper_connection as hc

    streams, hidden = 4, 4096
    residual = torch.randn(tokens, streams, hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, streams * hidden, device="cuda") * 0.005
    scale = torch.randn(3, device="cuda")
    base = torch.randn(24, device="cuda")
    norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    layer_output = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    head_fn, head_scale, head_base = fn[:streams].contiguous(), scale[:1].contiguous(), base[:streams].contiguous()
    allocated = []

    def allocate(shape, **kwargs):
        output = torch.empty(shape, **kwargs)
        allocated.append(output)
        return output

    def run():
        x, original, post, mix = hc.hc_pre(residual, fn, scale, base, 1e-6, 1e-6, 20, norm, 1e-5)
        assert original is residual
        assert x.shape == (tokens, hidden) and post.shape == (tokens, streams, 1)
        x, updated, post, mix = hc.hc_fused_post_pre(
            layer_output, original, post, mix, fn, scale, base, 1e-6, 1e-6, 20, norm, 1e-5
        )
        final = hc.hc_post(x, updated, post, mix)
        output = hc.hc_head(final.flatten(1), head_fn, head_scale, head_base, streams, hidden, 1e-6, 1e-6, allocate)
        assert output is allocated[-1]
        return output

    def reference():
        flat = residual.flatten(1)
        _, mix, post = _hc_pre_reference(flat, fn, scale, base, streams, 1e-6, 1e-6, 20)
        updated = _hc_post_reference(layer_output, flat, mix, post, streams)
        x, mix, post = _hc_pre_reference(updated, fn, scale, base, streams, 1e-6, 1e-6, 20)
        x = rmsnorm_forward(x, norm, 1e-5)
        final = _hc_post_reference(x, updated, mix, post, streams)
        inv_rms = torch.rsqrt(final.float().square().mean(-1, keepdim=True) + 1e-6)
        gates = torch.sigmoid(F.linear(final.float(), head_fn) * inv_rms * head_scale + head_base) + 1e-6
        return (final.view(tokens, streams, hidden).float() * gates.unsqueeze(-1)).sum(1).bfloat16()

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(2):
        residual.normal_()
        layer_output.normal_()
        graph.replay()
        torch.testing.assert_close(actual, reference(), atol=0.06, rtol=0.04)
