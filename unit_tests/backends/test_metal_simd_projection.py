"""FP32 reference and memory contract for runtime-sized Qwen projections."""
import os
import subprocess
import sys

import pytest
import torch

from lightllm.backends.metal.simd_projection import _library, project

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
SHAPES = [(1024, 5120), (1024, 7168), (1024, 8224), (2048, 1024), (3584, 1024)]


@pytest.mark.parametrize(
    "batch",
    [
        8,
        9,
        15,
        16,
        17,
        23,
        24,
        25,
        31,
        32,
        33,
        40,
        56,
        63,
        64,
        112,
        113,
        120,
        128,
        129,
        240,
        241,
        248,
        256,
        257,
        264,
        271,
        272,
    ],
)
@pytest.mark.parametrize("size,rows", SHAPES)
def test_simd_projection_cpu_oracle_offsets_and_complete_writes(batch, size, rows):
    gen = torch.Generator().manual_seed(91122)
    hx = torch.randn(batch, size, generator=gen, dtype=torch.float16)
    hw = torch.randn(rows, size, generator=gen, dtype=torch.float16) * 0.02
    expected = hx.float() @ hw.float().t()
    xs = torch.full((4 + hx.numel(),), -17.0, device="mps", dtype=hx.dtype)
    ws = torch.full((4 + hw.numel(),), -17.0, device="mps", dtype=hw.dtype)
    x = xs[4:].view(hx.shape)
    weight = ws[4:].view(hw.shape)
    x.copy_(hx)
    weight.copy_(hw)
    storage = torch.full((4 + batch * rows + 9,), -17.0, device="mps", dtype=x.dtype)
    out = storage[4:-9].view(batch, rows)
    out.fill_(float("nan"))
    with torch.inference_mode():
        result = project(x, weight, out=out)
    assert result is out
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.002, rtol=0.003)
    saved = storage.cpu()
    assert torch.all(saved[:4] == -17) and torch.all(saved[-9:] == -17)
    torch.testing.assert_close(x.cpu(), hx, atol=0, rtol=0)
    torch.testing.assert_close(weight.cpu(), hw, atol=0, rtol=0)
    assert torch.all(xs[:4].cpu() == -17) and torch.all(ws[:4].cpu() == -17)


def test_simd_projection_fallback_is_readonly():
    x = torch.randn(32, 1024, device="mps", dtype=torch.float16)
    w = torch.randn(7168, 1024, device="mps", dtype=x.dtype)
    xbefore, wbefore = x.cpu(), w.cpu()
    assert project(x.cpu(), w) is None
    assert project(x, w.cpu()) is None
    assert project(x.float(), w.float()) is None
    assert project(x[:4], w) is None
    assert project(x[:7], w) is None
    assert project(torch.empty(273, 1024, device=x.device, dtype=x.dtype), w) is None
    assert project(x, w[:2048]) is None
    assert project(x, w, bias=w[:, 0]) is None
    assert project(x.requires_grad_(), w) is None
    x.requires_grad_(False)
    assert project(x[:, ::2], w[:, ::2]) is None
    unaligned = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[1:].view(x.shape)
    assert project(unaligned, w) is None
    alias = w.view(-1)[: 32 * 7168].view(32, 7168)
    assert project(x, w, out=alias) is None
    bad_out = torch.full((32, 7169), -17.0, device=x.device, dtype=x.dtype)[:, :7168]
    assert project(x, w, out=bad_out) is None
    assert torch.all(bad_out.cpu() == -17)
    torch.testing.assert_close(x.cpu(), xbefore, atol=0, rtol=0)
    torch.testing.assert_close(w.cpu(), wbefore, atol=0, rtol=0)


def test_simd_projection_token_count_does_not_recompile():
    _library.cache_clear()
    w = torch.randn(5120, 1024, device="mps", dtype=torch.float16) * 0.02
    with torch.inference_mode():
        for batch in (32, 112, 256):
            x = torch.randn(batch, 1024, device="mps", dtype=w.dtype)
            result = project(x, w)
            assert result is not None and torch.isfinite(result).all()
    assert _library.cache_info().misses == 1
    assert _library.cache_info().hits == 2


def test_simd_projection_partial_token_count_does_not_recompile():
    _library.cache_clear()
    w = torch.randn(5120, 1024, device="mps", dtype=torch.float16) * 0.02
    with torch.inference_mode():
        for batch in (33, 113, 257):
            x = torch.randn(batch, 1024, device="mps", dtype=w.dtype)
            result = project(x, w)
            assert result is not None and torch.isfinite(result).all()
    assert _library.cache_info().misses == 1
    assert _library.cache_info().hits == 2


def test_simd_projection_selected_through_metal_weight_interface():
    code = r"""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from lightllm.common.quantization import MetalNoQuantization
from lightllm.backends.metal.simd_projection import project
torch.set_num_threads(4)
torch.manual_seed(91123)
method=object.__new__(MetalNoQuantization)
with torch.inference_mode():
    for batch in (8,9,16,24,25,112,257):
        hx=torch.randn(batch,1024,dtype=torch.float16)
        hw=torch.randn(5120,1024,dtype=torch.float16)*.02
        x,w=hx.to('mps'),hw.to('mps')
        expected=hx.float()@hw.float().t()
        with patch('lightllm.backends.metal.quantization.project_simd',wraps=project) as selected:
            with patch('lightllm.backends.metal.quantization.project_matrix',
                       side_effect=AssertionError('missed native dispatch')):
                actual=method.apply(x,SimpleNamespace(weight=w))
            assert selected.call_count==1
        torch.testing.assert_close(actual.cpu().float(),expected,atol=.002,rtol=.003)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LIGHTLLM_BACKEND": "metal"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
