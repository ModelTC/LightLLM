"""Batch tails, strided outputs and readonly fallback for projection dispatch."""

import os
import subprocess
import sys

import pytest
import torch

from lightllm.backends.metal.batched_gemv import project


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
CASES = [(b, 1024, n) for b in range(2, 8) for n in (5120, 7168, 8224)]
CASES += [(b, k, 1024) for b in range(2, 9) for k in (2048, 3584)]


@pytest.mark.parametrize("batch,size,rows", CASES)
def test_batched_projection_cpu_reference_and_untouched_output_padding(batch, size, rows):
    generator = torch.Generator().manual_seed(81505)
    hx = torch.randn(batch, size, dtype=torch.float16, generator=generator)
    hw = torch.randn(rows, size, dtype=torch.float16, generator=generator) * 0.02
    expected = hx.float() @ hw.float().t()
    xs = torch.empty(batch + 1, size, dtype=torch.float16, device="mps")
    ws = torch.empty(rows + 1, size, dtype=xs.dtype, device=xs.device)
    x, weight = xs[1:], ws[1:]
    x.copy_(hx)
    weight.copy_(hw)
    storage = torch.full((batch + 2, rows + 3), -17.0, dtype=xs.dtype, device=xs.device)
    out = storage[1:-1, 1 : rows + 1]
    out.fill_(float("nan"))
    with torch.inference_mode():
        result = project(x, weight, out=out)
    assert result is out
    torch.testing.assert_close(result.cpu().float(), expected, atol=0.002, rtol=0.003)
    actual = storage.cpu()
    assert torch.all(actual[[0, -1]] == -17)
    assert torch.all(actual[1:-1, 0] == -17) and torch.all(actual[1:-1, rows + 1 :] == -17)
    torch.testing.assert_close(x.cpu(), hx, atol=0, rtol=0)
    torch.testing.assert_close(weight.cpu(), hw, atol=0, rtol=0)


@pytest.mark.parametrize("batch", range(9, 17))
def test_large_narrow_projection_retains_torch(batch):
    x = torch.randn(batch, 3584, dtype=torch.float16, device="mps")
    weight = torch.randn(1024, 3584, dtype=x.dtype, device=x.device)
    assert project(x, weight) is None


def test_batched_projection_unsupported_and_aliases_are_readonly():
    x = torch.randn(4, 1024, dtype=torch.float16, device="mps")
    weight = torch.randn(7168, 1024, dtype=x.dtype, device=x.device)
    before = x.cpu().clone()
    assert project(x[:1], weight) is None
    assert project(x.float(), weight.float()) is None
    assert project(x[:, ::2], weight[:, ::2]) is None
    assert project(x, weight, bias=weight[:, 0]) is None
    assert project(x, weight, out=x) is None
    assert project(x.expand(2, *x.shape), weight) is None
    unaligned = torch.empty(4, 1028, device="mps", dtype=x.dtype)[:, 1:1025]
    assert project(unaligned, weight) is None
    alias = torch.empty(4 * 7168, dtype=x.dtype, device=x.device)
    before_alias = alias.cpu().clone()
    assert project(alias[: 4 * 1024].view(4, 1024), weight, out=alias.view(4, 7168)) is None
    torch.testing.assert_close(alias.cpu(), before_alias, equal_nan=True, atol=0, rtol=0)
    torch.testing.assert_close(x.cpu(), before, atol=0, rtol=0)


def test_metal_torch_projection_fallback_writes_offset_and_strided_views():
    # Backend selection happens before common model modules bind providers.
    # Isolate it from a test runner that may have loaded CUDA modules already.
    code = r"""
from types import SimpleNamespace
import torch
from lightllm.common.quantization import MetalNoQuantization
torch.manual_seed(81506)
torch.set_num_threads(4)
method=object.__new__(MetalNoQuantization)
with torch.inference_mode():
    for dtype in (torch.float16,torch.float32):
        hx=torch.randn(3,1031,dtype=dtype)
        hw=torch.randn(1027,1031,dtype=dtype)*.02
        x,w=hx.to('mps'),hw.to('mps')
        for use_bias in (False,True):
            hb=torch.randn(1027,dtype=dtype)*.1 if use_bias else None
            expected=hx.float()@hw.float().t()
            if hb is not None:expected+=hb.float()
            bias=hb.to('mps') if hb is not None else None
            for layout in ('offset','strided'):
                storage=torch.full((4,1027) if layout=='offset' else (3,2055),-17.,dtype=dtype,device='mps')
                out=storage[1:] if layout=='offset' else storage[:,1::2]
                out.fill_(float('nan'))
                actual=method.apply(x,SimpleNamespace(weight=w),out=out,use_custom_tensor_mananger=False,bias=bias)
                assert actual is out
                torch.testing.assert_close(actual.cpu().float(),expected,
                    atol=.002 if dtype==torch.float16 else 2e-5,rtol=.003 if dtype==torch.float16 else 2e-5)
                saved=storage.cpu()
                if layout=='offset':assert torch.all(saved[0]==-17)
                else:assert torch.all(saved[:,::2]==-17)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LIGHTLLM_BACKEND": "metal"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
