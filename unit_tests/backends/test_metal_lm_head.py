"""The output head preserves the weight interface and FP16 intermediate logits."""

import os
import subprocess
import sys

import pytest
import torch

from lightllm.backends.metal.lm_head import project


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")


def test_large_vocab_lm_head_fp32_reference_views_and_untouched_sources():
    generator = torch.Generator().manual_seed(81621)
    hx = torch.randn(16, 1024, dtype=torch.float16, generator=generator)
    hw = torch.randn(248320, 1024, dtype=torch.float16, generator=generator) * 0.02
    expected = hx.float() @ hw.float().t()
    x_storage = torch.empty(17, 1024, dtype=torch.float16, device="mps")
    w_storage = torch.empty(248321, 1024, dtype=x_storage.dtype, device=x_storage.device)
    x, weight = x_storage[1:], w_storage[1:]
    x.copy_(hx)
    weight.copy_(hw)
    with torch.inference_mode():
        actual = project(x.t(), weight)
        assert actual is not None and actual.shape == (248320, 16)
        assert actual.dtype == torch.float16
        assert actual.t().is_contiguous()
        torch.testing.assert_close(actual.t().cpu().float(), expected, atol=0.002, rtol=0.003)
        storage = torch.full((248322, 35), -17.0, dtype=x.dtype, device=x.device)
        out = storage[1:-1, 1:33:2]
        out.fill_(float("nan"))
        assert project(x.t(), weight, out=out) is out
        torch.testing.assert_close(out.t().cpu().float(), expected, atol=0.002, rtol=0.003)
    saved = storage.cpu()
    assert torch.all(saved[[0, -1]] == -17)
    assert torch.all(saved[1:-1, ::2] == -17) and torch.all(saved[1:-1, 33:] == -17)
    torch.testing.assert_close(x.cpu(), hx, atol=0, rtol=0)
    torch.testing.assert_close(weight.cpu(), hw, atol=0, rtol=0)


def test_unsupported_lm_head_returns_before_writes():
    x = torch.randn(1024, 8, dtype=torch.float16, device="mps")
    weight = torch.randn(128, 1024, dtype=x.dtype, device=x.device)
    out = torch.full((128, 8), -17.0, dtype=x.dtype, device=x.device)
    assert project(x, weight, out=out) is None
    assert project(x.cpu(), weight.cpu()) is None
    assert torch.all(out.cpu() == -17)


def test_lm_head_weight_backend_dispatch_and_fallback():
    code = r"""
from unittest.mock import patch
import torch
from lightllm.backends.metal.operators import MetalOperatorBackend
from lightllm.common.basemodel.layer_weights.meta_weights.embedding_weight import LMHeadWeight
op=object.__new__(LMHeadWeight)
op.weight=torch.arange(28,dtype=torch.float32).view(7,4)
forward=MetalOperatorBackend().select_weight_forward(op)
x=torch.arange(12,dtype=torch.float32).view(4,3)
sentinel=torch.ones(7,3)
with patch('lightllm.backends.metal.lm_head.project',return_value=sentinel) as native:
    assert forward(x) is sentinel
    native.assert_called_once_with(x,op.weight,out=None)
with patch('lightllm.backends.metal.lm_head.project',return_value=None):
    out=torch.empty(7,3)
    assert forward(x,out=out,alloc_func=lambda *a,**k:None) is out
    torch.testing.assert_close(out,op.weight@x)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LIGHTLLM_BACKEND": "metal"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
