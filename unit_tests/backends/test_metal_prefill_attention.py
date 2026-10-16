"""CPU FP32 oracles, tile boundaries and fallback for grouped prefill."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from lightllm.backends.metal.prefill_attention import prefill

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")

CASES = [
    (1, q, k, 1.0)
    for q, k in [
        (2, 2),
        (7, 31),
        (8, 32),
        (15, 127),
        (16, 128),
        (17, 129),
        (19, 1317),
        (31, 2048),
        (32, 2049),
        (33, 2048),
        (112, 112),
        (128, 128),
        (240, 2032),
        (255, 8191),
        (256, 8192),
        (257, 257),
        (511, 1024),
        (512, 2048),
        (513, 8192),
        (1008, 1008),
        (1024, 2048),
        (1024, 8192),
        (1025, 8192),
        (2032, 2032),
        (2048, 8192),
    ]
]
CASES += [
    (2, 17, 129, 1.0),
    (3, 32, 128, 1.0),
    (4, 128, 128, 1.0),
    (16, 112, 112, 1.0),
    (1, 19, 1317, 3.0),
    (1, 128, 2048, 3.0),
    (2, 33, 129, 3.0),
    (2, 257, 1024, 1.0),
    (3, 511, 2048, 1.0),
    (1, 512, 2048, 3.0),
]


@pytest.mark.parametrize("batch,queries,keys,magnitude", CASES)
def test_prefill_cpu_oracle_with_offsets_padding_and_causal_tails(batch, queries, keys, magnitude):
    gen = torch.Generator().manual_seed(91118)
    host = []
    device = []
    for tokens, heads, width, scale in [
        (queries, 12, 260, magnitude),
        (keys, 4, 260, magnitude),
        (keys, 4, 268, 0.3 if magnitude == 1.0 else 3.0),
    ]:
        flat = torch.randn(4 + batch * tokens * heads * width, generator=gen, dtype=torch.float16) * scale
        host.append(flat)
        device.append(flat.to("mps"))

    def views(pool):
        return (
            pool[0][4:].view(batch, queries, 12, 260)[:, :, 1:9, 4:],
            pool[1][4:].view(batch, keys, 4, 260)[:, :, 1:3, 4:],
            pool[2][4:].view(batch, keys, 4, 268)[:, :, 1:3, 4:260],
        )

    hq, hk, hv = views(host)
    q, k, v = views(device)
    start = keys - queries
    mask = torch.arange(keys)[None, :] <= torch.arange(start, keys)[:, None]
    expected = F.scaled_dot_product_attention(
        hq.float().transpose(1, 2),
        hk.float().repeat_interleave(4, 2).transpose(1, 2),
        hv.float().repeat_interleave(4, 2).transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)
    storage = torch.full((q.numel() + 7,), -17.0, device="mps", dtype=q.dtype)
    out = storage[3:-4].view(q.shape)
    out.fill_(float("nan"))
    with torch.inference_mode():
        actual = prefill(q, k, v, start, out=out)
    assert actual is out
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.0005, rtol=0.003)
    saved = storage.cpu()
    assert torch.all(saved[:3] == -17) and torch.all(saved[-4:] == -17)
    for got, original in zip(device, host):
        torch.testing.assert_close(got.cpu(), original, atol=0, rtol=0)


def test_prefill_unsupported_inputs_are_readonly():
    q = torch.randn(1, 16, 8, 256, device="mps", dtype=torch.float16)
    k = torch.randn(1, 128, 2, 256, device="mps", dtype=q.dtype)
    qbefore, kbefore = q.cpu(), k.cpu()
    assert prefill(q.cpu(), k, k, 112) is None
    assert prefill(q, k.cpu(), k, 112) is None
    assert prefill(q.float(), k.float(), k.float(), 112) is None
    assert prefill(q[0], k[0], k[0], 112) is None
    assert prefill(q[:, :1], k, k, 112) is None
    assert prefill(q[:, :, :4], k, k, 112) is None
    assert prefill(q, k[:, :, :1], k[:, :, :1], 112) is None
    assert prefill(q, k, k, -1) is None
    assert prefill(q, k, k, 113) is None
    assert prefill(q, k, k, 112.0, out=q) is None
    assert prefill(q, k, k, 112, out=q) is None
    assert prefill(q.requires_grad_(), k, k, 112) is None
    q.requires_grad_(False)
    q2049 = torch.empty(1, 2049, 8, 256, device=q.device, dtype=q.dtype)
    k8193 = torch.empty(1, 8193, 2, 256, device=q.device, dtype=q.dtype)
    assert prefill(q2049, k8193[:, :8192], k8193[:, :8192], 0) is None
    assert prefill(q, k8193, k8193, 0) is None
    unaligned = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[1:].view(q.shape)
    assert prefill(unaligned, k, k, 112) is None
    cpu_out = torch.full(q.shape, -17.0, dtype=q.dtype)
    assert prefill(q, k, k, 112, out=cpu_out) is None
    assert torch.all(cpu_out == -17)
    torch.testing.assert_close(q.cpu(), qbefore, atol=0, rtol=0)
    torch.testing.assert_close(k.cpu(), kbefore, atol=0, rtol=0)


def test_prefill_group_dispatch_and_controlled_fallback():
    code = r"""
import torch
import torch.nn.functional as F
from unittest.mock import patch
from lightllm.backends.metal.attention import _attend_group
from lightllm.backends.metal.prefill_attention import prefill
from lightllm.common.basemodel.attention.base_att import AttControl
torch.set_num_threads(4)
torch.manual_seed(91119)
with torch.inference_mode():
    q=torch.randn(1,16,8,256,device='mps',dtype=torch.float16)
    k=torch.randn(1,128,2,256,device='mps',dtype=q.dtype)
    v=torch.randn_like(k)
    with patch('lightllm.backends.metal.prefill_attention.prefill',wraps=prefill) as native:
        actual=_attend_group(q,k,v,112,AttControl())
        assert native.call_count==1
    for control in [AttControl(use_sliding_window=True,sliding_window=(31,0)),
                    AttControl(use_alibi=True,tp_alibi=torch.arange(8,device='mps',dtype=torch.float32)/16)]:
        with patch('lightllm.backends.metal.prefill_attention.prefill',
                   side_effect=AssertionError('unsupported control dispatched')):
            result=_attend_group(q,k,v,112,control)
        positions=torch.arange(112,128)[:,None]
        keys=torch.arange(128)[None,:]
        allowed=keys<=positions
        if control.use_sliding_window:allowed &= keys >= positions-31
        mask=allowed
        if control.use_alibi:
            bias=torch.arange(8,dtype=torch.float32)[None,:,None,None]/16*(keys-positions)
            mask=bias.masked_fill(~allowed,-float('inf'))
        expected=F.scaled_dot_product_attention(q.cpu().float().transpose(1,2),
            k.cpu().float().repeat_interleave(4,2).transpose(1,2),
            v.cpu().float().repeat_interleave(4,2).transpose(1,2),attn_mask=mask).transpose(1,2)
        torch.testing.assert_close(result.cpu().float(),expected,atol=.002,rtol=.003)
    with patch('lightllm.backends.metal.prefill_attention.prefill',return_value=None) as native:
        fallback=_attend_group(q,k,v,112,AttControl())
        assert native.call_count==1
    torch.testing.assert_close(actual,fallback,atol=.002,rtol=.003)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LIGHTLLM_BACKEND": "metal"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
