import torch
import math
import time
import pytest
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd
from lightllm.models.vit.triton_kernel import flashattention_nopad
from lightllm.common.basemodel.attention_vit.fa3.fp import Fa3VitAttBackend


_fa3_mark = pytest.mark.skipif(
    not flashattention_nopad._flash_attn_v3_available
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="FA3 requires sgl_kernel and a Hopper GPU",
)


def test_fa3_failure_computes_fallback_on_same_invocation(monkeypatch):
    calls = []
    q = torch.zeros(2, 1, 4)
    out = torch.full_like(q, -123)
    cu = torch.tensor([0, 2], dtype=torch.int32)

    def fail_fa3(q, k, v, o, cu_seqlens, max_seqlen):
        calls.append("fa3")
        o[0].fill_(99)  # A backend may fail after partially writing output.
        raise RuntimeError("shape-dependent FA3 failure")

    def fallback(q, k, v, o, cu_seqlens, max_seqlen):
        calls.append("triton")
        o.fill_(7)

    monkeypatch.setattr(flashattention_nopad, "_flash_attn_v3_available", True)
    monkeypatch.setattr(flashattention_nopad, "is_hopper", lambda: True)
    monkeypatch.setattr(flashattention_nopad, "flash_attention_v3_fwd", fail_fa3, raising=False)
    monkeypatch.setattr(flashattention_nopad, "_flash_attention_triton_fwd", fallback)
    flash_attention_fwd(q, q, q, out, cu, 2)
    assert calls == ["fa3", "triton"]
    assert torch.equal(out, torch.full_like(q, 7))
    assert not flashattention_nopad._flash_attn_v3_available

    # Later calls should use the disabled-backend path without retrying FA3.
    out.fill_(-123)
    flash_attention_fwd(q, q, q, out, cu, 2)
    assert calls == ["fa3", "triton", "triton"]
    assert torch.equal(out, torch.full_like(q, 7))


def test_fa3_success_does_not_run_fallback(monkeypatch):
    q = torch.zeros(2, 1, 4)
    out = torch.full_like(q, -123)
    cu = torch.tensor([0, 2], dtype=torch.int32)

    def fa3(q, k, v, o, cu_seqlens, max_seqlen):
        o.fill_(9)

    monkeypatch.setattr(flashattention_nopad, "_flash_attn_v3_available", True)
    monkeypatch.setattr(flashattention_nopad, "is_hopper", lambda: True)
    monkeypatch.setattr(flashattention_nopad, "flash_attention_v3_fwd", fa3, raising=False)
    monkeypatch.setattr(
        flashattention_nopad,
        "_flash_attention_triton_fwd",
        lambda *args: pytest.fail("a successful FA3 call must not run fallback"),
    )
    flash_attention_fwd(q, q, q, out, cu, 2)
    assert torch.equal(out, torch.full_like(q, 9))
    assert flashattention_nopad._flash_attn_v3_available


def reference_attention_varlen(q, k, v, cu):
    """
    q, k, v : (total_len, n_head, D)
    cu_seqlen      : prefix sums (batch+1,)
    """
    total, n_head, d = q.shape
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(d)

    for b in range(cu.numel() - 1):
        s, e = cu[b].item(), cu[b + 1].item()
        q_b, k_b, v_b = q[s:e], k[s:e], v[s:e]  # (seq, head, D)

        q_hsd = q_b.permute(1, 0, 2)  # (head, seq, D)
        k_hds = k_b.permute(1, 2, 0)  # (head, D,  seq)
        v_hsd = v_b.permute(1, 0, 2)  # (head, seq, D)

        scores = torch.matmul(q_hsd, k_hds) * scale  # (head, seq, seq)
        probs = torch.softmax(scores.float(), dim=-1)

        out_hsd = torch.matmul(probs, v_hsd.float())  # (head, seq, D)
        out[s:e] = out_hsd.permute(1, 0, 2).to(q.dtype)  # back to (seq, head, D)

    return out


@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-2), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize(
    "attention_fn",
    [
        flash_attention_fwd,
        pytest.param(
            getattr(flashattention_nopad, "flash_attention_v3_fwd", None),
            marks=_fa3_mark,
        ),
        pytest.param(
            Fa3VitAttBackend._vit_att_fwd,
            marks=_fa3_mark,
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_varlen(dtype, atol, attention_fn, batch=4, heads=8, d=80, device="cuda:0"):
    torch.manual_seed(0)
    lengths = torch.randint(1, 257, (batch,))
    max_len = int(lengths.max().item())

    cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(lengths, 0)
    tot = int(cu[-1])

    q = torch.randn(tot, heads, d, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out_tri = torch.randn_like(q)
    attention_fn(q, k, v, out_tri, cu, max_len)
    a = time.time()
    for _ in range(100):
        attention_fn(q, k, v, out_tri, cu, max_len)
    b = time.time()
    print(f"flash_attention_fwd time: {(b - a) / 100 * 1000:.2f} ms")
    out_ref = reference_attention_varlen(q, k, v, cu)

    max_err = (out_ref - out_tri).abs().max().item()
    mean_err = (out_ref - out_tri).abs().mean().item()
    print(f"{dtype}: max {max_err:.6f}, mean {mean_err:.6f}")
    torch.testing.assert_close(out_tri, out_ref, atol=atol, rtol=0)


if __name__ == "__main__":
    pytest.main()
