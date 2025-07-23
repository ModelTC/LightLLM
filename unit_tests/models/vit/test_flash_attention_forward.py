import torch
import math
import time
import pytest
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd


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
def test_varlen(dtype, atol, batch=4, heads=8, d=80, device="cuda:0"):
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
    flash_attention_fwd(q, k, v, out_tri, cu, max_len)
    a = time.time()
    for _ in range(100):
        flash_attention_fwd(q, k, v, out_tri, cu, max_len)
    b = time.time()
    print(f"flash_attention_fwd time: {(b - a) / 100 * 1000:.2f} ms")
    out_ref = reference_attention_varlen(q, k, v, cu)

    max_err = (out_ref - out_tri).abs().max().item()
    mean_err = (out_ref - out_tri).abs().mean().item()
    print(f"{dtype}: max {max_err:.6f}, mean {mean_err:.6f}")
    torch.testing.assert_close(out_tri, out_ref, atol=atol, rtol=0)


if __name__ == "__main__":
    pytest.main()
