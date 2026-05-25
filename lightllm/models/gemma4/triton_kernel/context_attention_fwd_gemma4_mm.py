"""Gemma-4 prefill attention kernel with image bidirectional masking.

Gemma-4 was trained with bidirectional attention inside each image span on its
sliding-window layers (matches HF/vllm `use_bidirectional_attention="vision"`).
Other lightllm multimodal models use causal attention on image tokens, so the
shared prefill kernel does not need this — keep the modification scoped to
this gemma4-private file rather than the common path.

The kernel mirrors `context_flashattention_nopad._fwd_kernel` (paged KV via
req_to_token_indexs, prompt_cache_len for chunked prefill, sliding window
support, head_dim=256/512 with BLOCK_M reduction) and adds two ideas borrowed
from `lightllm-neo/.../context_attention_fwd_neo`:

1. Per-Q `b_image_token_end` tensor of shape (sum_q,). For Q tokens inside an
   image span it carries the span's end index; for text tokens it is 0.
   The attention mask becomes `local_or_causal_mask | (k_pos < q_image_end)`.
2. K/V iteration upper bound is extended to `max(local_end, block_image_end)`
   so a Q tile in the middle of an image span actually loads K/V tiles past
   its causal end. Without this, the bidi mask in the original diff was a
   no-op on every tile but the last one of the image span.

The standalone `reference_attention` and `check_once` are runnable as a script
for unit testing image bidi correctness.
"""

import math
import torch
import triton
import triton.language as tl

from lightllm.utils.device_utils import is_tesla


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    B_Image_Token_End,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    kv_group_num,
    b_prompt_cache_len,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    SLIDING_WINDOW_LEFT: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    total_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = total_len - prompt_cache_len  # new tokens this step
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m
    if block_start_loc >= cur_batch_seq_len:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M)
    q_valid = offs_m < cur_batch_seq_len

    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_valid[:, None], other=0.0)

    # Per-Q image_end. 0 for non-image tokens, image-span end for image tokens.
    q_image_end = tl.load(
        B_Image_Token_End + cur_batch_in_all_start_index + offs_m,
        mask=q_valid,
        other=0,
    ).to(tl.int32)

    # Absolute position in the request (prompt_cache_len + offset within new tokens).
    q_pos = prompt_cache_len + offs_m  # [M]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    causal_end = tl.minimum(prompt_cache_len + block_start_loc + BLOCK_M, total_len)
    block_image_end = tl.minimum(tl.max(q_image_end, axis=0), total_len)
    block_end_loc = tl.maximum(causal_end, block_image_end)

    if USE_SLIDING_WINDOW:
        kv_start_index = block_start_loc + prompt_cache_len - SLIDING_WINDOW_LEFT
        kv_start_index = tl.maximum(kv_start_index, 0)
        block_kv_len = block_end_loc - kv_start_index
    else:
        kv_start_index = 0
        block_kv_len = block_end_loc

    for start_n in range(0, block_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_pos = kv_start_index + start_n + offs_n  # [N]
        k_valid = k_pos < block_end_loc

        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * k_pos,
            mask=k_valid,
            other=0,
        ).to(tl.int64)

        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=k_valid[None, :], other=0.0)
        qk = tl.dot(q, k)

        if USE_SLIDING_WINDOW:
            # Sliding window: FA-style left inclusive offset + causal (right=0).
            local_mask = ((q_pos[:, None] - k_pos[None, :]) <= SLIDING_WINDOW_LEFT) & (q_pos[:, None] >= k_pos[None, :])
        else:
            local_mask = q_pos[:, None] >= k_pos[None, :]
        # Image bidi: a Q in image span [_, e) attends to all K with k_pos < e.
        # For text Q (q_image_end == 0) this is k_pos < 0 = always False, so
        # the union with local_mask leaves text-attention unchanged.
        image_mask = k_pos[None, :] < q_image_end[:, None]
        mask = local_mask | image_mask

        qk = tl.where(mask, qk * sm_scale, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=k_valid[:, None], other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(Out + off_o, acc, mask=q_valid[:, None])


@torch.no_grad()
def context_attention_fwd_gemma4_mm(
    q,
    k,
    v,
    o,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    max_input_len,
    req_to_token_indexs,
    b_image_token_end,
    sliding_window=(-1, -1),
):
    """Prefill attention with image bidirectional masking on sliding layers.

    Args:
        sliding_window: ``(-1, -1)`` disables SWA; otherwise ``(left, 0)`` with
            FA-style inclusive left offset and causal right bound (right must be 0).
        b_image_token_end: int32 tensor of shape (sum_q,). For each Q token
            position (in the flattened new-token layout), value is the image
            span's end index (in absolute request position) if the token is
            inside an image span, else 0.
    """
    BLOCK_M = 128 if not is_tesla() else 64
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256, 512}
    if Lk >= 512:
        BLOCK_M = min(BLOCK_M, 32)
    elif Lk >= 256:
        BLOCK_M = min(BLOCK_M, 64)

    sm_scale = 1.0 / (Lq ** 0.5) * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = lambda meta: (triton.cdiv(max_input_len, meta["BLOCK_M"]), batch * head, 1)
    BLOCK_N = BLOCK_M
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 1

    if sliding_window == (-1, -1):
        use_sliding_window = False
        sliding_window_left = -1
    else:
        use_sliding_window = True
        assert int(sliding_window[1]) == 0, "sliding_window right must be 0"
        sliding_window_left = int(sliding_window[0])

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        o,
        b_start_loc,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        b_image_token_end,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        kv_group_num=kv_group_num,
        b_prompt_cache_len=b_prompt_cache_len,
        H=head,
        BLOCK_DMODEL=Lk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_SLIDING_WINDOW=use_sliding_window,
        SLIDING_WINDOW_LEFT=sliding_window_left,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# ---------------------------------------------------------------------------
# Reference implementation + standalone test harness
# ---------------------------------------------------------------------------


def reference_attention(
    q,
    k,
    v,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    req_to_token_indexs,
    b_image_token_end,
    sliding_window=(-1, -1),
):
    """Slow torch reference for the gemma4 mm prefill kernel.

    `sliding_window` is (left, 0) using FA-style inclusive left offset with causal
    right bound. (-1, -1) disables SWA.
    """
    device = q.device
    dtype = q.dtype
    sum_q, Hq, D = q.shape
    Hk = k.shape[1]
    kv_group_num = Hq // Hk

    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)

    if sliding_window == (-1, -1):
        use_sliding_window = False
        sliding_window_left = 0
    else:
        use_sliding_window = True
        sliding_window_left = int(sliding_window[0])
        assert int(sliding_window[1]) == 0, "sliding_window right must be 0"

    batch = b_seq_len.shape[0]
    for b in range(batch):
        req = int(b_req_idx[b].item())
        total_len = int(b_seq_len[b].item())
        prompt_len = int(b_prompt_cache_len[b].item())
        new_len = total_len - prompt_len
        q_start = int(b_start_loc[b].item())

        q_blk = q[q_start : q_start + new_len]  # [M, Hq, D]
        q_image_end = b_image_token_end[q_start : q_start + new_len].to(torch.int64)  # [M]

        token_locs = req_to_token_indexs[req, :total_len].to(torch.int64)
        k_blk = k[token_locs]
        v_blk = v[token_locs]

        k_hq = k_blk.repeat_interleave(kv_group_num, dim=1)
        v_hq = v_blk.repeat_interleave(kv_group_num, dim=1)

        q_pos = torch.arange(prompt_len, total_len, device=device, dtype=torch.int64)
        k_pos = torch.arange(0, total_len, device=device, dtype=torch.int64)

        if use_sliding_window:
            causal = ((q_pos[:, None] - k_pos[None, :]) <= sliding_window_left) & (q_pos[:, None] >= k_pos[None, :])
        else:
            causal = k_pos[None, :] <= q_pos[:, None]
        image = k_pos[None, :] < q_image_end[:, None]
        allow = causal | image

        q_t = q_blk.permute(1, 0, 2).to(torch.float32)
        k_t = k_hq.permute(1, 2, 0).to(torch.float32)
        scores = torch.matmul(q_t, k_t) * scale

        neg = torch.tensor(-1.0e9, device=device, dtype=torch.float32)
        scores = torch.where(allow[None, :, :], scores, neg)
        p = torch.softmax(scores, dim=-1)
        v_t = v_hq.permute(1, 0, 2).to(torch.float32)
        out_hq = torch.matmul(p, v_t)
        out[q_start : q_start + new_len] = out_hq.permute(1, 0, 2).to(dtype)

    return out


def make_test_case(
    device="cuda",
    dtype=torch.bfloat16,
    batch=3,
    Hq=8,
    Hk=4,
    D=256,
    seed=0,
    base_index=50000,
    sliding_window=(-1, -1),
):
    torch.manual_seed(seed)

    prompt_lens = torch.randint(low=0, high=8, size=(batch,), device=device)
    new_lens = torch.randint(low=4, high=24, size=(batch,), device=device)
    total_lens = (prompt_lens + new_lens).to(torch.int32)
    max_total_len = int(total_lens.max().item())
    max_new_len = int(new_lens.max().item())

    b_start_loc = torch.zeros((batch,), device=device, dtype=torch.int32)
    cur = 0
    for b in range(batch):
        b_start_loc[b] = cur
        cur += int(new_lens[b].item())
    sum_q = cur

    b_seq_len = total_lens
    b_prompt_cache_len = prompt_lens.to(torch.int32)
    b_req_idx = torch.arange(batch, device=device, dtype=torch.int32)

    sum_kv = int(total_lens.sum().item())
    kv_size = base_index + sum_kv + 1024
    pool = torch.randperm(kv_size - base_index, device=device, dtype=torch.int64)[:sum_kv] + base_index

    req_to_token_indexs = torch.zeros((batch, max_total_len), device=device, dtype=torch.int32)
    p = 0
    for r in range(batch):
        L = int(total_lens[r].item())
        req_to_token_indexs[r, :L] = pool[p : p + L].to(torch.int32)
        p += L

    # Inject one image span per batch into the new-token region with prob 0.7.
    b_image_token_end = torch.zeros((sum_q,), device=device, dtype=torch.int32)
    for b in range(batch):
        M = int(new_lens[b].item())
        P = int(prompt_lens[b].item())
        start = int(b_start_loc[b].item())
        if M >= 4 and torch.rand((), device=device).item() > 0.3:
            s = int(torch.randint(0, M - 2, (1,), device=device).item())
            span_len = int(torch.randint(2, max(3, M - s + 1), (1,), device=device).item())
            e = min(M, s + span_len)
            # image_end is absolute (request-position) = prompt_len + new-offset
            b_image_token_end[start + s : start + e] = P + e

    q = torch.randn((sum_q, Hq, D), device=device, dtype=dtype)
    k = torch.randn((kv_size, Hk, D), device=device, dtype=dtype)
    v = torch.randn((kv_size, Hk, D), device=device, dtype=dtype)
    o = torch.empty((sum_q, Hq, D), device=device, dtype=dtype)

    return (
        q,
        k,
        v,
        o,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_new_len,
        req_to_token_indexs,
        b_image_token_end,
        sliding_window,
    )


def check_once(seed=0, dtype=torch.bfloat16, sliding_window=(-1, -1), D=256):
    case = make_test_case(seed=seed, dtype=dtype, sliding_window=sliding_window, D=D)
    (
        q,
        k,
        v,
        o,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_new_len,
        req_to_token_indexs,
        b_image_token_end,
        sliding_window,
    ) = case

    context_attention_fwd_gemma4_mm(
        q,
        k,
        v,
        o,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_new_len,
        req_to_token_indexs,
        b_image_token_end,
        sliding_window=sliding_window,
    )

    ref = reference_attention(
        q,
        k,
        v,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        req_to_token_indexs,
        b_image_token_end,
        sliding_window=sliding_window,
    )

    diff = (o - ref).abs()
    max_abs = diff.max().item()
    denom = ref.abs().max().item() + 1e-6
    max_rel = max_abs / denom
    has_image = (b_image_token_end > 0).any().item()
    print(
        f"seed={seed} dtype={dtype} D={D} sw={sliding_window} has_image={has_image} "
        f"max_abs={max_abs:.4e} max_rel={max_rel:.4e}"
    )
    assert max_abs < 5e-2, f"max_abs too large: {max_abs}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA, skip.")
    else:
        # Vary D, sliding window, and image presence.
        for seed in (0, 1, 2):
            check_once(seed=seed, D=128, sliding_window=(-1, -1))
            check_once(seed=seed, D=128, sliding_window=(4096, 0))
            check_once(seed=seed, D=256, sliding_window=(4096, 0))
        print("ok")
