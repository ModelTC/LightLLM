import os

import torch


FLASHMLA_MIN_HEADS = 64
FLASHMLA_TOPK_MULTIPLE = 128
DSV4_DEBUG_TORCH_SPARSE_ATTN = os.getenv("DSV4_DEBUG_TORCH_SPARSE_ATTN", "0") == "1"


def _pad_topk_for_flashmla(topk_idxs):
    K = topk_idxs.shape[-1]
    padded_K = ((K + FLASHMLA_TOPK_MULTIPLE - 1) // FLASHMLA_TOPK_MULTIPLE) * FLASHMLA_TOPK_MULTIPLE
    if padded_K == K:
        return topk_idxs.contiguous()
    padded = torch.full((*topk_idxs.shape[:-1], padded_K), -1, device=topk_idxs.device, dtype=topk_idxs.dtype)
    padded[..., :K] = topk_idxs
    return padded.contiguous()


def _compact_topk_indices(topk_idxs, kv_len):
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    topk_lens = valid.sum(dim=-1).to(torch.int32)
    if valid.all():
        return topk_idxs.contiguous(), topk_lens.contiguous()

    compact = torch.full_like(topk_idxs, -1)
    ranks = valid.to(torch.int32).cumsum(dim=-1) - 1
    rows = torch.arange(topk_idxs.shape[0], device=topk_idxs.device).unsqueeze(1).expand_as(topk_idxs)
    compact[rows[valid], ranks[valid].long()] = topk_idxs[valid]
    return compact.contiguous(), topk_lens.contiguous()


def _pad_heads_for_flashmla(q, attn_sink):
    h = q.shape[1]
    if h == FLASHMLA_MIN_HEADS:
        return q.contiguous(), attn_sink.to(torch.float32).contiguous(), h
    if h > FLASHMLA_MIN_HEADS:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA sparse attention only supports up to 64 local heads, got {h}")

    q_pad = q.new_zeros(q.shape[0], FLASHMLA_MIN_HEADS, q.shape[2])
    q_pad[:, :h] = q
    sink_pad = torch.full((FLASHMLA_MIN_HEADS,), -float("inf"), device=q.device, dtype=torch.float32)
    sink_pad[:h] = attn_sink.to(torch.float32)
    return q_pad.contiguous(), sink_pad.contiguous(), h


def _torch_sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    return _torch_sparse_attn_flat(q[0], kv[0], attn_sink, topk_idxs[0], scale).unsqueeze(0)


def _torch_sparse_attn_flat(q, kv, attn_sink, topk_idxs, scale):
    q0 = q.float()
    kv0 = kv.float()
    indices = topk_idxs.long()
    valid = (indices >= 0) & (indices < kv0.shape[0])
    safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
    kv_sel = kv0[safe_indices]
    scores = torch.einsum("mhd,mkd->mhk", q0, kv_sel) * scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    sink = attn_sink.float().view(1, -1)
    max_scores = torch.maximum(scores.max(dim=-1).values, sink)
    exp_scores = torch.exp(scores - max_scores.unsqueeze(-1)).masked_fill(~valid.unsqueeze(1), 0.0)
    exp_sink = torch.exp(sink - max_scores)
    denom = exp_scores.sum(dim=-1) + exp_sink
    out = torch.einsum("mhk,mkd->mhd", exp_scores / denom.unsqueeze(-1), kv_sel)
    return out.to(q.dtype)


def vllm_sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """DeepSeek-V4 sparse MLA through vLLM FlashMLA.

    q:[1,m,h,d], kv:[1,n,d] (single KV head shared over h), attn_sink:[h],
    topk_idxs:[1,m,K] int (-1 = invalid/skip). Returns o:[1,m,h,d].
    """
    b, m, h, d = q.shape
    if b != 1 or kv.shape[0] != 1 or topk_idxs.shape[0] != 1:
        raise RuntimeError("DeepSeek-V4 FlashMLA sparse attention wrapper expects one request per call")
    if d != 512:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA sparse attention requires head_dim=512, got {d}")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA sparse attention requires bf16 q/kv, got {q.dtype}/{kv.dtype}")

    return vllm_sparse_attn_flat(q[0], kv[0], attn_sink, topk_idxs[0], scale).unsqueeze(0)


def vllm_sparse_attn_flat(q, kv, attn_sink, topk_idxs, scale, already_compact=False):
    """FlashMLA sparse attention over a flat KV arena.

    q:[m,h,d], kv:[n,d], topk_idxs:[m,K] int. Indices are global offsets into
    the flat kv tensor, so callers can concatenate per-request KV candidates and
    run one FlashMLA call for the whole batch. When already_compact=True, each
    row must place all valid indices before invalid (-1) entries.
    """
    m, h, d = q.shape
    if d != 512:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA sparse attention requires head_dim=512, got {d}")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise RuntimeError(f"DeepSeek-V4 FlashMLA sparse attention requires bf16 q/kv, got {q.dtype}/{kv.dtype}")
    if q.shape[0] == 0:
        return q.new_empty((0, h, d))

    if DSV4_DEBUG_TORCH_SPARSE_ATTN:
        return _torch_sparse_attn_flat(q, kv, attn_sink, topk_idxs, scale)

    from vllm.third_party.flashmla.flash_mla_interface import flash_mla_sparse_fwd

    q_pad, sink_pad, real_heads = _pad_heads_for_flashmla(q, attn_sink)
    topk_idxs = topk_idxs.to(torch.int32)
    if already_compact:
        valid = (topk_idxs >= 0) & (topk_idxs < kv.shape[0])
        indices = topk_idxs.contiguous()
        topk_lens = valid.sum(dim=-1).to(torch.int32).contiguous()
    else:
        indices, topk_lens = _compact_topk_indices(topk_idxs, kv.shape[0])
    indices = _pad_topk_for_flashmla(indices).unsqueeze(1)
    kv_flat = kv.unsqueeze(1).contiguous()
    out, _, _ = flash_mla_sparse_fwd(
        q=q_pad,
        kv=kv_flat,
        indices=indices,
        sm_scale=scale,
        attn_sink=sink_pad,
        topk_length=topk_lens,
        out=None,
    )
    return out[:, :real_heads].to(q.dtype)


def build_prefill_topk_idxs(seqlen, window, ratio, n_window, device):
    """Per-query candidate indices into [window_kv (n_window tokens) ++ compressed_kv (ncomp entries)].

    Returns int32 [seqlen, window + ncomp] with -1 for invalid. Window part indexes the per-token KV
    (here stored as tokens 0..seqlen-1, so n_window == seqlen); compressed part is offset by n_window.
    For prompts where ncomp <= index_topk the indexer is a no-op, so all causally-valid compressed
    entries are attended (matches the reference for short context).
    """
    t = torch.arange(seqlen, device=device)
    offsets = torch.arange(window, device=device)
    win = t.unsqueeze(1) - (window - 1 - offsets).unsqueeze(0)
    win = torch.where(win >= 0, win, torch.full_like(win, -1))
    if ratio:
        ncomp = seqlen // ratio
        c = torch.arange(ncomp, device=device)
        comp_valid = c.unsqueeze(0) < ((t.unsqueeze(1) + 1) // ratio)  # [s, ncomp]
        comp_idx = (c.unsqueeze(0) + n_window).expand(seqlen, ncomp)
        comp = torch.where(comp_valid, comp_idx, torch.full((seqlen, ncomp), -1, device=device, dtype=torch.long))
        return torch.cat([win, comp], dim=1).int()
    return win.int()
