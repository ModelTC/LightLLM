import torch
import torch.nn.functional as F

# DeepSeek-V4 attention: MLA with a single shared KV head (head_dim=512), per-head learnable attention
# sink, and a candidate set = sliding-window tokens (size `window`) ++ compressed KV entries. Pure-torch
# transcription of the bundled reference (inference/model.py Attention.forward + kernel.py sparse_attn).
# Correctness-first prefill path. head_dim=512 > 256 so FlashAttention is unusable anyway; a fused
# triton sparse-gather kernel is a perf follow-up.


def torch_sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """Gather-then-softmax attention with a per-head sink, matching reference kernel.sparse_attn.

    q:[b,m,h,d], kv:[b,n,d] (single KV head shared over h), attn_sink:[h] (fp32),
    topk_idxs:[b,m,K] int (-1 = invalid/skip). Returns o:[b,m,h,d].
    """
    b, m, h, d = q.shape
    n = kv.shape[1]
    K = topk_idxs.shape[-1]
    idx = topk_idxs.clamp(min=0).long()  # [b,m,K]
    keys = torch.gather(kv.unsqueeze(1).expand(b, m, n, d), 2, idx.unsqueeze(-1).expand(b, m, K, d))  # [b,m,K,d]
    qf, kf = q.float(), keys.float()
    scores = torch.einsum("bmhd,bmkd->bmhk", qf, kf) * scale  # [b,m,h,K]
    valid = (topk_idxs != -1).unsqueeze(2)  # [b,m,1,K]
    scores = scores.masked_fill(~valid, float("-inf"))
    mx = scores.amax(dim=-1, keepdim=True)  # [b,m,h,1]
    mx = torch.nan_to_num(mx, neginf=0.0)
    ex = (scores - mx).exp()  # [b,m,h,K]
    denom = ex.sum(-1) + (attn_sink.view(1, 1, h) - mx.squeeze(-1)).exp()  # [b,m,h]
    o = torch.einsum("bmhk,bmkd->bmhd", ex, kf) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def build_prefill_topk_idxs(seqlen, window, ratio, n_window, device):
    """Per-query candidate indices into [window_kv (n_window tokens) ++ compressed_kv (ncomp entries)].

    Returns int32 [seqlen, window + ncomp] with -1 for invalid. Window part indexes the per-token KV
    (here stored as tokens 0..seqlen-1, so n_window == seqlen); compressed part is offset by n_window.
    For prompts where ncomp <= index_topk the indexer is a no-op, so all causally-valid compressed
    entries are attended (matches the reference for short context).
    """
    t = torch.arange(seqlen, device=device)
    # sliding window: query t attends tokens [max(0, t-window+1) .. t]
    j = torch.arange(n_window, device=device)
    win = j.unsqueeze(0).expand(seqlen, n_window).clone()  # [s, n_window]
    win_valid = (j.unsqueeze(0) <= t.unsqueeze(1)) & (j.unsqueeze(0) > (t.unsqueeze(1) - window))
    win = torch.where(win_valid, win, torch.full_like(win, -1))
    if ratio:
        ncomp = seqlen // ratio
        c = torch.arange(ncomp, device=device)
        comp_valid = c.unsqueeze(0) < ((t.unsqueeze(1) + 1) // ratio)  # [s, ncomp]
        comp_idx = (c.unsqueeze(0) + n_window).expand(seqlen, ncomp)
        comp = torch.where(comp_valid, comp_idx, torch.full((seqlen, ncomp), -1, device=device, dtype=torch.long))
        return torch.cat([win, comp], dim=1).int()
    return win.int()
