import torch
import torch.nn.functional as F

# Manifold-constrained Hyper-Connections (mHC). Replaces the plain residual add: the hidden state is
# carried as ``hc_mult`` parallel streams. Each sub-layer (attn / ffn) collapses the streams to one
# vector (hc_pre), runs the sub-layer, then re-expands into the streams via learned post/comb weights
# (hc_post). A doubly-stochastic (Sinkhorn-normalized) ``comb`` matrix mixes the residual streams.
# Pure-torch transcription of the bundled reference inference/model.py (Block.hc_pre/hc_post,
# ParallelHead.hc_head) + inference/kernel.py (hc_split_sinkhorn). All math in fp32, as in the reference.


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """mixes:[N, (2+hc)*hc] fp32 -> pre[N,hc], post[N,hc], comb[N,hc,hc] (doubly stochastic)."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc :].view(hc, hc)
    # comb = softmax(comb, dim=-1) + eps
    comb = torch.softmax(comb, dim=-1) + eps
    # one column normalization, then (iters-1) of (row, column)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(streams, hc_fn, hc_scale, hc_base, hc_mult, dim, eps, sinkhorn_iters):
    """streams:[N, hc*dim] -> (collapsed[N,dim], post[N,hc], comb[N,hc,hc])."""
    dtype = streams.dtype
    x = streams.float()  # [N, hc*dim]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(x, hc_fn) * rsqrt  # [N, (2+hc)*hc]
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    streams3 = x.view(-1, hc_mult, dim)
    collapsed = torch.sum(pre.unsqueeze(-1) * streams3, dim=1)  # [N, dim]
    return collapsed.to(dtype), post, comb


def hc_post(x, residual, post, comb, hc_mult, dim):
    """x:[N,dim] sub-layer output, residual:[N, hc*dim] -> [N, hc*dim]."""
    res = residual.float().view(-1, hc_mult, dim)  # [N, hc, dim]
    xf = x.float()
    # post: [N,hc] -> [N,hc,dim]; comb mixes residual streams: out[i] = post[i]*x + sum_j comb[i,j]*res[j]
    y = post.unsqueeze(-1) * xf.unsqueeze(-2) + torch.einsum("nij,njd->nid", comb, res)
    return y.reshape(-1, hc_mult * dim).to(x.dtype)


def hc_head(streams, hc_fn, hc_scale, hc_base, hc_mult, dim, eps):
    """Final stream collapse before the lm_head. streams:[N, hc*dim] -> [N, dim] (sigmoid gate, no sinkhorn)."""
    dtype = streams.dtype
    x = streams.float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(x, hc_fn) * rsqrt  # [N, hc]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + eps  # [N, hc]
    streams3 = x.view(-1, hc_mult, dim)
    collapsed = torch.sum(pre.unsqueeze(-1) * streams3, dim=1)
    return collapsed.to(dtype)
