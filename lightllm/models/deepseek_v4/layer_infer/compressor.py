import torch
import torch.nn.functional as F
from ..triton_kernel.rotary_emb import apply_rotary_emb

# KV compressor: pools every `ratio` consecutive tokens into one compressed KV entry via gated
# (softmax) pooling + a learned absolute-position bias (ape), RMSNorm, and rope on the trailing
# rope_dim. ratio==4 uses overlapping windows (two-series Ca/Cb scheme). Pure-torch transcription of
# the bundled reference inference/model.py Compressor.forward for the prefill (start_pos==0) path.
# NOTE: the reference also applies an FP8/FP4 QAT activation sim to the compressed entry; omitted here
# for the correctness-first prefill path (negligible vs argmax; revisit if e2e diverges).


def _overlap_transform(tensor, ratio, d, value):
    # tensor: [nwin, ratio, 2*d] -> [nwin, 2*ratio, d]; slots [ratio:]=Cb(current), [:ratio]=Ca(previous window)
    nwin = tensor.shape[0]
    out = tensor.new_full((nwin, 2 * ratio, d), value)
    out[:, ratio:] = tensor[:, :, d:]
    out[1:, :ratio] = tensor[:-1, :, :d]
    return out


def _rmsnorm(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(x.dtype)


def compress_prefill(x, wkv_w, wgate_w, norm_w, ape, ratio, head_dim, rope_dim, cos_table, sin_table, eps):
    """x:[s,dim] (one request, start_pos=0) -> compressed kv [nwin, head_dim] (rope applied to last rope_dim).

    nwin = s // ratio (remainder tokens are decode-state, handled in the decode path). wkv_w/wgate_w:
    [coff*head_dim, dim]; norm_w:[head_dim]; ape:[ratio, coff*head_dim]; cos_table/sin_table: compress rope tables.
    """
    overlap = ratio == 4
    coff = 2 if overlap else 1
    d = head_dim
    s = x.shape[0]
    nwin = s // ratio
    if nwin == 0:
        # fewer than `ratio` tokens -> no completed window -> no compressed entry (matches reference)
        return x.new_zeros(0, head_dim)
    cutoff = nwin * ratio
    xf = x.float()
    kv = F.linear(xf, wkv_w.float())[:cutoff].view(nwin, ratio, coff * d)
    score = F.linear(xf, wgate_w.float())[:cutoff].view(nwin, ratio, coff * d) + ape.float()
    if overlap:
        kv = _overlap_transform(kv, ratio, d, 0.0)
        score = _overlap_transform(score, ratio, d, float("-inf"))
    kv = (kv * torch.softmax(score, dim=1)).sum(dim=1)  # [nwin, d] fp32
    kv = _rmsnorm(kv.to(x.dtype), norm_w, eps)  # [nwin, d]
    pos = torch.arange(nwin, device=x.device) * ratio
    kv_rope = apply_rotary_emb(kv[:, -rope_dim:], cos_table[pos], sin_table[pos])  # cos/sin: [nwin, rope_dim//2]
    return torch.cat([kv[:, :-rope_dim], kv_rope], dim=1)


def new_compressor_state(ratio, head_dim, device, dtype=torch.float32):
    """Per-request compressor running state (matches reference Compressor.kv_state/score_state)."""
    coff = 2 if ratio == 4 else 1
    kv_state = torch.zeros(coff * ratio, coff * head_dim, device=device, dtype=dtype)
    score_state = torch.full((coff * ratio, coff * head_dim), float("-inf"), device=device, dtype=dtype)
    return kv_state, score_state


def _finish_entry(kv, norm_w, ape_unused, rope_dim, cos_table, sin_table, position, eps, dtype):
    kv = _rmsnorm(kv.to(dtype), norm_w, eps)  # [d]
    cos = cos_table[position : position + 1]  # [1, rope_dim//2]
    sin = sin_table[position : position + 1]
    kv_rope = apply_rotary_emb(kv[-rope_dim:].unsqueeze(0), cos, sin)[0]
    return torch.cat([kv[:-rope_dim], kv_rope], dim=0)


def compressor_prefill_state(x, wkv_w, wgate_w, norm_w, ape, ratio, head_dim, rope_dim, cos_table, sin_table, eps):
    """Faithful reference start_pos==0 path (incl. remainder). Returns (entries[ncomp,d], kv_state, score_state).

    entries have rope applied; kv_state/score_state carry the partial window for the decode path.
    """
    overlap = ratio == 4
    coff = 2 if overlap else 1
    d = head_dim
    s = x.shape[0]
    dtype = x.dtype
    xf = x.float()
    kv = F.linear(xf, wkv_w.float())  # [s, coff*d]
    score = F.linear(xf, wgate_w.float())  # [s, coff*d]
    ape = ape.float()
    kv_state, score_state = new_compressor_state(ratio, head_dim, x.device)
    should_compress = s >= ratio
    remainder = s % ratio
    cutoff = s - remainder
    offset = ratio if overlap else 0
    if overlap and cutoff >= ratio:
        kv_state[:ratio] = kv[cutoff - ratio : cutoff]
        score_state[:ratio] = score[cutoff - ratio : cutoff] + ape
    if remainder > 0:
        kv_state[offset : offset + remainder] = kv[cutoff:]
        score_state[offset : offset + remainder] = score[cutoff:] + ape[:remainder]
        kv = kv[:cutoff]
        score = score[:cutoff]
    if not should_compress:
        return x.new_zeros(0, head_dim), kv_state, score_state
    nwin = cutoff // ratio
    kvw = kv.view(nwin, ratio, coff * d)
    scw = score.view(nwin, ratio, coff * d) + ape
    if overlap:
        kvw = _overlap_transform(kvw, ratio, d, 0.0)
        scw = _overlap_transform(scw, ratio, d, float("-inf"))
    comp = (kvw * torch.softmax(scw, dim=1)).sum(dim=1)  # [nwin, d] fp32
    comp = _rmsnorm(comp.to(dtype), norm_w, eps)
    pos = torch.arange(nwin, device=x.device) * ratio
    comp_rope = apply_rotary_emb(comp[:, -rope_dim:], cos_table[pos], sin_table[pos])
    comp = torch.cat([comp[:, :-rope_dim], comp_rope], dim=1)
    return comp, kv_state, score_state


def compressor_decode_step(
    x_new,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    ratio,
    head_dim,
    rope_dim,
    cos_table,
    sin_table,
    eps,
    kv_state,
    score_state,
    start_pos,
):
    """Faithful reference start_pos>0 path for one new token. Mutates kv_state/score_state in place.
    Returns the new compressed entry [d] (rope applied) when a window completes, else None."""
    overlap = ratio == 4
    d = head_dim
    dtype = x_new.dtype
    xf = x_new.float().view(-1)  # [dim]
    kv = F.linear(xf, wkv_w.float())  # [coff*d]
    score = F.linear(xf, wgate_w.float()) + ape.float()[start_pos % ratio]  # [coff*d]
    should_compress = (start_pos + 1) % ratio == 0
    if overlap:
        kv_state[ratio + start_pos % ratio] = kv
        score_state[ratio + start_pos % ratio] = score
        if should_compress:
            kv_cat = torch.cat([kv_state[:ratio, :d], kv_state[ratio:, d:]], dim=0)  # [2*ratio, d]
            sc_cat = torch.cat([score_state[:ratio, :d], score_state[ratio:, d:]], dim=0)
            entry = (kv_cat * torch.softmax(sc_cat, dim=0)).sum(dim=0)  # [d]
            kv_state[:ratio] = kv_state[ratio:]
            score_state[:ratio] = score_state[ratio:]
    else:
        kv_state[start_pos % ratio] = kv
        score_state[start_pos % ratio] = score
        if should_compress:
            entry = (kv_state * torch.softmax(score_state, dim=0)).sum(dim=0)  # [d]
    if not should_compress:
        return None
    return _finish_entry(entry, norm_w, ape, rope_dim, cos_table, sin_table, start_pos + 1 - ratio, eps, dtype)
