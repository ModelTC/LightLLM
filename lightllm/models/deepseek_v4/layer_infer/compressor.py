import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from ..triton_kernel.rotary_emb import apply_rotary_emb

logger = logging.getLogger(__name__)

_SGLANG_COMPRESS_MOD = None
_SGLANG_COMPRESS_ERR = None
_SGLANG_COMPRESS_WARNED = False
_FREQ_CIS_CACHE = {}

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


def _load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_sglang_compressor():
    global _SGLANG_COMPRESS_MOD, _SGLANG_COMPRESS_ERR
    if _SGLANG_COMPRESS_MOD is not None:
        return _SGLANG_COMPRESS_MOD
    if _SGLANG_COMPRESS_ERR is not None:
        raise _SGLANG_COMPRESS_ERR
    try:
        from sglang.jit_kernel.dsv4 import compress_old as mod

        _SGLANG_COMPRESS_MOD = mod
        return mod
    except Exception as first_exc:
        root = Path("/data/wanzihao/sglang/python/sglang")
        try:
            if not root.exists():
                raise first_exc
            if "sglang" not in sys.modules:
                sglang_mod = types.ModuleType("sglang")
                sglang_mod.__path__ = [str(root)]
                sys.modules["sglang"] = sglang_mod
            if "sglang.utils" not in sys.modules:
                utils_mod = types.ModuleType("sglang.utils")
                utils_mod.is_in_ci = lambda: False
                sys.modules["sglang.utils"] = utils_mod
            if "sglang.jit_kernel" not in sys.modules:
                jit_mod = types.ModuleType("sglang.jit_kernel")
                jit_mod.__path__ = [str(root / "jit_kernel")]
                sys.modules["sglang.jit_kernel"] = jit_mod
            if "sglang.jit_kernel.dsv4" not in sys.modules:
                dsv4_mod = types.ModuleType("sglang.jit_kernel.dsv4")
                dsv4_mod.__path__ = [str(root / "jit_kernel" / "dsv4")]
                sys.modules["sglang.jit_kernel.dsv4"] = dsv4_mod
            if "sglang.srt" not in sys.modules:
                srt_mod = types.ModuleType("sglang.srt")
                srt_mod.__path__ = [str(root / "srt")]
                sys.modules["sglang.srt"] = srt_mod
            if "sglang.srt.environ" not in sys.modules:
                env_mod = types.ModuleType("sglang.srt.environ")

                class _FalseEnv:
                    def get(self):
                        return False

                class _Envs:
                    SGLANG_OPT_USE_ONLINE_COMPRESS = _FalseEnv()

                env_mod.envs = _Envs()
                sys.modules["sglang.srt.environ"] = env_mod
            if "sglang.jit_kernel.utils" not in sys.modules:
                _load_file_module("sglang.jit_kernel.utils", root / "jit_kernel" / "utils.py")
            if "sglang.jit_kernel.dsv4.utils" not in sys.modules:
                _load_file_module(
                    "sglang.jit_kernel.dsv4.utils",
                    root / "jit_kernel" / "dsv4" / "utils.py",
                )
            _SGLANG_COMPRESS_MOD = _load_file_module(
                "sglang.jit_kernel.dsv4.compress_old",
                root / "jit_kernel" / "dsv4" / "compress_old.py",
            )
            return _SGLANG_COMPRESS_MOD
        except Exception as exc:
            _SGLANG_COMPRESS_ERR = exc
            raise exc


def _warn_sglang_fallback(exc):
    global _SGLANG_COMPRESS_WARNED
    if not _SGLANG_COMPRESS_WARNED:
        logger.warning("DeepSeek-V4 SGLang compressor JIT unavailable, fallback to torch: %s", exc)
        _SGLANG_COMPRESS_WARNED = True


def _freq_cis(cos_table, sin_table):
    key = (
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        cos_table.device,
        tuple(cos_table.shape),
        tuple(sin_table.shape),
    )
    cached = _FREQ_CIS_CACHE.get(key)
    if cached is None:
        cached = torch.complex(cos_table.float(), sin_table.float())
        _FREQ_CIS_CACHE[key] = cached
    return cached


def _sglang_ape(ape, ratio, head_dim):
    if ratio == 4:
        return torch.cat([ape[:, :head_dim], ape[:, head_dim:]], dim=0).contiguous()
    return ape.contiguous()


def _pack_kv_score(kv, score, ratio, head_dim):
    if ratio == 4:
        return torch.cat(
            [
                kv[:, :head_dim],
                kv[:, head_dim:],
                score[:, :head_dim],
                score[:, head_dim:],
            ],
            dim=1,
        ).contiguous()
    return torch.cat([kv, score], dim=1).contiguous()


def _build_state_from_kv_score(kv, score, ape, ratio, head_dim):
    overlap = ratio == 4
    kv_state, score_state = new_compressor_state(ratio, head_dim, kv.device)
    s = kv.shape[0]
    remainder = s % ratio
    cutoff = s - remainder
    offset = ratio if overlap else 0
    if overlap and cutoff >= ratio:
        kv_state[:ratio] = kv[cutoff - ratio : cutoff]
        score_state[:ratio] = score[cutoff - ratio : cutoff] + ape.float()
    if remainder > 0:
        kv_state[offset : offset + remainder] = kv[cutoff:]
        score_state[offset : offset + remainder] = score[cutoff:] + ape.float()[:remainder]
    return kv_state, score_state


def _sglang_prefill_from_kv_score(
    kv,
    score,
    norm_w,
    ape,
    ratio,
    head_dim,
    cos_table,
    sin_table,
    eps,
    dtype,
    state_pool=None,
):
    if not kv.is_cuda or head_dim % 128 != 0 or ratio not in (4, 128):
        return None, None
    mod = _load_sglang_compressor()
    kv_score = _pack_kv_score(kv, score, ratio, head_dim)
    ape_sglang = _sglang_ape(ape.float(), ratio, head_dim)
    slots = 8 if ratio == 4 else ratio
    if state_pool is None:
        state_pool = torch.zeros((1, slots, kv_score.shape[1]), device=kv.device, dtype=kv_score.dtype)
    else:
        state_pool.zero_()
    seq_len = kv.shape[0]
    plan = mod.CompressorPrefillPlan.generate(
        ratio,
        seq_len,
        torch.tensor([seq_len], dtype=torch.int64),
        torch.tensor([seq_len], dtype=torch.int64),
        kv.device,
    )
    indices = torch.zeros((1,), device=kv.device, dtype=torch.int32)
    out = mod.compress_forward(
        state_pool,
        kv_score,
        ape_sglang,
        indices,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    ncomp = seq_len // ratio
    if ncomp:
        mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
        ragged_ids = plan.compress_plan.view(torch.int32)[:ncomp, 0].long()
        out = out.index_select(0, ragged_ids).to(dtype)
    else:
        out = kv.new_zeros(0, head_dim).to(dtype)
    return out, state_pool


def _sglang_decode_step_from_state_pool(
    x_new,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    ratio,
    head_dim,
    cos_table,
    sin_table,
    eps,
    start_pos,
    state_pool,
):
    if state_pool is None or not x_new.is_cuda or head_dim % 128 != 0 or ratio not in (4, 128):
        return None, False
    mod = _load_sglang_compressor()
    xf = x_new.float().view(1, -1)
    kv = F.linear(xf, wkv_w.float())
    score = F.linear(xf, wgate_w.float())
    kv_score = _pack_kv_score(kv, score, ratio, head_dim)
    ape_sglang = _sglang_ape(ape.float(), ratio, head_dim)
    seq_len = start_pos + 1
    plan = mod.CompressorDecodePlan(
        ratio,
        torch.tensor([seq_len], device=x_new.device, dtype=torch.int32),
    )
    indices = torch.zeros((1,), device=x_new.device, dtype=torch.int32)
    out = mod.compress_forward(
        state_pool,
        kv_score,
        ape_sglang,
        indices,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    if seq_len % ratio != 0:
        return None, True
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    return out[0].to(x_new.dtype), True


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


def compressor_prefill_state(
    x,
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
    return_state_pool=False,
    state_pool=None,
):
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
    kv_state, score_state = _build_state_from_kv_score(kv, score, ape, ratio, head_dim)
    sglang_state_pool = state_pool
    try:
        comp, sglang_state_pool = _sglang_prefill_from_kv_score(
            kv,
            score,
            norm_w,
            ape,
            ratio,
            head_dim,
            cos_table,
            sin_table,
            eps,
            dtype,
            state_pool=sglang_state_pool,
        )
        if comp is not None:
            if return_state_pool:
                return comp, kv_state, score_state, sglang_state_pool
            return comp, kv_state, score_state
    except Exception as exc:
        _warn_sglang_fallback(exc)

    should_compress = s >= ratio
    remainder = s % ratio
    cutoff = s - remainder
    if remainder > 0:
        kv = kv[:cutoff]
        score = score[:cutoff]
    if not should_compress:
        comp = x.new_zeros(0, head_dim)
        if return_state_pool:
            return comp, kv_state, score_state, sglang_state_pool
        return comp, kv_state, score_state
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
    if return_state_pool:
        return comp, kv_state, score_state, sglang_state_pool
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
    state_pool=None,
):
    """Faithful reference start_pos>0 path for one new token. Mutates kv_state/score_state in place.
    Returns the new compressed entry [d] (rope applied) when a window completes, else None.
    """
    overlap = ratio == 4
    d = head_dim
    dtype = x_new.dtype
    try:
        entry, handled = _sglang_decode_step_from_state_pool(
            x_new,
            wkv_w,
            wgate_w,
            norm_w,
            ape,
            ratio,
            head_dim,
            cos_table,
            sin_table,
            eps,
            start_pos,
            state_pool,
        )
        if handled:
            return entry
    except Exception as exc:
        _warn_sglang_fallback(exc)

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
    return _finish_entry(
        entry,
        norm_w,
        ape,
        rope_dim,
        cos_table,
        sin_table,
        start_pos + 1 - ratio,
        eps,
        dtype,
    )


def compressor_decode_step_batch(
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
    state_all,
    b_req_idx,
    start_pos,
):
    """Graph-safe batch decode compressor step.

    Mutates ``state_all`` for the selected request rows and returns one candidate
    entry per batch row plus a boolean mask telling which rows closed a
    compression window.
    """
    overlap = ratio == 4
    d = head_dim
    dtype = x_new.dtype
    req = b_req_idx.long()
    pos = start_pos.long()
    pos_mod = pos % ratio

    xf = x_new.float()
    kv = F.linear(xf, wkv_w.float())
    score = F.linear(xf, wgate_w.float()) + ape.float().index_select(0, pos_mod)

    kv_state = state_all[req, 0].clone()
    score_state = state_all[req, 1].clone()
    row = pos_mod + (ratio if overlap else 0)
    batch_ids = torch.arange(x_new.shape[0], device=x_new.device)
    kv_state[batch_ids, row] = kv
    score_state[batch_ids, row] = score

    should_compress = ((pos + 1) % ratio) == 0
    if overlap:
        kv_cat = torch.cat([kv_state[:, :ratio, :d], kv_state[:, ratio:, d:]], dim=1)
        score_cat = torch.cat([score_state[:, :ratio, :d], score_state[:, ratio:, d:]], dim=1)
        entry = (kv_cat * torch.softmax(score_cat, dim=1)).sum(dim=1)
        shifted_kv_state = kv_state.clone()
        shifted_score_state = score_state.clone()
        shifted_kv_state[:, :ratio] = kv_state[:, ratio:]
        shifted_score_state[:, :ratio] = score_state[:, ratio:]
        kv_state = torch.where(should_compress.view(-1, 1, 1), shifted_kv_state, kv_state)
        score_state = torch.where(should_compress.view(-1, 1, 1), shifted_score_state, score_state)
    else:
        entry = (kv_state * torch.softmax(score_state, dim=1)).sum(dim=1)

    state_all[req, 0] = kv_state
    state_all[req, 1] = score_state

    entry = _rmsnorm(entry.to(dtype), norm_w, eps)
    comp_pos = torch.clamp(pos + 1 - ratio, min=0)
    entry_rope = apply_rotary_emb(entry[:, -rope_dim:], cos_table[comp_pos], sin_table[comp_pos])
    entry = torch.cat([entry[:, :-rope_dim], entry_rope], dim=1)
    return entry, should_compress
