import torch


_SGLANG_COMPRESS_ERR = None
_SGLANG_COMPRESS_MOD = None
_SGLANG_LINEAR_BF16_FP32 = None
_FREQ_CIS_CACHE = {}


def _load_sglang_compressor():
    global _SGLANG_COMPRESS_ERR, _SGLANG_COMPRESS_MOD, _SGLANG_LINEAR_BF16_FP32
    if _SGLANG_COMPRESS_MOD is not None:
        return _SGLANG_COMPRESS_MOD, _SGLANG_LINEAR_BF16_FP32
    if _SGLANG_COMPRESS_ERR is not None:
        raise _SGLANG_COMPRESS_ERR
    try:
        from sglang.jit_kernel.dsv4 import linear_bf16_fp32
        from sglang.jit_kernel.dsv4 import compress_old as compress_mod
    except Exception as exc:
        _SGLANG_COMPRESS_ERR = RuntimeError(
            "DeepSeek-V4 fused compressor requires sglang.jit_kernel.dsv4 "
            "(linear_bf16_fp32 + compress_old). Install/export the SGLang package "
            "or vendor the DSv4 compressor JIT into LightLLM."
        )
        raise _SGLANG_COMPRESS_ERR from exc
    _SGLANG_COMPRESS_MOD = compress_mod
    _SGLANG_LINEAR_BF16_FP32 = linear_bf16_fp32
    return compress_mod, linear_bf16_fp32


def _load_paged_compress_data_fn():
    from sglang.jit_kernel.dsv4 import triton_create_paged_compress_data

    return triton_create_paged_compress_data


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


def _compressor_weight(wkv_w, wgate_w):
    return torch.cat([wkv_w, wgate_w], dim=0).contiguous()


def _project_kv_score(x, wkv_w, wgate_w):
    _, linear_bf16_fp32 = _load_sglang_compressor()
    return linear_bf16_fp32(x, _compressor_weight(wkv_w, wgate_w))


def _state_pool_view(state_pool):
    if state_pool is None:
        raise RuntimeError("DeepSeek-V4 fused compressor requires a persistent state_pool")
    if state_pool.dim() == 4 and state_pool.shape[1] == 1:
        return state_pool.squeeze(1)
    return state_pool


def compressor_prefill_state(
    x,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    ratio,
    head_dim,
    cos_table,
    sin_table,
    eps,
    state_pool,
):
    """start_pos==0 prefill for ONE request: x [s, dim] -> compressed entries [s//ratio, head_dim]
    (rope applied). state_pool is the request's persistent jit state slice [1, slots, coff*2*head_dim];
    it is rebuilt in place so the decode path can continue from the trailing partial window."""
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x, wkv_w, wgate_w)
    pool = _state_pool_view(state_pool)
    pool.zero_()
    seq_len = x.shape[0]
    plan = mod.CompressorPrefillPlan.generate(
        ratio,
        seq_len,
        torch.tensor([seq_len], dtype=torch.int64),
        torch.tensor([seq_len], dtype=torch.int64),
        x.device,
    )
    indices = torch.zeros((1,), device=x.device, dtype=torch.int32)
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        indices,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    ncomp = seq_len // ratio
    if ncomp == 0:
        return x.new_zeros(0, head_dim)
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    ragged_ids = plan.compress_plan.view(torch.int32)[:ncomp, 0].long()
    return out.index_select(0, ragged_ids).to(x.dtype)


def compressor_decode_step_single(
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
    state_pool,
    start_pos,
):
    """One token for ONE request (chunked-prefill extend path). Returns the finished compressed
    entry [head_dim] when (start_pos+1) % ratio == 0, else None. Mutates state_pool in place."""
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x_new.view(1, -1), wkv_w, wgate_w)
    pool = _state_pool_view(state_pool)
    seq_len = start_pos + 1
    plan = mod.CompressorDecodePlan(
        ratio,
        torch.tensor([seq_len], device=x_new.device, dtype=torch.int32),
    )
    indices = torch.zeros((1,), device=x_new.device, dtype=torch.int32)
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        indices,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    if seq_len % ratio != 0:
        return None
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    return out[0].to(x_new.dtype)


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
    state_pool,
    b_req_idx,
    start_pos,
):
    mod, _ = _load_sglang_compressor()
    kv_score = _project_kv_score(x_new, wkv_w, wgate_w)
    pool = _state_pool_view(state_pool)
    seq_lens = (start_pos + 1).to(torch.int32).contiguous()
    plan = mod.CompressorDecodePlan(ratio, seq_lens)
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        b_req_idx.to(torch.int32).contiguous(),
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    should_compress = (seq_lens % ratio) == 0
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    return out.to(x_new.dtype), should_compress


# ---------------------------------------------------------------------------- paged state (c4)
# 与 sglang srt compressor 的 paged 路径同构(compress_old 内核 + 分组槽 indices + overlap
# extra_data): state 槽位由 swa 槽位算术派生(翻译③ state_loc = page*ring + swa_loc%ring,
# 分组槽 = state_loc//ratio),state 随 swa 页生灭,radix 命中零拷贝续算。


def paged_state_rows(num_swa_pages: int, ring: int, ratio: int) -> int:
    """state 池行数 = 页数*ring + ring(HOLD 页) + 1(哨兵行),向上取整到 ratio 整除
    (分组视图 [-1, ratio, last_dim] 需要)。与 sglang CompressStatePool 的 _size 公式一致。"""
    rows = num_swa_pages * ring + ring + 1
    return (rows + ratio - 1) // ratio * ratio


def init_paged_state_pool(buffer: torch.Tensor) -> None:
    """末行为哨兵: kv 半边置 0、score 半边置 -inf(KVAndScore.clear 语义)。其余行无需初始化
    (内核在组起点覆写)。buffer: [rows, 2*coff*head_dim] fp32。"""
    half = buffer.shape[-1] // 2
    buffer[-1, :half].zero_()
    buffer[-1, half:].fill_(float("-inf"))
    return


def _paged_state_group_slot(req_to_token, full_to_swa, b_req_idx, positions, page_size, ring, ratio):
    """位置 -> state 分组槽(= sglang create_paged_compressor_data.get_raw_loc):
    state_loc = (swa_loc//page)*ring + swa_loc%ring; 分组槽 = state_loc//ratio。
    负位置按 sglang 语义 mask 到 0;已出窗(swa_loc<0)的位置落到 -1(哨兵行,score=-inf)。"""
    positions = positions.masked_fill(positions < 0, 0)
    full = req_to_token[b_req_idx.long(), positions]
    swa_loc = full_to_swa[full.long()].long()
    state_loc = torch.div(swa_loc, page_size, rounding_mode="floor") * ring + swa_loc % ring
    state_loc = torch.where(swa_loc < 0, torch.full_like(state_loc, -1), state_loc)
    return torch.div(state_loc, ratio, rounding_mode="floor").to(torch.int32)


def paged_decode_state_slots(
    req_to_token,
    full_to_swa,
    b_req_idx,
    b_seq_len,
    page_size: int,
    ring: int,
    ratio: int,
    hold_req_id: int,
    num_swa_pages: int,
):
    """decode 步的 state 分组槽(写槽 = 当前组 clip_down(seq-1) 的槽,overlap 伙伴 = 前一组)。
    纯张量算术(prep 已写本步 req_to_token),图安全。padding(HOLD)行重定向到 HOLD 页的
    state 槽,隔离其垃圾累加。"""
    seq = b_seq_len.long()
    write_positions = torch.div(seq - 1, ratio, rounding_mode="floor") * ratio
    write_slot = _paged_state_group_slot(req_to_token, full_to_swa, b_req_idx, write_positions, page_size, ring, ratio)
    overlap_slot = _paged_state_group_slot(
        req_to_token, full_to_swa, b_req_idx, write_positions - ratio, page_size, ring, ratio
    )
    hold_slot = num_swa_pages * ring // ratio  # HOLD 页区域([pages*ring, pages*ring+ring))的首个分组槽
    is_hold = b_req_idx.long() == hold_req_id
    write_slot = torch.where(is_hold, torch.full_like(write_slot, hold_slot), write_slot)
    overlap_slot = torch.where(is_hold, torch.full_like(overlap_slot, hold_slot), overlap_slot)
    return write_slot, overlap_slot


def paged_prefill_compress_data(req_to_token, full_to_swa, req_idx: int, ready_len: int, seq_len: int, ring: int):
    """单请求 prefill chunk 的 (indices, extra_data, plan): 与 sglang 同走
    triton_create_paged_compress_data(按请求产出,内核经 plan 逐 token 步进)。仅 c4(overlap)。
    三者都与层无关,同一 forward 内可跨全部 c4 层复用。"""
    mod, _ = _load_sglang_compressor()
    fn = _load_paged_compress_data_fn()
    device = req_to_token.device
    n_new = seq_len - ready_len
    write_loc, extra_data = fn(
        compress_ratio=4,
        is_overlap=True,
        swa_page_size=128,
        ring_size=ring,
        req_pool_indices=torch.tensor([req_idx], device=device, dtype=torch.int64),
        seq_lens=torch.tensor([seq_len], device=device, dtype=torch.int64),
        extend_seq_lens=torch.tensor([n_new], device=device, dtype=torch.int64),
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa,
    )
    plan = mod.CompressorPrefillPlan.generate(
        4,
        n_new,
        torch.tensor([seq_len], dtype=torch.int64),
        torch.tensor([n_new], dtype=torch.int64),
        device,
    )
    return write_loc, extra_data, plan


def compressor_paged_prefill(
    x,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    head_dim,
    cos_table,
    sin_table,
    eps,
    state_buffer,
    compress_data,
    ready_len,
    seq_len,
):
    """单请求 prefill/extend chunk(c4 paged): x [n_new, dim] 为位置 [ready, seq) 的 hidden,
    state 写到 swa 派生的分组槽(compress_data 来自 paged_prefill_compress_data,跨层复用)。
    返回本 chunk 完结组的压缩条目 [seq//4 - ready//4, head_dim](rope 已施加)。"""
    mod, _ = _load_sglang_compressor()
    ratio = 4
    kv_score = _project_kv_score(x, wkv_w, wgate_w)
    pool = state_buffer.view(-1, ratio, state_buffer.shape[-1])
    write_loc, extra_data, plan = compress_data
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        write_loc,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
        extra_data=extra_data,
    )
    ncomp = seq_len // ratio - ready_len // ratio
    if ncomp == 0:
        return x.new_zeros(0, head_dim)
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    ragged_ids = plan.compress_plan.view(torch.int32)[:ncomp, 0].long()
    return out.index_select(0, ragged_ids).to(x.dtype)


def compressor_paged_decode_batch(
    x_new,
    wkv_w,
    wgate_w,
    norm_w,
    ape,
    head_dim,
    cos_table,
    sin_table,
    eps,
    state_buffer,
    write_slot,
    overlap_slot,
    b_seq_len,
):
    """批量 decode 一步(c4 paged): state 槽位为 swa 派生分组槽(paged_decode_state_slots,
    可跨层复用)。返回 (entries [bs, head_dim], should_compress [bs])。"""
    mod, _ = _load_sglang_compressor()
    ratio = 4
    kv_score = _project_kv_score(x_new, wkv_w, wgate_w)
    pool = state_buffer.view(-1, ratio, state_buffer.shape[-1])
    seq_lens = b_seq_len.to(torch.int32).contiguous()
    plan = mod.CompressorDecodePlan(ratio, seq_lens)
    out = mod.compress_forward(
        pool,
        kv_score,
        _sglang_ape(ape.float(), ratio, head_dim),
        write_slot,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
        extra_data=overlap_slot.view(-1, 1),
    )
    should_compress = (seq_lens % ratio) == 0
    mod.compress_fused_norm_rope_inplace(out, norm_w.float(), eps, _freq_cis(cos_table, sin_table), plan)
    return out.to(x_new.dtype), should_compress
