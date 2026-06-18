import torch
import triton
import triton.language as tl


@triton.jit
def _gather_c4_indexer_k_kernel(
    req_idx_ptr,  # [batch] int — req_manager slot per batch position
    c4_len_ptr,  # [batch] int — number of causal c4 entries per request (= seq_len // ratio)
    req_to_token_ptr,
    req_to_token_stride0,
    full_to_c4_ptr,
    SlabFp8_ptr,  # c4 indexer pool, viewed as fp8 (flat)
    SlabF32_ptr,  # same pool, viewed as f32 (flat)
    Kout_fp8_ptr,  # [batch*c4_cap, HEAD_DIM] fp8
    Kout_scale_ptr,  # [batch*c4_cap] f32
    Slots_out_ptr,  # [batch*c4_cap] int32 (compact->c4-slot map; -1 for padding)
    c4_cap,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,  # page_size * head_dim (byte offset of the scale tail)
):
    # entry index on grid-X (limit ~2^31), batch on grid-Y (<= running_max_req_size): c4_cap reaches
    # 65536 at 256K context, which would blow the 65535 grid-Y cap if entries were the grid-Y axis.
    e = tl.program_id(0)
    r = tl.program_id(1)
    # int64: out_pos*HEAD_DIM can exceed int32 at high batch + long context (read side is already int64).
    out_pos = r.to(tl.int64) * c4_cap + e
    c4_len = tl.load(c4_len_ptr + r)
    if e >= c4_len:
        # padding entry: mark slot invalid; K is never read (ke bounds the scorer range).
        tl.store(Slots_out_ptr + out_pos, -1)
        return

    # group-end token of compressed entry e lives at position e*RATIO + (RATIO-1).
    req = tl.load(req_idx_ptr + r).to(tl.int64)
    end_tok = e * RATIO + (RATIO - 1)
    full_slot = tl.load(req_to_token_ptr + req * req_to_token_stride0 + end_tok).to(tl.int64)
    c4_slot = tl.load(full_to_c4_ptr + full_slot).to(tl.int64)
    valid = c4_slot >= 0

    # inline PackedPagePool byte addressing (matches destindex_copy_indexer_k_dsv4 / gather_indexer_k):
    # fp8 K at page*bytes_per_page + tok*head_dim; fp32 scale at (page*bytes_per_page + scale_off)//4 + tok.
    page = c4_slot // PAGE_SIZE
    tok = c4_slot % PAGE_SIZE
    data_base = page * BYTES_PER_PAGE + tok * HEAD_DIM
    scale_base = (page * BYTES_PER_PAGE + SCALE_OFFSET) // 4 + tok

    offs_d = tl.arange(0, HEAD_DIM)
    k_fp8 = tl.load(SlabFp8_ptr + data_base + offs_d, mask=valid, other=0.0)
    k_scale = tl.load(SlabF32_ptr + scale_base, mask=valid, other=0.0)
    tl.store(Kout_fp8_ptr + out_pos * HEAD_DIM + offs_d, k_fp8)
    tl.store(Kout_scale_ptr + out_pos, k_scale)
    tl.store(Slots_out_ptr + out_pos, tl.where(valid, c4_slot, -1).to(tl.int32))


@torch.no_grad()
def gather_c4_indexer_k_ragged(
    mem_manager,
    layer_index: int,
    b_req_idx: torch.Tensor,
    c4_len: torch.Tensor,
    c4_cap: int,
    req_to_token_indexs: torch.Tensor,
):
    """Gather each request's causal c4 indexer keys into a padded-per-request ragged buffer for the
    deep_gemm fp8_mqa_logits scorer (mirrors deepseek3_2's extract_indexer_ks, but reads our
    PackedPagePool by c4 slot instead of a token-indexed [N,1,132] buffer).

    For batch position r and compressed entry e in [0, c4_len[r]):
        c4_slot = full_to_c4[req_to_token[b_req_idx[r], e*ratio + (ratio-1)]]
    The raw fp8 key + f32 scale at that slot land at row r*c4_cap + e of the output (so query token t
    of request r reads keys [r*c4_cap, r*c4_cap + (pos+1)//ratio) -- absolute ks/ke offsets the caller
    builds). Returns (k_fp8 [batch*c4_cap, HEAD_DIM] fp8, k_scale [batch*c4_cap] f32, slots
    [batch*c4_cap] int32 = compact-row -> c4 pool slot, -1 for padding). Fixed shapes -> cuda-graph
    safe (c4_cap is pinned per graph bucket); the padding region is never read by the scorer.
    """
    pool = mem_manager.c4_indexer_pool
    head_dim = mem_manager.indexer_head_dim
    buf = pool.get_layer_buffer(mem_manager.layer_to_c4_idx[layer_index]).view(-1)
    slab_fp8 = buf.view(torch.float8_e4m3fn)
    slab_f32 = buf.view(torch.float32)
    batch = b_req_idx.shape[0]
    n = batch * c4_cap
    k_fp8 = torch.empty((n, head_dim), dtype=torch.float8_e4m3fn, device=buf.device)
    k_scale = torch.empty((n,), dtype=torch.float32, device=buf.device)
    slots = torch.empty((n,), dtype=torch.int32, device=buf.device)
    _gather_c4_indexer_k_kernel[(c4_cap, batch)](
        b_req_idx,
        c4_len,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        mem_manager.full_to_c4_indexs,
        slab_fp8,
        slab_f32,
        k_fp8,
        k_scale,
        slots,
        c4_cap,
        RATIO=4,
        HEAD_DIM=head_dim,
        PAGE_SIZE=pool.page_size,
        BYTES_PER_PAGE=pool.bytes_per_page,
        SCALE_OFFSET=pool.scale_offset_in_page,
        num_warps=1,
    )
    return k_fp8, k_scale, slots


@triton.jit
def _build_c4_indexer_page_table_kernel(
    req_idx_ptr,  # [batch] int
    c4_len_ptr,  # [batch] int
    req_to_token_ptr,
    req_to_token_stride0,
    full_to_c4_ptr,
    page_table_ptr,  # [batch, page_cap] int32
    valid_flag_ptr,  # [1] int32, initialized to 1; set to 0 on layout mismatch
    page_cap,
    hold_req_id,
    RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    VALIDATE: tl.constexpr,
):
    p = tl.program_id(0)
    r = tl.program_id(1)
    req = tl.load(req_idx_ptr + r).to(tl.int64)
    c4_len = tl.load(c4_len_ptr + r).to(tl.int64)
    page_start = p * PAGE_SIZE
    active = (req != hold_req_id) & (page_start < c4_len)

    full_pos0 = page_start * RATIO + (RATIO - 1)
    full_slot0 = tl.load(
        req_to_token_ptr + req * req_to_token_stride0 + full_pos0,
        mask=active,
        other=0,
    ).to(tl.int64)
    c4_slot0 = tl.load(full_to_c4_ptr + full_slot0, mask=active, other=0).to(tl.int64)
    phys_page = c4_slot0 // PAGE_SIZE
    tl.store(page_table_ptr + r * page_cap + p, tl.where(active, phys_page, 0).to(tl.int32))

    if VALIDATE:
        offs = tl.arange(0, PAGE_SIZE)
        e = page_start + offs
        valid = active & (e < c4_len)
        full_pos = e * RATIO + (RATIO - 1)
        full_slot = tl.load(
            req_to_token_ptr + req * req_to_token_stride0 + full_pos,
            mask=valid,
            other=0,
        ).to(tl.int64)
        c4_slot = tl.load(full_to_c4_ptr + full_slot, mask=valid, other=-1).to(tl.int64)
        expected = phys_page * PAGE_SIZE + offs
        ok = tl.where(valid, (c4_slot == expected) & (c4_slot >= 0), True)
        if tl.min(ok.to(tl.int32), axis=0) == 0:
            tl.store(valid_flag_ptr, 0)


@torch.no_grad()
def build_c4_indexer_page_table(
    mem_manager,
    b_req_idx: torch.Tensor,
    c4_len: torch.Tensor,
    c4_cap: int,
    req_to_token_indexs: torch.Tensor,
    hold_req_id: int,
    validate: bool = False,
):
    """Build the logical-c4-page -> physical-c4-page table expected by DeepGEMM paged logits.

    This is safe only when each logical c4 page maps to a physical page with matching offsets:
        c4_slot(entry p*64 + o) == page_table[p] * 64 + o
    The optional validation flag checks that invariant and lets the caller fall back to the
    gather path while we keep the current token-slot allocator.
    """
    pool = mem_manager.c4_indexer_pool
    page_size = pool.page_size
    assert c4_cap % page_size == 0
    batch = b_req_idx.shape[0]
    page_cap = c4_cap // page_size
    page_table = torch.empty((batch, page_cap), dtype=torch.int32, device=b_req_idx.device)
    valid_flag = torch.ones((1,), dtype=torch.int32, device=b_req_idx.device)
    _build_c4_indexer_page_table_kernel[(page_cap, batch)](
        b_req_idx,
        c4_len,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        mem_manager.full_to_c4_indexs,
        page_table,
        valid_flag,
        page_cap,
        int(hold_req_id),
        RATIO=4,
        PAGE_SIZE=page_size,
        VALIDATE=validate,
        num_warps=1,
    )
    return page_table, valid_flag
