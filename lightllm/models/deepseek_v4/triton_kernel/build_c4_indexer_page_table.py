import torch
import triton
import triton.language as tl


@triton.jit
def _build_c4_indexer_page_table_kernel(
    token_req_idx,  # [query rows] int
    c4_lengths,  # [query rows] int
    req_to_token_indexs,
    req_to_token_stride,
    page_table,  # [query rows, page_cap] int32
    token_num,
    page_cap,
    hold_req_id,
    COMPRESS_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    page_idx = tl.program_id(0) * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    row = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row < token_num
    req_idx = tl.load(token_req_idx + row, row_mask, other=hold_req_id).to(tl.int64)
    c4_len = tl.load(c4_lengths + row, row_mask, other=0).to(tl.int64)
    page_start = page_idx * PAGE_SIZE
    in_table = row_mask[:, None] & (page_idx[None, :] < page_cap)
    active = in_table & (req_idx[:, None] != hold_req_id) & (page_start[None, :] < c4_len[:, None])

    group_end_position = page_start * COMPRESS_RATIO + (COMPRESS_RATIO - 1)
    mem_index = tl.load(
        req_to_token_indexs + req_idx[:, None] * req_to_token_stride + group_end_position[None, :],
        mask=active,
        other=0,
    ).to(tl.int64)
    compressed_kv_slot = mem_index // COMPRESS_RATIO
    physical_page_idx = compressed_kv_slot // PAGE_SIZE
    tl.store(
        page_table + row[:, None] * page_cap + page_idx[None, :],
        tl.where(active, physical_page_idx, 0).to(tl.int32),
        in_table,
    )


@torch.no_grad()
def build_c4_indexer_page_table(
    token_req_idx: torch.Tensor,
    c4_lengths: torch.Tensor,
    c4_cap: int,
    req_to_token_indexs: torch.Tensor,
    hold_req_id: int,
    out: torch.Tensor = None,
):
    """Build DeepGEMM's query-row page table directly from the request token mapping.

    Repeated request ids are allowed: each row uses its own causal c4 length.
    No intermediate batch page table or request-owned copy is maintained.

    Safe only when each logical c4 page maps to a physical page with matching offsets:
        c4_slot(entry page_idx*64 + o) == page_table[page_idx] * 64 + o
    which the current token-slot allocator guarantees.
    """
    page_size = 64
    assert c4_cap % page_size == 0
    token_num = token_req_idx.shape[0]
    assert c4_lengths.shape == token_req_idx.shape
    page_cap = c4_cap // page_size
    if out is None:
        page_table = torch.empty((token_num, page_cap), dtype=torch.int32, device=token_req_idx.device)
    else:
        assert out.shape == (token_num, page_cap)
        assert out.dtype == torch.int32 and out.device == token_req_idx.device and out.is_contiguous()
        page_table = out
    if token_num == 0:
        return page_table
    block_pages = min(128, triton.next_power_of_2(page_cap))
    # Amortize block scheduling across adjacent queries for long-context prefill.
    block_rows = min(16, triton.next_power_of_2(token_num))
    _build_c4_indexer_page_table_kernel[(triton.cdiv(page_cap, block_pages), triton.cdiv(token_num, block_rows))](
        token_req_idx,
        c4_lengths,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        page_table,
        token_num,
        page_cap,
        int(hold_req_id),
        COMPRESS_RATIO=4,
        PAGE_SIZE=page_size,
        BLOCK_PAGES=block_pages,
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )
    return page_table
