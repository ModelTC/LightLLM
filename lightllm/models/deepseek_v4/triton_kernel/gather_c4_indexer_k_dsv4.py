import torch
import triton
import triton.language as tl


@triton.jit
def _build_c4_indexer_page_table_kernel(
    req_idx_ptr,  # [batch] int
    c4_len_ptr,  # [batch] int
    req_to_token_ptr,
    req_to_token_stride0,
    full_to_c4_ptr,
    page_table_ptr,  # [batch, page_cap] int32
    page_cap,
    hold_req_id,
    RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
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


@torch.no_grad()
def build_c4_indexer_page_table(
    mem_manager,
    b_req_idx: torch.Tensor,
    c4_len: torch.Tensor,
    c4_cap: int,
    req_to_token_indexs: torch.Tensor,
    hold_req_id: int,
):
    """Build the logical-c4-page -> physical-c4-page table expected by DeepGEMM paged logits.

    Safe only when each logical c4 page maps to a physical page with matching offsets:
        c4_slot(entry p*64 + o) == page_table[p] * 64 + o
    which the current token-slot allocator guarantees.
    """
    pool = mem_manager.c4_indexer_pool
    page_size = pool.page_size
    assert c4_cap % page_size == 0
    batch = b_req_idx.shape[0]
    page_cap = c4_cap // page_size
    page_table = torch.empty((batch, page_cap), dtype=torch.int32, device=b_req_idx.device)
    _build_c4_indexer_page_table_kernel[(page_cap, batch)](
        b_req_idx,
        c4_len,
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        mem_manager.full_to_c4_indexs,
        page_table,
        page_cap,
        int(hold_req_id),
        RATIO=4,
        PAGE_SIZE=page_size,
        num_warps=1,
    )
    return page_table
