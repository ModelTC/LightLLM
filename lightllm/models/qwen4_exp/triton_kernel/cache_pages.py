import triton
import triton.language as tl


@triton.jit
def copy_slot_pages(
    Slots,
    Pages,
    MemIndices,
    PageIndices,
    Readies,
    SLOT_STRIDE: tl.constexpr,
    LAYER_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TOKENS: tl.constexpr,
    LAYERS: tl.constexpr,
    DIM: tl.constexpr,
    OFFLOAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    page_index = tl.load(PageIndices + page).to(tl.int64)
    active = page_index >= 0
    if OFFLOAD:
        active = active & ~tl.load(Readies + page).to(tl.int1)
    if active:
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < TOKENS * LAYERS * DIM
        token = col // (LAYERS * DIM)
        layer = ((col // DIM) % LAYERS).to(tl.int64)
        slot = tl.load(MemIndices + page * TOKENS + token, mask, other=-1).to(tl.int64)
        mask = mask & (slot >= 0)
        live = Slots + slot * SLOT_STRIDE + layer * LAYER_STRIDE + col % DIM
        cached = Pages + page_index * PAGE_STRIDE + col
        if OFFLOAD:
            value = tl.load(live, mask, other=0)
            tl.store(cached, value, mask)
        else:
            value = tl.load(cached, mask, other=0)
            tl.store(live, value, mask)


@triton.jit
def copy_state_pages(
    States,
    Pages,
    StateIndices,
    PageIndices,
    Readies,
    WIDTH: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    OFFLOAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    page_index = tl.load(PageIndices + page).to(tl.int64)
    active = page_index >= 0
    if OFFLOAD:
        active = active & ~tl.load(Readies + page).to(tl.int1)
    if active:
        state = tl.load(StateIndices + page).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < WIDTH
        live = States + state * WIDTH + col
        cached = Pages + page_index * PAGE_STRIDE + col
        if OFFLOAD:
            value = tl.load(live, mask, other=0)
            tl.store(cached, value, mask)
        else:
            value = tl.load(cached, mask, other=0)
            tl.store(live, value, mask)
