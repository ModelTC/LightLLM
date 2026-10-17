"""Copy the previous window prefix into a forward-local packed KV workspace."""

import triton
import triton.language as tl


@triton.jit
def swa_page(
    req,
    logical,
    metadata,
    RING_PAGES: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    HOLD_PAGE: tl.constexpr,
    PREFILL: tl.constexpr,
):
    physical = req * RING_PAGES + logical % RING_PAGES
    if PREFILL:
        first = tl.load(metadata + req * 3)
        end = tl.load(metadata + req * 3 + 1)
        start = tl.load(metadata + req * 3 + 2)
        physical = tl.where((logical >= first) & (logical < end), start + logical - first, physical)
    return tl.where(req == HOLD_REQ, HOLD_PAGE, physical)


@triton.jit
def _prepare_prefill_pages(
    metadata,
    req_ids,
    ready_lens,
    seq_lens,
    BATCH: tl.constexpr,
    WINDOW: tl.constexpr,
    RING_PAGES: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    WORKSPACE_BASE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.arange(0, BLOCK)
    req = tl.load(req_ids + batch, batch < BATCH, 0)
    ready = tl.load(ready_lens + batch, batch < BATCH, 0)
    end = tl.cdiv(tl.load(seq_lens + batch, batch < BATCH, 0), 128)
    first = tl.maximum(0, ready - WINDOW + 1) // 128
    # The whole attention interval must have distinct slots. Short chunks fit
    # directly in the fixed ring; only overflow needs a forward workspace.
    workspace = (end - first > RING_PAGES) & (req != HOLD_REQ) & (batch < BATCH)
    pages = tl.where(workspace, end - first, 0)
    start = WORKSPACE_BASE + tl.cumsum(pages) - pages
    tl.store(metadata + req * 3, tl.where(workspace, first, 0), batch < BATCH)
    tl.store(metadata + req * 3 + 1, tl.where(workspace, end, 0), batch < BATCH)
    tl.store(metadata + req * 3 + 2, tl.where(workspace, start, 0), batch < BATCH)


@triton.jit
def _copy_prefix(
    buffer,
    metadata,
    req_ids,
    ready_lens,
    RING_PAGES: tl.constexpr,
    HOLD_REQ: tl.constexpr,
    LAYER_STRIDE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, prefix_page, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req = tl.load(req_ids + batch)
    if req == HOLD_REQ or tl.load(metadata + req * 3 + 1) == 0:
        return
    first = tl.load(metadata + req * 3)
    logical = first + prefix_page
    if logical * 128 >= tl.load(ready_lens + batch):
        return
    start = tl.load(metadata + req * 3 + 2)
    source = req * RING_PAGES + logical % RING_PAGES
    target = start + prefix_page
    lanes = tl.arange(0, BLOCK)
    for chunk in range(0, PAGE_BYTES, BLOCK):
        offset = chunk + lanes
        value = tl.load(buffer + layer * LAYER_STRIDE + source * PAGE_BYTES + offset, offset < PAGE_BYTES, 0)
        tl.store(buffer + layer * LAYER_STRIDE + target * PAGE_BYTES + offset, value, offset < PAGE_BYTES)


def prepare_prefill_workspace(requests, req_ids, ready_lens, seq_lens, microbatch_index):
    if not req_ids.numel():
        return
    _prepare_prefill_pages[(1,)](
        requests.prefill_pages,
        req_ids,
        ready_lens,
        seq_lens,
        BATCH=req_ids.numel(),
        WINDOW=requests.sliding_window,
        RING_PAGES=requests.swa_pages_per_req,
        HOLD_REQ=requests.HOLD_REQUEST_ID,
        WORKSPACE_BASE=requests.swa_runtime_pages + microbatch_index * requests.swa_scratch_pages,
        BLOCK=triton.next_power_of_2(req_ids.numel()),
        num_warps=4,
    )
    buffer = requests.swa_buffer
    _copy_prefix[(req_ids.numel(), triton.cdiv(requests.sliding_window - 1, 128) + 1, buffer.shape[0])](
        buffer,
        requests.prefill_pages,
        req_ids,
        ready_lens,
        RING_PAGES=requests.swa_pages_per_req,
        HOLD_REQ=requests.HOLD_REQUEST_ID,
        LAYER_STRIDE=buffer.stride(0),
        PAGE_BYTES=buffer.shape[-1],
        BLOCK=4096,
        num_warps=4,
    )
