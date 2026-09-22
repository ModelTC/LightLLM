"""Bounded KV storage for parallel-block speculative decoding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_new(
    FEATURES,
    REQS,
    STARTS,
    FIRST,
    LENGTHS,
    OUT,
    POS,
    ENDS,
    COUNTS,
    N: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, offset = row // N, row % N
    req = tl.load(REQS + batch).to(tl.int64)
    first = tl.load(FIRST + batch).to(tl.int64)
    length = tl.load(LENGTHS + batch)
    end = first + length
    recent = tl.maximum(first, end - C)
    count = end - recent
    position = recent + offset
    valid = offset < count
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(STARTS + batch).to(tl.int64) + position - first
    value = tl.load(FEATURES + source * H + dim, valid & (dim < H), other=0)
    tl.store(OUT + row * H + dim, value, dim < H)
    if tl.program_id(1) == 0:
        tl.store(POS + row, tl.where(valid, position, -1))
        if offset == 0:
            tl.store(ENDS + req, end)
            tl.store(COUNTS + req, tl.minimum(end, C))


@triton.jit
def _write_ring(
    POOL,
    NEW,
    REQS,
    POS,
    N: tl.constexpr,
    C: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(REQS + row // N).to(tl.int64)
    position = tl.load(POS + row)
    slot = position % C
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (position >= 0) & (dim < WIDTH)
    value = tl.load(NEW + row * WIDTH + dim, valid, other=0)
    tl.store(POOL + (req * C + slot) * WIDTH + dim, value, valid)


@triton.jit
def _pack_ring(
    POOL,
    NOISE,
    REQS,
    ENDS,
    OUT,
    C: tl.constexpr,
    B: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, slot = row // (C + B), row % (C + B)
    req = tl.load(REQS + batch).to(tl.int64)
    end = tl.load(ENDS + req)
    count = tl.minimum(end, C)
    position = tl.maximum(0, end - C) + slot
    ring_slot = position % C
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    context = tl.load(POOL + (req * C + ring_slot) * WIDTH + dim, (slot < count) & (dim < WIDTH), other=0)
    noise = tl.load(
        NOISE + (batch * B + slot - count) * WIDTH + dim, (slot >= count) & (slot < count + B) & (dim < WIDTH), other=0
    )
    tl.store(OUT + row * WIDTH + dim, context + noise, dim < WIDTH)


class WindowKVStore:
    """Per-layer recent-window K/V, indexed independently of target KV.

    Only newly accepted positions are projected and written. Each window slot
    has at most one writer, including prefill chunks larger than the window.
    On request reuse, the new end/count hides stale slots; all visible positions
    are overwritten by the first prefill. Draft noise K/V never enters the ring.
    """

    def __init__(self, requests, layers, kv_heads, head_dim, dtype, device, window):
        self.capacity = window
        self.kv = torch.zeros((layers, requests, window, 2 * kv_heads, head_dim), dtype=dtype, device=device)
        self.ends = torch.zeros(requests, dtype=torch.int64, device=device)
        self.counts = torch.zeros(requests, dtype=torch.int32, device=device)

    def reset(self):
        """Invalidate all request windows without changing captured buffer addresses."""
        self.ends.zero_()
        self.counts.zero_()

    def reset_req(self, req_idx):
        self.ends[req_idx] = 0
        self.counts[req_idx] = 0

    def save_checkpoint(self, req_idx, buffers, slot):
        # Copy contiguous per-layer windows, preserving the physical p % W layout.
        # The prefill stream event covers these D2H copies before radix publication.
        for layer in range(self.kv.shape[0]):
            buffers.kv[slot, layer].copy_(self.kv[layer, req_idx], non_blocking=True)
        buffers.ends[slot : slot + 1].copy_(self.ends[req_idx : req_idx + 1], non_blocking=True)
        buffers.counts[slot : slot + 1].copy_(self.counts[req_idx : req_idx + 1], non_blocking=True)

    def restore_checkpoint(self, req_idx, buffers, slot):
        for layer in range(self.kv.shape[0]):
            self.kv[layer, req_idx].copy_(buffers.kv[slot, layer], non_blocking=True)
        self.ends[req_idx : req_idx + 1].copy_(buffers.ends[slot : slot + 1], non_blocking=True)
        self.counts[req_idx : req_idx + 1].copy_(buffers.counts[slot : slot + 1], non_blocking=True)
        # Small-page matching can release the source node immediately after return.
        # Finish H2D before its CPU slot can be evicted and overwritten.
        if self.kv.is_cuda:
            torch.cuda.current_stream(self.kv.device).synchronize()

    def prepare(self, reqs, features, starts, first, lengths, max_new):
        n = min(max_new, self.capacity)
        packed = features.new_empty((reqs.numel() * n, features.shape[-1]))
        positions = torch.empty(reqs.numel() * n, dtype=torch.int64, device=features.device)
        _gather_new[(reqs.numel() * n, triton.cdiv(features.shape[-1], 512))](
            features,
            reqs,
            starts,
            first,
            lengths,
            packed,
            positions,
            self.ends,
            self.counts,
            n,
            features.shape[-1],
            self.capacity,
            512,
        )
        return packed, positions

    def write(self, layer, reqs, positions, new_kv):
        width = new_kv.shape[-2] * new_kv.shape[-1]
        _write_ring[(positions.numel(), triton.cdiv(width, 512))](
            self.kv[layer],
            new_kv,
            reqs,
            positions,
            positions.numel() // reqs.numel(),
            self.capacity,
            width,
            512,
        )

    def pack(self, layer, reqs, noise_kv, block):
        heads, dim = self.kv.shape[-2:]
        output = noise_kv.new_empty((reqs.numel(), self.capacity + block, heads, dim))
        _pack_ring[(reqs.numel() * (self.capacity + block), triton.cdiv(heads * dim, 512))](
            self.kv[layer],
            noise_kv,
            reqs,
            self.ends,
            output,
            self.capacity,
            block,
            heads * dim,
            512,
        )
        return output

    def retained_positions(self, reqs):
        """Diagnostic only: logical positions, in attention order."""
        end = self.ends.index_select(0, reqs)[:, None]
        slots = torch.arange(self.capacity, device=reqs.device)[None, :]
        positions = (end - self.capacity).clamp_min(0) + slots
        return positions.masked_fill(slots >= end.clamp_max(self.capacity), -1)


class WindowKVPageHelper:
    """Append a request's draft window to the existing PD att_state page.

    The page uses global KV heads, just like ordinary PD KV pages, so P and D
    may use different TP sizes. The ring's physical slot order is preserved.
    """

    def __init__(self, mem_manager, offset=0):
        from lightllm.utils.envs_utils import get_env_start_args

        self.mem_manager = mem_manager
        store = mem_manager.windowed_draft_kv
        args = get_env_start_args()
        tp_world_size = args.tp // args.dp
        layers, _, window, heads, dim = store.kv.shape
        self.shape = (window, layers, heads * tp_world_size, dim)
        self.kv_offset = (offset + 7) // 8 * 8
        self.kv_nbytes = window * layers * heads * tp_world_size * dim * store.kv.element_size()
        self.end_offset = (self.kv_offset + self.kv_nbytes + 7) // 8 * 8
        self.state_nbytes = self.end_offset + 12

    def assert_page_size(self):
        page = self.mem_manager.kv_move_buffer[0]
        page_nbytes = page.numel() * page.element_size()
        assert page_nbytes >= self.state_nbytes, (
            f"PD page bytes {page_nbytes} is smaller than attention state bytes {self.state_nbytes}; "
            "increase --pd_kv_page_size on both P and D nodes"
        )

    def copy_req_page(self, page_index, req_idx, dp_mems, mode):
        from lightllm.common.kv_trans_kernel.nixl_kv_trans import page_io

        assert mode in ("read", "write")
        assert req_idx is not None
        page = self.mem_manager.kv_move_buffer[page_index].view(torch.uint8).reshape(-1)
        kv_page = (
            page[self.kv_offset : self.kv_offset + self.kv_nbytes]
            .view(self.mem_manager.windowed_draft_kv.kv.dtype)
            .view(self.shape)
        )
        end_page = page[self.end_offset : self.end_offset + 8].view(torch.int64)
        count_page = page[self.end_offset + 8 : self.end_offset + 12].view(torch.int32)
        window = self.shape[0]
        indexes = torch.arange(req_idx * window, (req_idx + 1) * window, dtype=torch.int64, device=page.device)
        for tp_index, mem in enumerate(dp_mems):
            store = mem.windowed_draft_kv
            layers, requests, capacity, heads, dim = store.kv.shape
            assert (capacity, layers, heads * len(dp_mems), dim) == self.shape
            assert store.kv.dtype == kv_page.dtype
            page_io(
                mem_indexes=indexes,
                page_tensor=kv_page,
                kv_buffer=store.kv.view(layers, requests * capacity, heads, dim),
                tp_index=tp_index,
                tp_world_size=len(dp_mems),
                mode=mode,
            )
            if mode == "read":
                store.ends[req_idx : req_idx + 1].copy_(end_page, non_blocking=True)
                store.counts[req_idx : req_idx + 1].copy_(count_page, non_blocking=True)
            elif tp_index == 0:
                end_page.copy_(store.ends[req_idx : req_idx + 1], non_blocking=True)
                count_page.copy_(store.counts[req_idx : req_idx + 1], non_blocking=True)
