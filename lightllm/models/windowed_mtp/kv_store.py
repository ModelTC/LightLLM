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
    W: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, offset = row // N, row % N
    req = tl.load(REQS + batch).to(tl.int64)
    first = tl.load(FIRST + batch).to(tl.int64)
    length = tl.load(LENGTHS + batch)
    end = first + length
    if W > 0:
        sinks = tl.minimum(tl.maximum(S - first, 0), length)
        recent = tl.maximum(tl.maximum(first, S), end - W)
        count = sinks + tl.maximum(end - recent, 0)
        position = tl.where(offset < sinks, first + offset, recent + offset - sinks)
    else:
        count = length
        position = first + offset
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
    W: tl.constexpr,
    S: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(REQS + row // N).to(tl.int64)
    position = tl.load(POS + row)
    if W > 0:
        slot = tl.where(position < S, position, S + (position - S) % W)
    else:
        slot = position
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
    W: tl.constexpr,
    S: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, slot = row // (C + B), row % (C + B)
    req = tl.load(REQS + batch).to(tl.int64)
    end = tl.load(ENDS + req)
    count = tl.minimum(end, C)
    if W > 0:
        sinks = tl.minimum(end, S)
        recent = tl.maximum(S, end - W)
        position = tl.where(slot < sinks, slot, recent + slot - sinks)
        ring_slot = tl.where(position < S, position, S + (position - S) % W)
    else:
        ring_slot = slot
    dim = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    context = tl.load(POOL + (req * C + ring_slot) * WIDTH + dim, (slot < count) & (dim < WIDTH), other=0)
    noise = tl.load(
        NOISE + (batch * B + slot - count) * WIDTH + dim, (slot >= count) & (slot < count + B) & (dim < WIDTH), other=0
    )
    tl.store(OUT + row * WIDTH + dim, context + noise, dim < WIDTH)


class WindowKVStore:
    """Per-layer sink + recent-window K/V, indexed independently of target KV.

    Only newly accepted positions are projected and written. Each window slot
    has at most one writer, including prefill chunks larger than the window.
    On request reuse, the new end/count hides stale slots; all visible positions
    are overwritten by the first prefill. Draft noise K/V never enters the ring.
    """

    def __init__(self, requests, capacity, layers, kv_heads, head_dim, dtype, device, window, sinks):
        self.capacity, self.window, self.sinks = capacity, window, sinks
        self.kv = torch.zeros((layers, requests, capacity, 2 * kv_heads, head_dim), dtype=dtype, device=device)
        self.ends = torch.zeros(requests, dtype=torch.int64, device=device)
        self.counts = torch.zeros(requests, dtype=torch.int32, device=device)

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
            self.window,
            self.sinks,
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
            self.window,
            self.sinks,
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
            self.window,
            self.sinks,
            heads * dim,
            512,
        )
        return output

    def retained_positions(self, reqs):
        """Diagnostic only: logical positions, in attention order."""
        end = self.ends.index_select(0, reqs)[:, None]
        slots = torch.arange(self.capacity, device=reqs.device)[None, :]
        if self.window:
            sinks = end.clamp_max(self.sinks)
            recent = (end - self.window).clamp_min(self.sinks)
            positions = torch.where(slots < sinks, slots, recent + slots - sinks)
        else:
            positions = slots.expand(reqs.numel(), -1)
        return positions.masked_fill(slots >= end.clamp_max(self.capacity), -1)
