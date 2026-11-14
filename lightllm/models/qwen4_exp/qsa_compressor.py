import triton
import triton.language as tl

from lightllm.common.attention_state.bounded_att.window_compressor.window_compressor_state import (
    WindowCompressorState,
    WindowCompressorStateConfig,
)


class QSACompressorState(WindowCompressorState):
    def __init__(self, layers, dim, rotary_dim, ratio, dtype, requests, mtp_step, tp_rank, tp_world_size, device):
        width = dim + 2 * rotary_dim
        if width % 2:
            raise ValueError("QSA compressor fields must have an even total width")
        super().__init__(
            WindowCompressorStateConfig(
                logical_layer_ids=layers,
                window_size=ratio,
                compress_stride=ratio,
                candidate_dim=width // 2,
                target="indexer",
                dtype=dtype,
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
            ),
            requests,
            mtp_step,
            device,
        )
        self.dim, self.rotary_dim = dim, rotary_dim

    def get_state_type(self):
        return "qwen4_qsa_" + super().get_state_type()

    def reset_bounded_att_state(self, req_indices):
        for request in req_indices:
            self.buffer[:, request].zero_()


@triton.jit
def _pool_ring(
    Raw,
    Cos,
    Sin,
    Ring,
    Indices,
    QueryRows,
    Visible,
    Norm,
    Cache,
    CAPACITY: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    RING_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    token = tl.program_id(0)
    request = tl.load(QueryRows + token).to(tl.int64)
    visible = tl.load(Visible + token)
    if (visible > 0) & (visible % RATIO == 0):
        block = visible // RATIO - 1
        position = block * RATIO + tl.arange(0, BR)
        col = tl.arange(0, BD)
        source = token - RATIO + 1 + tl.arange(0, BR)
        source_mask = (source >= 0) & (tl.arange(0, BR) < RATIO)
        source_request = tl.load(QueryRows + source, source_mask, other=-1)
        source_visible = tl.load(Visible + source, source_mask, other=0)
        current = (source >= 0) & (source_request == request) & (source_visible == position + 1)
        values = tl.load(
            Raw + source[:, None].to(tl.int64) * RAW_STRIDE + col[None, :],
            current[:, None] & (tl.arange(0, BR)[:, None] < RATIO) & (col[None, :] < D),
            other=0,
        ).to(tl.float32)
        previous = tl.load(
            Ring + (request * RING_SIZE + position[:, None] % RING_SIZE) * WIDTH + col[None, :],
            ~current[:, None] & (tl.arange(0, BR)[:, None] < RATIO) & (col[None, :] < D),
            other=0,
        ).to(tl.float32)
        pooled = (tl.sum(values + previous, 0) / RATIO).to(Raw.dtype.element_ty).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / D + EPS)
        weight = tl.load(Norm + col, col < D, other=0).to(tl.float32)
        normalized = ((pooled * inv) * (1 + weight)).to(Raw.dtype.element_ty).to(tl.float32)
        if R > 0:
            first_position = block * RATIO
            current_row = (token - RATIO + 1).to(tl.int64)
            first_request = tl.load(QueryRows + current_row, current_row >= 0, other=-1)
            first_visible = tl.load(Visible + current_row, current_row >= 0, other=0)
            first_current = (current_row >= 0) & (first_request == request) & (first_visible == first_position + 1)
            ring_row = (request * RING_SIZE + first_position % RING_SIZE) * WIDTH
            cos = tl.load(Cos + current_row * R + col, first_current & (col < R), other=0).to(tl.float32)
            sin = tl.load(Sin + current_row * R + col, first_current & (col < R), other=0).to(tl.float32)
            cos += tl.load(Ring + ring_row + D + col, ~first_current & (col < R), other=0).to(tl.float32)
            sin += tl.load(Ring + ring_row + D + R + col, ~first_current & (col < R), other=0).to(tl.float32)
            partner = tl.where(col < R // 2, col + R // 2, col - R // 2)
            partner = tl.minimum(tl.maximum(partner, 0), BD - 1)
            other = tl.gather(normalized, partner, 0)
            other = tl.where(col < R // 2, -other, other)
            left = (normalized * cos).to(Raw.dtype.element_ty).to(tl.float32)
            right = (other * sin).to(Raw.dtype.element_ty).to(tl.float32)
            normalized = tl.where(col < R, left + right, normalized)
        destination = tl.load(Indices + request * CAPACITY + block * RATIO + RATIO - 1).to(tl.int64) // RATIO
        tl.store(Cache + destination * D + col, normalized.to(Raw.dtype.element_ty), col < D)


@triton.jit
def _store_ring(
    Raw,
    Cos,
    Sin,
    Ring,
    Starts,
    Requests,
    Lengths,
    RAW_STRIDE: tl.constexpr,
    RING_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(Requests + row).to(tl.int64)
    start = tl.load(Starts + row)
    count = tl.load(Starts + row + 1) - start
    length = tl.load(Lengths + row)
    offset = tl.program_id(1)
    if (request >= 0) & (offset < count):
        source_row = start + count - 1 - offset
        position = length - 1 - offset
        col = tl.arange(0, BLOCK)
        raw = tl.load(Raw + source_row.to(tl.int64) * RAW_STRIDE + col, col < D, other=0)
        if R > 0:
            cos = tl.load(Cos + source_row.to(tl.int64) * R + col - D, (col >= D) & (col < D + R), other=0)
            sin = tl.load(
                Sin + source_row.to(tl.int64) * R + col - D - R,
                (col >= D + R) & (col < D + 2 * R),
                other=0,
            )
            raw = tl.where(col < D, raw, tl.where(col < D + R, cos, sin))
        tl.store(Ring + (request * RING_SIZE + position % RING_SIZE) * WIDTH + col, raw, col < WIDTH)


def pool_with_ring(indexer, raw, ring, batch, cache):
    _pool_ring[(raw.shape[0],)](
        raw,
        batch.cos,
        batch.sin,
        ring,
        batch.slot_table,
        batch.query_rows,
        batch.visible_lengths,
        indexer.key_norm,
        cache,
        batch.slot_table.shape[1],
        raw.stride(0),
        ring.shape[1],
        ring.shape[2],
        indexer.head_dim,
        batch.cos.shape[1],
        indexer.compress_ratio,
        indexer.eps,
        triton.next_power_of_2(indexer.head_dim),
        triton.next_power_of_2(indexer.compress_ratio),
        enable_fp_fusion=False,
    )
    _store_ring[(batch.active_rows.numel(), ring.shape[1])](
        raw,
        batch.cos,
        batch.sin,
        ring,
        batch.starts,
        batch.active_rows,
        batch.key_lengths,
        raw.stride(0),
        ring.shape[1],
        ring.shape[2],
        indexer.head_dim,
        batch.cos.shape[1],
        triton.next_power_of_2(ring.shape[2]),
    )
