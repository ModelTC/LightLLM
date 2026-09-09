import torch
import triton
import triton.language as tl


@triton.jit
def _copy_sliding_window_checkpoint(
    Pool,
    ReqToTokens,
    seq_len,
    Checkpoint,
    req_idx,
    table_stride,
    pool_stride_l,
    pool_stride_t,
    WINDOW: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    TOTAL_BYTES: tl.constexpr,
    RESTORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(TOTAL_BYTES, BLOCK), tl.num_programs(0)):
        offsets = tl.cast(block, tl.int64) * BLOCK + tl.arange(0, BLOCK)
        layer = offsets // (WINDOW * TOKEN_BYTES)
        ring_pos = offsets // TOKEN_BYTES % WINDOW
        # Keep the existing CPU checkpoint order: absolute token position % W.
        position = seq_len - 1 - (seq_len - 1 - ring_pos + WINDOW) % WINDOW
        valid = (offsets < TOTAL_BYTES) & (position >= 0)
        slot = tl.load(ReqToTokens + tl.cast(req_idx, tl.int64) * table_stride + position, valid, other=0).to(tl.int64)
        pool_ptr = Pool + layer * pool_stride_l + slot * pool_stride_t + offsets % TOKEN_BYTES
        if RESTORE:
            value = tl.load(Checkpoint + offsets, valid, other=0)
            tl.store(pool_ptr, value, valid)
        else:
            value = tl.load(pool_ptr, valid, other=0)
            tl.store(Checkpoint + offsets, value, offsets < TOTAL_BYTES)


def copy_sliding_window_checkpoint(pool, req_to_tokens, seq_len: int, req_idx: int, checkpoint, restore=False):
    """Gather/scatter GPU slots to a pinned CPU [layer, W, ...] checkpoint, byte-exact."""
    pool_bytes = pool.view(torch.uint8)
    checkpoint_bytes = checkpoint.view(torch.uint8)
    _copy_sliding_window_checkpoint[(16,)](
        pool_bytes,
        req_to_tokens,
        seq_len,
        checkpoint_bytes,
        req_idx,
        req_to_tokens.stride(0),
        pool_bytes.stride(0),
        pool_bytes.stride(1),
        WINDOW=checkpoint.shape[1],
        TOKEN_BYTES=pool_bytes.stride(1),
        TOTAL_BYTES=checkpoint_bytes.numel(),
        RESTORE=restore,
        BLOCK=4096,
    )
