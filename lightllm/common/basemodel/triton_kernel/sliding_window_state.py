import torch
import triton
import triton.language as tl


@triton.jit
def _commit_sliding_window_state(
    LayerBuffer,
    BReqIdx,
    BSeqLen,
    BQSeqLen,
    BQStartLoc,
    stride_token,
    stride_head,
    stride_dim,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    SCRATCH_START: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    tail_offset = tl.program_id(1)
    head_idx = tl.program_id(2)
    req_idx = tl.load(BReqIdx + batch_idx)
    seq_len = tl.load(BSeqLen + batch_idx)
    q_len = tl.load(BQSeqLen + batch_idx)
    q_start = tl.load(BQStartLoc + batch_idx)
    q_offset = tl.maximum(q_len - WINDOW, 0) + tail_offset
    pos = seq_len - q_len + q_offset
    mask_token = q_offset < q_len
    src_token = SCRATCH_START + q_start + q_offset
    dst_token = req_idx * WINDOW + pos % WINDOW
    dims = tl.arange(0, BLOCK_D)
    mask = mask_token & (head_idx < HEAD_NUM) & (dims < HEAD_DIM)
    src = src_token * stride_token + head_idx * stride_head + dims * stride_dim
    dst = dst_token * stride_token + head_idx * stride_head + dims * stride_dim
    value = tl.load(LayerBuffer + src, mask=mask, other=0.0)
    tl.store(LayerBuffer + dst, value, mask=mask)


@torch.no_grad()
def commit_sliding_window_state(
    layer_buffer,
    b_req_idx,
    b_seq_len,
    b_q_seq_len,
    b_q_start_loc,
    sliding_window,
    scratch_start,
    max_q_seq_len,
):
    block_d = triton.next_power_of_2(layer_buffer.shape[-1])
    # Earlier chunk tokens are only used by attention, never by the next step.
    grid = (b_req_idx.shape[0], min(max_q_seq_len, sliding_window), layer_buffer.shape[1])
    _commit_sliding_window_state[grid](
        layer_buffer,
        b_req_idx,
        b_seq_len,
        b_q_seq_len,
        b_q_start_loc,
        *layer_buffer.stride(),
        HEAD_NUM=layer_buffer.shape[1],
        HEAD_DIM=layer_buffer.shape[2],
        WINDOW=sliding_window,
        SCRATCH_START=scratch_start,
        BLOCK_D=block_d,
    )
