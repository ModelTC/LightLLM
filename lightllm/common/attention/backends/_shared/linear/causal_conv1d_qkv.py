# SPDX-License-Identifier: Apache-2.0

"""Width-four BF16 prefill convolution fused with Q/K norm and history update."""

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_qkv_kernel(
    x,
    weight,
    conv_states,
    query_start_loc,
    cache_indices,
    cache_seqlens,
    q,
    k,
    v,
    head_num: tl.constexpr,
    seq_num: tl.constexpr,
    PACKED: tl.constexpr,
    head_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    if PACKED:
        tile_id = tl.program_id(1)
        seq_id = tl.full((), 0, tl.int32)
        # floor(token offset / BLOCK_T) + request id gives each request
        # enough tiles, including unaligned starts and empty requests.
        for shift in tl.static_range((seq_num - 1).bit_length() - 1, -1, -1):
            candidate = seq_id + (1 << shift)
            tile_start = tl.load(query_start_loc + candidate, candidate < seq_num, 0) // BLOCK_T + candidate
            seq_id = tl.where((candidate < seq_num) & (tile_start <= tile_id), candidate, seq_id)
        seq_start = tl.load(query_start_loc + seq_id)
        tile_idx = tile_id - (seq_start // BLOCK_T + seq_id)
        head_idx = tl.program_id(0)
    else:
        seq_id = tl.program_id(0)
        seq_start = tl.load(query_start_loc + seq_id)
        tile_idx = tl.program_id(1)
        head_idx = tl.program_id(2)
    token_idx = tile_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    qkv_id = head_idx // head_num
    head_id = head_idx % head_num
    dim_idx = tl.arange(0, head_dim)
    channel_idx = head_idx * head_dim + dim_idx
    seq_len = tl.load(query_start_loc + seq_id + 1) - seq_start
    cache_idx = tl.load(cache_indices + seq_id)
    has_history = tl.load(cache_seqlens + seq_id) > 0
    mask = token_idx[:, None] < seq_len
    if tile_idx * BLOCK_T < seq_len:
        if cache_idx == -1:
            y = tl.load(x + (seq_start + token_idx[:, None]) * stride_x_token + channel_idx[None, :], mask, 0)
        else:
            history0 = tl.full((head_dim,), 0, tl.float32)
            history1 = tl.full((head_dim,), 0, tl.float32)
            history2 = tl.full((head_dim,), 0, tl.float32)
            if has_history and tile_idx == 0:
                state_ptr = conv_states + cache_idx * stride_conv_state_seq + channel_idx * stride_conv_state_dim
                history0 = tl.load(state_ptr).to(tl.float32)
                history1 = tl.load(state_ptr + stride_conv_state_tok).to(tl.float32)
                history2 = tl.load(state_ptr + 2 * stride_conv_state_tok).to(tl.float32)
            acc = tl.full((BLOCK_T, head_dim), 0, tl.float32)
            for conv_idx in tl.static_range(4):
                input_token = token_idx + conv_idx - 3
                x_val = tl.load(
                    x + (seq_start + input_token[:, None]) * stride_x_token + channel_idx[None, :],
                    mask & (input_token[:, None] >= 0),
                    0,
                ).to(tl.float32)
                if conv_idx < 3:
                    if has_history and tile_idx == 0:
                        history = tl.where(
                            input_token[:, None] == -3,
                            history0[None, :],
                            tl.where(input_token[:, None] == -2, history1[None, :], history2[None, :]),
                        )
                        x_val = tl.where(input_token[:, None] < 0, history, x_val)
                w_val = tl.load(weight + channel_idx * 4 + conv_idx).to(tl.float32)
                acc = tl.fma(x_val, w_val[None, :], acc)
            # Preserve the BF16 rounding between convolution and Q/K norm.
            y = (acc / (1 + tl.exp(-acc))).to(q.dtype.element_ty)
        if qkv_id < 2:
            qk = y.to(tl.float32)
            norm = tl.rsqrt(tl.sum(qk * qk, 1)[:, None] + 1e-6)
            y = (qk * norm).to(q.dtype.element_ty)
        output = tl.where(qkv_id == 0, q, tl.where(qkv_id == 1, k, v))
        out_idx = ((seq_start + token_idx[:, None]) * head_num + head_id) * head_dim + dim_idx[None, :]
        tl.store(output + out_idx, y, mask)

    if tile_idx == 0 and cache_idx >= 0:
        # Only this program reads and updates these channels' initial history.
        tl.debug_barrier()
        state_pos = tl.arange(0, 4)
        input_token = seq_len - 3 + state_pos
        state_mask = state_pos[:, None] < 3
        tail = tl.load(
            x + (seq_start + input_token[:, None]) * stride_x_token + channel_idx[None, :],
            state_mask & (input_token[:, None] >= 0),
            0,
        )
        history = tl.load(
            conv_states
            + cache_idx * stride_conv_state_seq
            + channel_idx[None, :] * stride_conv_state_dim
            + (input_token[:, None] + 3) * stride_conv_state_tok,
            # Match SGL: empty requests clear history, including warm requests.
            state_mask & (input_token[:, None] < 0) & has_history & (seq_len > 0),
            0,
        )
        # Finish all history reads before any lane overwrites a cached value.
        tl.debug_barrier()
        state_ptr = (
            conv_states
            + cache_idx * stride_conv_state_seq
            + channel_idx[None, :] * stride_conv_state_dim
            + state_pos[:, None] * stride_conv_state_tok
        )
        tl.store(state_ptr, tl.where(input_token[:, None] >= 0, tail, history), state_mask)


def causal_conv1d_qkv(x, weight, conv_states, query_start_loc, cache_indices, cache_seqlens, max_q_seq_len):
    """Convolve projected QKV and return contiguous Q/K/V with Q/K L2 norm.

    x is BF16 [tokens, 3 * head_num * 128] with unit channel stride;
    weight is contiguous [3 * head_num * 128, 4]. conv_states and request
    metadata follow the KDA backend contract. cache_seqlens gives each
    request's cached token count; zero means no history. The first token tile
    updates each head's raw-input history after consuming the previous history.
    """
    block_t, num_warps = 4, 1
    tokens, qkv_dim = x.shape
    head_num, head_dim = qkv_dim // (3 * 128), 128
    q, k, v = [torch.empty((1, tokens, head_num, head_dim), dtype=x.dtype, device=x.device) for _ in range(3)]
    stride_conv_state_seq, stride_conv_state_dim, stride_conv_state_tok = conv_states.stride()
    seq_num = cache_indices.numel()
    # Avoid launching mostly empty tiles for highly uneven request lengths.
    packed = max_q_seq_len * seq_num > 2 * tokens
    if packed:
        grid = (3 * head_num, tokens // block_t + seq_num)
    else:
        grid = (seq_num, max(1, triton.cdiv(max_q_seq_len, block_t)), 3 * head_num)
    _conv_qkv_kernel[grid](
        x,
        weight,
        conv_states,
        query_start_loc,
        cache_indices,
        cache_seqlens,
        q,
        k,
        v,
        head_num,
        seq_num if packed else 1,  # Only packed grids specialize on batch size.
        packed,
        head_dim,
        x.stride(0),
        stride_conv_state_seq,
        stride_conv_state_dim,
        stride_conv_state_tok,
        BLOCK_T=block_t,
        num_warps=num_warps,
    )
    return q, k, v
