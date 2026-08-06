# Vendored from vLLM v0.14.1
#   source: vllm/model_executor/layers/mamba/ops/causal_conv1d.py
#   commit: d7de043d55d1dd629554467e23874097e1c48993
# Adapted for LightLLM:
#   - imports point at standard triton instead of vLLM's triton-lite.
#   - vLLM block-table params (block_idx_last_scheduled_token, initial_state_idx,
#     null_block_id) are dropped; LightLLM uses contiguous per-request slots.
#   - IS_VARLEN / IS_SPEC_DECODING / non-spec paths removed; this kernel now
#     exclusively serves the spec-decode varlen path (with num_accepted_tokens,
#     query_start_loc and mtp_step all required).
#   - One widened conv_state slot per request holds K-1+mtp_step positions.
#     The read offset is num_accepted_tokens-1; writes go back to the same slot.
#
# Upstream copyright notice:
#   SPDX-License-Identifier: Apache-2.0
#   SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#   Copyright (c) 2024, Tri Dao.
#   Adapted from
#   https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (num_tokens, dim)
    w_ptr,  # (dim, width)
    bias_ptr,  # (dim,) or nullptr
    conv_state_ptr,  # (num_slots, dim, state_len)
    conv_state_indices_ptr,  # (batch,)
    num_accepted_tokens_ptr,  # (batch,)
    query_start_loc_ptr,  # (batch + 1,)
    o_ptr,  # (num_tokens, dim) — overwrites x in-place
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    state_len: tl.constexpr,  # width - 1 + mtp_step
    # Strides
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # LightLLM uses contiguous per-request slots; read and write both target
    # conv_state_indices[idx_seq].
    conv_state_init = 0

    # cache_idx
    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init).to(
        tl.int64
    )

    if conv_states_input_coord == pad_slot_id:
        # padded entry — nothing to do
        return

    query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
    query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
    seqlen = query_end_index - query_start_index

    if query_start_index == query_end_index:
        return

    # The rolling of conv state:
    #
    # Before forward, the conv_state is:
    # [history1, history2, ..., historyM].
    #
    # After forward, the conv_state becomes:
    # [history2, ..., historyM, draft1, draft2, ..., draftN].
    #
    # After acceptance, it becomes:
    #
    # - accept 1 tokens: [history2, ..., historyM, draft1]
    # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
    # - and so on.
    conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
    mask_w = idx_feats < dim

    # STEP 1: load initial history columns from conv_state
    #   col_k = conv_state[slot, :, offset + k]  for k = 0..KERNEL_WIDTH-2
    conv_states_base = (
        conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
    )

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok  # [BLOCK_N]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: update conv_state with a sliding window
    #
    # Preserve KERNEL_WIDTH-2 tokens starting from offset+1, then append
    # the seqlen incoming x tokens.  The resulting state is written back
    # to positions 0..state_len-1 of the same slot.
    #
    # For KERNEL_WIDTH=2, restore_conv_state_len = 0 so the mask is
    # always false — the state is fully overwritten by loaded_x.
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # read from conv_state at (offset + 1 + idx_tokens); the +1 accounts
    # for the fact that the next call will slide offset by num_accepted.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((conv_state_token_offset + idx_tokens + 1) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    # preserve KERNEL_WIDTH-2 history tokens from the old state
    restore_conv_state_len = KERNEL_WIDTH - 1 - 1
    mask = (idx_tokens < restore_conv_state_len)[:, None] & (idx_feats < dim)[None, :]
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
    x_base = x_ptr + query_start_index * stride_x_token + (idx_feats * stride_x_dim)  # [BLOCK_N]

    # move_idx_tokens = idx_tokens - restore_conv_state_len offsets the
    # incoming x tokens so they fill positions after the preserved history
    # inside new_conv_state via tl.where below.
    move_idx_tokens = idx_tokens - restore_conv_state_len
    x_ptrs = x_base[None, :] + (move_idx_tokens * stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (move_idx_tokens >= 0)[:, None] & (move_idx_tokens < seqlen)[:, None] & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    # Write the updated state back to the same slot that was read.
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)  # slot offset
        + (idx_feats * stride_conv_state_dim)[None, :]  # dim offset
        + idx_tokens[:, None] * stride_conv_state_tok  # token offset
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)  # [BLOCK_N] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)  # [BLOCK_N] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = idx_feats < dim  # feature-index
        o_ptrs = o_ptr + query_start_index * stride_o_token + idx_token * stride_o_token + (idx_feats * stride_o_dim)

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    mtp_step: int,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
):
    """Spec-decode causal depthwise conv1d update.

    Processes ``mtp_step + 1`` tokens per request in varlen layout.
    Uses a single widened conv_state slot per request that holds
    ``width - 1 + mtp_step`` positions.  The read offset for each request
    is ``num_accepted_tokens[b] - 1``; after the forward pass the updated
    state is written back to the same slot, ready for the next decode step.

    Args:
        x: ``(num_tokens, dim)`` float — flattened varlen input grouped by
            ``query_start_loc``.  Each request contributes ``mtp_step + 1``
            tokens.
        conv_state: ``(num_slots, dim, state_len)`` float with
            ``state_len == width - 1 + mtp_step``.
        weight: depthwise filter of shape ``(dim, width)``.
        mtp_step: number of speculative (draft) tokens per request
            (``seqlen == mtp_step + 1``).
        bias: optional ``(dim,)`` float bias.
        activation: ``None``, ``"silu"`` or ``"swish"``.
        conv_state_indices: ``(batch,)`` int32 — maps each request to a
            conv_state slot.
        num_accepted_tokens: ``(batch,)`` int32 — the conv_state read offset
            for each request is ``num_accepted_tokens[b] - 1``.
        query_start_loc: ``(batch + 1,)`` int32 — cumulative token offsets
            for the varlen x tensor.
        pad_slot_id: int — slot id that marks padded (skipped) entries.

    Returns:
        Output tensor with the same shape as ``x`` (the kernel overwrites
        ``x`` in place), one conv output per input token.
    """
    if activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    # x shape is (num_tokens, dim)
    assert conv_state_indices is not None
    batch = conv_state_indices.size(0)  # number of requests
    dim = x.size(1)
    _, width = weight.shape
    # conv_state: (num_slots, dim, state_len) with state_len == width - 1 + mtp_step
    _, _, state_len = conv_state.size()

    assert state_len == width - 1 + mtp_step

    # adopt the strategy in vLLM that overwrites 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    # X (num_tokens, dim)
    stride_x_token, stride_x_dim = x.stride()
    stride_o_token, stride_o_dim = out.stride()

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0)

    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        out,
        # Matrix dimensions
        batch,
        dim,
        state_len,
        # stride
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        NP2_STATELEN=np2_statelen,
        BLOCK_N=256,
    )

    return out.to(original_x_dtype)
