# SPDX-License-Identifier: Apache-2.0

"""Causal depthwise convolution without external CUDA extensions.

Packed prefill reads strided [channels, total_tokens] projection views.
Outputs are allocated separately; convolution states are updated in place.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from lightllm.common.triton_utils.autotuner import AutotuneKernelType, autotune


def _prefill_configs():
    return [
        {"BLOCK_TOKENS": block_tokens, "BLOCK_CHANNELS": block_channels, "num_warps": num_warps}
        for block_tokens in (8, 16, 32, 64)
        for block_channels in (64, 128, 256)
        for num_warps in (2, 4, 8)
        if block_tokens * block_channels <= num_warps * 32 * 64
    ]


def _update_configs():
    return [
        {"BLOCK_CHANNELS": block_channels, "num_warps": num_warps}
        for block_channels in (64, 128, 256)
        for num_warps in (1, 2, 4)
    ]


def _conv_static_key(x, weight, bias, activation, conv_state):
    # Group by memory order so nearby token counts can reuse a configuration.
    if x.stride(-2) == 1:
        layout = "channel"
    elif x.stride(-1) == 1:
        layout = "token"
    else:
        layout = "strided"
    return {
        "channels": weight.shape[0],
        "width": weight.shape[1],
        "dtype": str(x.dtype),
        "layout": layout,
        "state_stride": f"{conv_state.stride(-2)}_{conv_state.stride(-1)}",
        "bias": bias is not None,
        "silu": activation is not None,
    }


def _prefill_static_key(x, weight, bias, conv_states, activation):
    return _conv_static_key(x, weight, bias, activation, conv_states)


def _prefill_run_key(x, query_start_loc, max_seqlen):
    batch_size = query_start_loc.numel() - 1
    max_query_len = x.shape[-1] if max_seqlen is None else max_seqlen
    return f"{batch_size}_{max_query_len}"


def _prefill_key_distance(run_key, config_key):
    batch_size, max_query_len = map(int, run_key.split("_"))
    config_batch_size, config_max_query_len = map(int, config_key.split("_"))
    return (
        abs(batch_size * max_query_len - config_batch_size * config_max_query_len),
        abs(batch_size - config_batch_size),
    )


def _update_static_key(x, conv_state, weight, bias, activation, cache_seqlens):
    input_3d = x.unsqueeze(-1) if x.ndim == 2 else x
    key = _conv_static_key(input_3d, weight, bias, activation, conv_state)
    key.update(
        tokens=input_3d.shape[-1],
        state_len=conv_state.shape[-1],
        circular=cache_seqlens is not None,
    )
    return key


@triton.jit
def _causal_conv1d_fwd_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    conv_states_ptr,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    out_ptr,
    num_channels,
    stride_x_channel,
    stride_x_token,
    stride_weight_channel,
    stride_weight_tap,
    stride_state_cache,
    stride_state_channel,
    stride_state_token,
    stride_out_channel,
    stride_out_token,
    CONV_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    HISTORY_LEN: tl.constexpr = CONV_WIDTH - 1
    tl.static_assert(CONV_WIDTH >= 2 and CONV_WIDTH <= 4, "Supported convolution widths are 2, 3 and 4")
    tl.static_assert(BLOCK_TOKENS >= HISTORY_LEN, "Only the first token block may read initial state")

    sequence_index = tl.program_id(0)
    token_block_index = tl.program_id(1)
    channel_indices = tl.program_id(2) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    valid_channels = channel_indices < num_channels
    sequence_start = tl.load(query_start_loc_ptr + sequence_index)
    sequence_length = tl.load(query_start_loc_ptr + sequence_index + 1) - sequence_start
    if token_block_index * BLOCK_TOKENS >= sequence_length:
        return

    token_indices = token_block_index * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid_output = (token_indices[:, None] < sequence_length) & valid_channels[None, :]
    input_base = x_ptr + sequence_start * stride_x_token + channel_indices * stride_x_channel
    output_base = out_ptr + sequence_start * stride_out_token + channel_indices * stride_out_channel
    output_ptrs = output_base[None, :] + token_indices[:, None] * stride_out_token
    cache_index = tl.load(cache_indices_ptr + sequence_index)
    if cache_index == PAD_SLOT_ID:
        # Padding requests retain their input, as in the CUDA implementation.
        padded_input = tl.load(input_base[None, :] + token_indices[:, None] * stride_x_token, valid_output, other=0)
        tl.store(output_ptrs, padded_input, valid_output)
        return

    state_base = conv_states_ptr + cache_index * stride_state_cache + channel_indices * stride_state_channel
    weight_base = weight_ptr + channel_indices * stride_weight_channel
    use_initial_state = tl.load(has_initial_state_ptr + sequence_index)
    accumulator = tl.full((BLOCK_TOKENS, BLOCK_CHANNELS), 0, tl.float32)
    if HAS_BIAS:
        bias_values = tl.load(bias_ptr + channel_indices, valid_channels, other=0).to(tl.float32)
        accumulator += bias_values[None, :]

    # Negative token positions come from cached history, or zero for a new request.
    for tap in tl.static_range(CONV_WIDTH):
        source_tokens = token_indices[:, None] + tap - HISTORY_LEN
        input_values = tl.load(
            input_base[None, :] + source_tokens * stride_x_token, valid_output & (source_tokens >= 0), other=0
        ).to(tl.float32)
        history_values = tl.load(
            state_base[None, :] + (source_tokens + HISTORY_LEN) * stride_state_token,
            valid_output & (source_tokens < 0) & use_initial_state,
            other=0,
        ).to(tl.float32)
        window_values = tl.where(source_tokens < 0, history_values, input_values)
        conv_weight = tl.load(weight_base + tap * stride_weight_tap, valid_channels, other=0).to(tl.float32)
        accumulator = tl.fma(window_values, conv_weight[None, :], accumulator)
    if APPLY_SILU:
        accumulator = accumulator / (1.0 + tl.exp(-accumulator))
    tl.store(output_ptrs, accumulator, valid_output)

    if token_block_index == 0:
        # Only the first token block reads/writes initial state. Later blocks
        # obtain their history from input, avoiding cross-CTA state races.
        state_offsets = tl.arange(0, triton.next_power_of_2(HISTORY_LEN))[:, None]
        tail_tokens = sequence_length - HISTORY_LEN + state_offsets
        valid_state = (state_offsets < HISTORY_LEN) & valid_channels[None, :]
        tail_inputs = tl.load(
            input_base[None, :] + tail_tokens * stride_x_token, valid_state & (tail_tokens >= 0), other=0
        )
        tail_history = tl.load(
            state_base[None, :] + (tail_tokens + HISTORY_LEN) * stride_state_token,
            valid_state & (tail_tokens < 0) & use_initial_state,
            other=0,
        )
        updated_state = tl.where(tail_tokens >= 0, tail_inputs, tail_history)
        # Finish all prefix reads before overwriting their state slots,
        # including requests shorter than the history window.
        tl.debug_barrier()
        tl.store(state_base[None, :] + state_offsets * stride_state_token, updated_state, valid_state)


@autotune(
    kernel_name="causal_conv1d_prefill:v1",
    configs_gen_func=_prefill_configs,
    static_key_func=_prefill_static_key,
    run_key_func=_prefill_run_key,
    run_key_distance_func=_prefill_key_distance,
    mutates_args=["conv_states"],
)
def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_states: torch.Tensor,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
    max_seqlen: Optional[int] = None,
    run_config: Optional[dict] = None,
    **kwargs,
):
    """Convolve packed [channels, total_tokens] inputs and update mapped states.

    Request boundaries, cache indices and initial-state flags are required,
    matching the GDN prefill caller. States have shape [slots, channels, width - 1]
    and may be strided MTP slices. Metadata tensors are contiguous.
    max_seqlen bounds the longest query; total tokens are a safe default that
    avoids reading lengths back from the GPU. Empty/padding requests keep state.
    """
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError(f"Unsupported activation: {activation}")
    num_channels, total_tokens = x.shape
    batch_size = query_start_loc.numel() - 1
    out = torch.empty_like(x)
    if batch_size == 0 or x.numel() == 0:
        return out

    bias = bias.contiguous() if bias is not None else None
    max_query_len = total_tokens if max_seqlen is None else max_seqlen
    if run_config is None:
        run_config = {"BLOCK_TOKENS": 16, "BLOCK_CHANNELS": 128, "num_warps": 4}
    block_tokens = run_config["BLOCK_TOKENS"]
    block_channels = run_config["BLOCK_CHANNELS"]
    grid = (batch_size, triton.cdiv(max_query_len, block_tokens), triton.cdiv(num_channels, block_channels))
    _causal_conv1d_fwd_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        out,
        num_channels,
        *x.stride(),
        *weight.stride(),
        *conv_states.stride(),
        *out.stride(),
        CONV_WIDTH=weight.shape[1],
        HAS_BIAS=bias is not None,
        APPLY_SILU=activation is not None,
        PAD_SLOT_ID=pad_slot_id,
        BLOCK_TOKENS=block_tokens,
        BLOCK_CHANNELS=block_channels,
        num_warps=run_config["num_warps"],
    )
    return out


@triton.jit
def _causal_conv1d_update_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,
    conv_state_indices_ptr,
    out_ptr,
    num_channels,
    stride_x_batch,
    stride_x_channel,
    stride_x_token,
    stride_weight_channel,
    stride_weight_tap,
    stride_state_cache,
    stride_state_channel,
    stride_state_token,
    stride_out_batch,
    stride_out_channel,
    stride_out_token,
    CONV_WIDTH: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CIRCULAR_BUFFER: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tl.static_assert(CONV_WIDTH >= 2 and CONV_WIDTH <= 4, "Supported convolution widths are 2, 3 and 4")
    tl.static_assert(STATE_LEN >= CONV_WIDTH - 1, "State must hold the full convolution history")

    batch_index = tl.program_id(0)
    channel_indices = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    valid_channels = channel_indices < num_channels
    input_base = x_ptr + batch_index * stride_x_batch + channel_indices * stride_x_channel
    output_base = out_ptr + batch_index * stride_out_batch + channel_indices * stride_out_channel
    cache_index = tl.load(conv_state_indices_ptr + batch_index) if HAS_INDICES else batch_index
    if cache_index == PAD_SLOT_ID:
        for token_index in range(NUM_TOKENS):
            padded_input = tl.load(input_base + token_index * stride_x_token, valid_channels, other=0)
            tl.store(output_base + token_index * stride_out_token, padded_input, valid_channels)
        return
    state_base = conv_state_ptr + cache_index * stride_state_cache + channel_indices * stride_state_channel
    weight_base = weight_ptr + channel_indices * stride_weight_channel
    state_cursor = tl.load(cache_seqlens_ptr + batch_index) % STATE_LEN if CIRCULAR_BUFFER else STATE_LEN
    history_start = state_cursor + STATE_LEN - (CONV_WIDTH - 1)

    # Keep the short convolution window and weights in registers across tokens.
    history_0 = tl.load(
        state_base + (history_start % STATE_LEN) * stride_state_token,
        valid_channels,
        other=0,
    ).to(tl.float32)
    weight_0 = tl.load(weight_base, valid_channels, other=0).to(tl.float32)
    weight_1 = tl.load(weight_base + stride_weight_tap, valid_channels, other=0).to(tl.float32)
    if CONV_WIDTH >= 3:
        history_1 = tl.load(
            state_base + ((history_start + 1) % STATE_LEN) * stride_state_token,
            valid_channels,
            other=0,
        ).to(tl.float32)
        weight_2 = tl.load(weight_base + 2 * stride_weight_tap, valid_channels, other=0).to(tl.float32)
    if CONV_WIDTH == 4:
        history_2 = tl.load(
            state_base + ((history_start + 2) % STATE_LEN) * stride_state_token, valid_channels, other=0
        ).to(tl.float32)
        weight_3 = tl.load(weight_base + 3 * stride_weight_tap, valid_channels, other=0).to(tl.float32)
    bias_values = tl.full((BLOCK_CHANNELS,), 0, tl.float32)
    if HAS_BIAS:
        bias_values = tl.load(bias_ptr + channel_indices, valid_channels, other=0).to(tl.float32)

    for token_index in range(NUM_TOKENS):
        input_values = tl.load(input_base + token_index * stride_x_token, valid_channels, other=0).to(tl.float32)
        accumulator = tl.fma(weight_0, history_0, bias_values)
        if CONV_WIDTH == 2:
            accumulator = tl.fma(weight_1, input_values, accumulator)
            history_0 = input_values
        elif CONV_WIDTH == 3:
            accumulator = tl.fma(weight_1, history_1, accumulator)
            accumulator = tl.fma(weight_2, input_values, accumulator)
            history_0, history_1 = history_1, input_values
        else:
            accumulator = tl.fma(weight_1, history_1, accumulator)
            accumulator = tl.fma(weight_2, history_2, accumulator)
            accumulator = tl.fma(weight_3, input_values, accumulator)
            history_0, history_1, history_2 = history_1, history_2, input_values
        if APPLY_SILU:
            accumulator = accumulator / (1.0 + tl.exp(-accumulator))
        tl.store(output_base + token_index * stride_out_token, accumulator, valid_channels)
        if CIRCULAR_BUFFER:
            tl.store(
                state_base + ((state_cursor + token_index) % STATE_LEN) * stride_state_token,
                input_values,
                valid_channels,
            )
    if not CIRCULAR_BUFFER:
        # Keep the last STATE_LEN tokens of [old state, new input].
        state_offsets = tl.arange(0, triton.next_power_of_2(STATE_LEN))[:, None]
        valid_state = (state_offsets < STATE_LEN) & valid_channels[None, :]
        source_offsets = state_offsets + NUM_TOKENS
        old_history = tl.load(
            state_base[None, :] + source_offsets * stride_state_token,
            valid_state & (source_offsets < STATE_LEN),
            other=0,
        )
        new_inputs = tl.load(
            input_base[None, :] + (source_offsets - STATE_LEN) * stride_x_token,
            valid_state & (source_offsets >= STATE_LEN),
            other=0,
        )
        updated_state = tl.where(source_offsets < STATE_LEN, old_history, new_inputs)
        tl.debug_barrier()
        tl.store(state_base[None, :] + state_offsets * stride_state_token, updated_state, valid_state)


@autotune(
    kernel_name="causal_conv1d_update:v1",
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    configs_gen_func=_update_configs,
    static_key_func=_update_static_key,
    run_key_func=lambda x: x.shape[0],
    mutates_args=["conv_state"],
)
def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    run_config: Optional[dict] = None,
):
    """Decode [batch, channels] or [batch, channels, tokens], updating linear/ring states.

    State shape is [slots, channels, state_len], with state_len >= width - 1.
    Cache indices and sequence lengths, when supplied, are contiguous batch vectors.
    """
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError(f"Unsupported activation: {activation}")
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out

    input_3d = x.unsqueeze(-1) if x.ndim == 2 else x
    output_3d = out.unsqueeze(-1) if x.ndim == 2 else out
    batch_size, num_channels, num_tokens = input_3d.shape
    bias = bias.contiguous() if bias is not None else None
    if run_config is None:
        run_config = {"BLOCK_CHANNELS": 128, "num_warps": 4}
    block_channels = run_config["BLOCK_CHANNELS"]
    grid = (batch_size, triton.cdiv(num_channels, block_channels))
    _causal_conv1d_update_kernel[grid](
        input_3d,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        output_3d,
        num_channels,
        *input_3d.stride(),
        *weight.stride(),
        *conv_state.stride(),
        *output_3d.stride(),
        CONV_WIDTH=weight.shape[1],
        NUM_TOKENS=num_tokens,
        STATE_LEN=conv_state.shape[2],
        HAS_INDICES=conv_state_indices is not None,
        HAS_BIAS=bias is not None,
        CIRCULAR_BUFFER=cache_seqlens is not None,
        APPLY_SILU=activation is not None,
        PAD_SLOT_ID=pad_slot_id,
        BLOCK_CHANNELS=block_channels,
        num_warps=run_config["num_warps"],
    )
    return out
