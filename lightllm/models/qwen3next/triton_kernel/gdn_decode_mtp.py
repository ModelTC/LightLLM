"""
Optimized GDN Decode MTP (Multi-Token Prediction) Kernel

This module provides an optimized Triton kernel for GDN decode with MTP support,
eliminating the need for sequential Python loops and reducing memory operations.

Key optimizations:
1. Fused data reorganization from interleaved to batched layout
2. Parallel processing of all batch items with proper state indexing
3. Auto-tuned configurations for different batch sizes and model dimensions
"""

import torch
import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _reorganize_mtp_data_kernel(
    # Input pointers (interleaved layout: [step0_batch0, step0_batch1, ..., step1_batch0, ...])
    src_ptr,
    # Output pointers (batched layout: [batch0_step0, batch0_step1, ..., batch1_step0, ...])
    dst_ptr,
    # Dimensions
    batch_size,
    mtp_size,
    dim_size,
    # Strides
    src_stride_token,
    src_stride_dim,
    dst_stride_token,
    dst_stride_dim,
    # Block sizes
    BLOCK_DIM: tl.constexpr,
):
    """
    Reorganize data from interleaved MTP layout to batched layout.

    Input layout:  [step0_batch0, step0_batch1, ..., step0_batchN, step1_batch0, ...]
    Output layout: [batch0_step0, batch0_step1, ..., batch0_stepM, batch1_step0, ...]

    This enables efficient processing with the recurrent kernel.
    """
    batch_idx = tl.program_id(0)
    step_idx = tl.program_id(1)
    block_dim_idx = tl.program_id(2)

    # Calculate source and destination token indices
    src_token_idx = step_idx * batch_size + batch_idx
    dst_token_idx = batch_idx * mtp_size + step_idx

    # Calculate dimension offsets
    dim_start = block_dim_idx * BLOCK_DIM
    dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
    mask = dim_offsets < dim_size

    # Load from source (interleaved layout)
    src_offset = src_token_idx * src_stride_token + dim_offsets * src_stride_dim
    data = tl.load(src_ptr + src_offset, mask=mask, other=0.0)

    # Store to destination (batched layout)
    dst_offset = dst_token_idx * dst_stride_token + dim_offsets * dst_stride_dim
    tl.store(dst_ptr + dst_offset, data, mask=mask)


@triton.jit
def _reorganize_mtp_data_back_kernel(
    # Input pointers (batched layout): [batch_size, mtp_size, num_heads, head_dim]
    src_ptr,
    # Output pointers (interleaved layout): [total_tokens, 1, num_heads, head_dim]
    dst_ptr,
    # Dimensions
    batch_size,
    mtp_size,
    num_heads,
    head_dim,
    # Strides for src: [batch_size, mtp_size, num_heads, head_dim]
    src_stride_batch,
    src_stride_mtp,
    src_stride_head,
    src_stride_dim,
    # Strides for dst: [total_tokens, 1, num_heads, head_dim]
    dst_stride_token,
    dst_stride_seq,
    dst_stride_head,
    dst_stride_dim,
    # Block sizes
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Reorganize output data from batched layout back to interleaved layout.

    Input shape:  [batch_size, mtp_size, num_heads, head_dim]
    Output shape: [batch_size * mtp_size, 1, num_heads, head_dim] (interleaved)

    Mapping: src[b, s, h, d] -> dst[s * batch_size + b, 0, h, d]
    """
    batch_idx = tl.program_id(0)
    step_idx = tl.program_id(1)
    block_idx = tl.program_id(2)

    # Decompose block_idx into head and dim blocks
    num_dim_blocks = tl.cdiv(head_dim, BLOCK_DIM)
    block_head_idx = block_idx // num_dim_blocks
    block_dim_idx = block_idx % num_dim_blocks

    # Calculate destination token index (interleaved)
    dst_token_idx = step_idx * batch_size + batch_idx

    # Calculate offsets
    head_start = block_head_idx * BLOCK_HEAD
    dim_start = block_dim_idx * BLOCK_DIM

    head_offsets = head_start + tl.arange(0, BLOCK_HEAD)
    dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)

    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    mask = head_mask[:, None] & dim_mask[None, :]

    # Load from source (batched layout): [batch_size, mtp_size, num_heads, head_dim]
    src_base = src_ptr + batch_idx * src_stride_batch + step_idx * src_stride_mtp
    src_offset = head_offsets[:, None] * src_stride_head + dim_offsets[None, :] * src_stride_dim
    data = tl.load(src_base + src_offset, mask=mask, other=0.0)

    # Store to destination (interleaved layout): [total_tokens, 1, num_heads, head_dim]
    # The seq dimension (1) is skipped since it's always 0
    dst_base = dst_ptr + dst_token_idx * dst_stride_token
    dst_offset = head_offsets[:, None] * dst_stride_head + dim_offsets[None, :] * dst_stride_dim
    tl.store(dst_base + dst_offset, data, mask=mask)


def _get_reorganize_mtp_configs():
    """Generate candidate configurations for MTP data reorganization."""
    configs = []
    for block_dim in [64, 128, 256, 512]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                configs.append(
                    {
                        "BLOCK_DIM": block_dim,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                )
    return configs


def _get_reorganize_static_key(src: torch.Tensor, mtp_size: int):
    """Static key based on tensor properties."""
    return {
        "dtype": str(src.dtype),
        "mtp_size": mtp_size,
    }


def _get_reorganize_run_key(src: torch.Tensor, mtp_size: int):
    """Run key based on batch size and dimension."""
    total_tokens = src.shape[0]
    batch_size = total_tokens // mtp_size
    dim_size = src.shape[-1]
    return f"{batch_size}_{dim_size}"


@autotune(
    kernel_name="gdn_decode_mtp_reorganize:v1",
    configs_gen_func=_get_reorganize_mtp_configs,
    static_key_func=_get_reorganize_static_key,
    run_key_func=_get_reorganize_run_key,
    mutates_args=["dst"],
)
def reorganize_mtp_to_batched(
    src: torch.Tensor,
    dst: torch.Tensor,
    mtp_size: int,
    run_config: dict = None,
):
    """
    Reorganize data from interleaved MTP layout to batched layout.

    Args:
        src: Input tensor with interleaved layout [total_tokens, dim]
               Layout: [step0_batch0, step0_batch1, ..., step1_batch0, ...]
        dst: Output tensor with batched layout [total_tokens, dim]
               Layout: [batch0_step0, batch0_step1, ..., batch1_step0, ...]
        mtp_size: Number of MTP steps
        run_config: Auto-tuned configuration
    """
    total_tokens = src.shape[0]
    batch_size = total_tokens // mtp_size
    dim_size = src.shape[-1]

    if run_config is None:
        BLOCK_DIM = triton.next_power_of_2(min(dim_size, 256))
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_DIM = run_config["BLOCK_DIM"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_dim = triton.cdiv(dim_size, BLOCK_DIM)

    grid = (batch_size, mtp_size, num_blocks_dim)

    _reorganize_mtp_data_kernel[grid](
        src,
        dst,
        batch_size,
        mtp_size,
        dim_size,
        src.stride(0),
        src.stride(-1) if src.ndim > 1 else 1,
        dst.stride(0),
        dst.stride(-1) if dst.ndim > 1 else 1,
        BLOCK_DIM=BLOCK_DIM,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _get_reorganize_back_configs():
    """Generate candidate configurations for MTP output reorganization."""
    configs = []
    for block_head in [4, 8, 16, 32]:
        for block_dim in [32, 64, 128]:
            for num_warps in [2, 4, 8]:
                for num_stages in [2, 3]:
                    if block_head * block_dim <= 4096:  # Limit shared memory
                        configs.append(
                            {
                                "BLOCK_HEAD": block_head,
                                "BLOCK_DIM": block_dim,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )
    return configs


def _get_reorganize_back_static_key(
    src: torch.Tensor,
    batch_size: int,
    mtp_size: int,
    num_heads: int,
    head_dim: int,
):
    """Static key for output reorganization."""
    return {
        "dtype": str(src.dtype),
        "mtp_size": mtp_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


def _get_reorganize_back_run_key(
    src: torch.Tensor,
    batch_size: int,
    mtp_size: int,
    num_heads: int,
    head_dim: int,
):
    """Run key for output reorganization."""
    return batch_size


@autotune(
    kernel_name="gdn_decode_mtp_reorganize_back:v1",
    configs_gen_func=_get_reorganize_back_configs,
    static_key_func=_get_reorganize_back_static_key,
    run_key_func=_get_reorganize_back_run_key,
    mutates_args=["dst"],
)
def reorganize_mtp_output_to_interleaved(
    src: torch.Tensor,
    dst: torch.Tensor,
    batch_size: int,
    mtp_size: int,
    num_heads: int,
    head_dim: int,
    run_config: dict = None,
):
    """
    Reorganize output from batched layout back to interleaved layout.

    Args:
        src: Input tensor [batch_size, mtp_size, num_heads, head_dim] (4D)
        dst: Output tensor [batch_size * mtp_size, 1, num_heads, head_dim] (4D)
        batch_size: Number of batch items
        mtp_size: Number of MTP steps
        num_heads: Number of attention heads
        head_dim: Head dimension
        run_config: Auto-tuned configuration

    Mapping: src[b, s, h, d] -> dst[s * batch_size + b, 0, h, d]
    """
    if run_config is None:
        BLOCK_HEAD = min(triton.next_power_of_2(num_heads), 16)
        BLOCK_DIM = min(triton.next_power_of_2(head_dim), 64)
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_HEAD = run_config["BLOCK_HEAD"]
        BLOCK_DIM = run_config["BLOCK_DIM"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_head_blocks = triton.cdiv(num_heads, BLOCK_HEAD)
    num_dim_blocks = triton.cdiv(head_dim, BLOCK_DIM)
    num_blocks_total = num_head_blocks * num_dim_blocks

    grid = (batch_size, mtp_size, num_blocks_total)

    # src is 4D: [batch_size, mtp_size, num_heads, head_dim]
    # dst is 4D: [total_tokens, 1, num_heads, head_dim]
    _reorganize_mtp_data_back_kernel[grid](
        src,
        dst,
        batch_size,
        mtp_size,
        num_heads,
        head_dim,
        src.stride(0),  # batch stride
        src.stride(1),  # mtp stride
        src.stride(2),  # head stride
        src.stride(3),  # dim stride
        dst.stride(0),  # token stride
        dst.stride(1),  # seq stride (=1)
        dst.stride(2),  # head stride
        dst.stride(3),  # dim stride
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DIM=BLOCK_DIM,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _prepare_mtp_indices_kernel(
    # Input indices (per-step buffer indices)
    buffer_idx_ptr,
    # Output 2D indices for recurrent kernel
    output_idx_ptr,
    # Dimensions
    batch_size,
    mtp_size,
    # Strides
    input_stride,
    output_stride_batch,
    output_stride_step,
):
    """
    Prepare 2D indices for the fused recurrent kernel.

    Input: mtp_size tensors of shape [batch_size] (buffer indices for each step)
    Output: 2D tensor [batch_size, mtp_size] for ssm_state_indices
    """
    batch_idx = tl.program_id(0)
    step_idx = tl.program_id(1)

    # Load the buffer index for this batch and step
    buffer_idx = tl.load(buffer_idx_ptr + step_idx * input_stride + batch_idx)

    # Store to the 2D output
    output_offset = batch_idx * output_stride_batch + step_idx * output_stride_step
    tl.store(output_idx_ptr + output_offset, buffer_idx)


def prepare_mtp_state_indices(
    mtp_buffer_idx_list: list,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Prepare 2D state indices for the fused recurrent kernel.

    Args:
        mtp_buffer_idx_list: List of buffer index tensors, one per MTP step
        batch_size: Number of batch items
        device: Target device

    Returns:
        2D tensor of shape [batch_size, mtp_size] for ssm_state_indices
    """

    # Stack indices to create [mtp_size, batch_size] tensor
    stacked_indices = torch.stack(mtp_buffer_idx_list, dim=0)

    # Transpose to get [batch_size, mtp_size]
    return stacked_indices.T.contiguous()


@triton.jit
def _fused_conv1d_mtp_step_kernel(
    # Input/output data
    mixed_qkv_ptr,
    # Conv state buffer
    conv_states_ptr,
    # Conv weight and bias
    conv_weight_ptr,
    conv_bias_ptr,
    # Buffer indices (one per MTP step, each [batch_size])
    buffer_indices_ptr,
    next_buffer_indices_ptr,
    # Dimensions
    batch_size,
    dim_size,
    conv_width,
    # Step info
    step_idx,
    mtp_size,
    is_last_step: tl.constexpr,
    # Strides
    qkv_stride_token,
    qkv_stride_dim,
    state_stride_buffer,
    state_stride_dim,
    state_stride_width,
    weight_stride_dim,
    weight_stride_width,
    # Block sizes
    BLOCK_DIM: tl.constexpr,
    ACTIVATION_SILU: tl.constexpr,
):
    """
    Fused kernel for conv1d update in MTP decode.

    Handles one MTP step for all batch items:
    1. Reads current conv state
    2. Updates with new input
    3. Computes conv1d output
    4. Optionally copies state to next MTP step
    """
    batch_idx = tl.program_id(0)
    block_dim_idx = tl.program_id(1)

    # Calculate token index in interleaved layout
    token_idx = step_idx * batch_size + batch_idx

    # Load buffer indices
    cur_buffer_idx = tl.load(buffer_indices_ptr + batch_idx).to(tl.int64)

    # Calculate dimension offsets
    dim_start = block_dim_idx * BLOCK_DIM
    dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < dim_size

    # Load input value
    input_offset = token_idx * qkv_stride_token + dim_offsets * qkv_stride_dim
    input_val = tl.load(mixed_qkv_ptr + input_offset, mask=dim_mask, other=0.0)

    # Load conv bias
    bias_val = tl.load(conv_bias_ptr + dim_offsets, mask=dim_mask, other=0.0)

    # Compute conv1d output and update state
    output_val = bias_val
    state_base = conv_states_ptr + cur_buffer_idx * state_stride_buffer

    # Process each position in the conv window
    for w in range(conv_width):
        # Load weight for this position
        weight_offset = dim_offsets * weight_stride_dim + w * weight_stride_width
        weight_val = tl.load(conv_weight_ptr + weight_offset, mask=dim_mask, other=0.0)

        if w < conv_width - 1:
            # Load from state buffer
            state_offset = dim_offsets * state_stride_dim + w * state_stride_width
            state_val = tl.load(state_base + state_offset, mask=dim_mask, other=0.0)
            output_val += state_val * weight_val
        else:
            # Use current input for the last position
            output_val += input_val * weight_val

    # Update conv state (shift and insert new value)
    for w in range(conv_width - 2, -1, -1):
        if w == conv_width - 2:
            # Insert new input at the end
            state_offset = dim_offsets * state_stride_dim + w * state_stride_width
            tl.store(state_base + state_offset, input_val, mask=dim_mask)
        else:
            # Shift state
            src_offset = dim_offsets * state_stride_dim + (w + 1) * state_stride_width
            dst_offset = dim_offsets * state_stride_dim + w * state_stride_width
            val = tl.load(state_base + src_offset, mask=dim_mask, other=0.0)
            tl.store(state_base + dst_offset, val, mask=dim_mask)

    # Apply activation (SiLU)
    if ACTIVATION_SILU:
        output_val = output_val * tl.sigmoid(output_val)

    # Store output
    tl.store(mixed_qkv_ptr + input_offset, output_val, mask=dim_mask)

    # Copy state to next step if not last
    if not is_last_step:
        next_buffer_idx = tl.load(next_buffer_indices_ptr + batch_idx).to(tl.int64)
        next_state_base = conv_states_ptr + next_buffer_idx * state_stride_buffer

        for w in range(conv_width - 1):
            state_offset = dim_offsets * state_stride_dim + w * state_stride_width
            val = tl.load(state_base + state_offset, mask=dim_mask, other=0.0)
            tl.store(next_state_base + state_offset, val, mask=dim_mask)


def _get_conv1d_mtp_configs():
    """Generate candidate configurations for conv1d MTP kernel."""
    configs = []
    for block_dim in [64, 128, 256, 512]:
        for num_warps in [2, 4, 8]:
            for num_stages in [1, 2, 3]:
                configs.append(
                    {
                        "BLOCK_DIM": block_dim,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                )
    return configs


def _get_conv1d_mtp_static_key(
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weight: torch.Tensor,
    mtp_size: int,
):
    """Static key for conv1d MTP kernel."""
    return {
        "dtype": str(mixed_qkv.dtype),
        "dim_size": mixed_qkv.shape[-1],
        "conv_width": conv_weight.shape[-1],
        "mtp_size": mtp_size,
    }


def _get_conv1d_mtp_run_key(
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weight: torch.Tensor,
    mtp_size: int,
):
    """Run key for conv1d MTP kernel."""
    total_tokens = mixed_qkv.shape[0]
    batch_size = total_tokens // mtp_size
    return batch_size


@autotune(
    kernel_name="gdn_conv1d_mtp:v1",
    configs_gen_func=_get_conv1d_mtp_configs,
    static_key_func=_get_conv1d_mtp_static_key,
    run_key_func=_get_conv1d_mtp_run_key,
    mutates_args=["mixed_qkv", "conv_states"],
)
def fused_conv1d_mtp_update(
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    mtp_buffer_idx_list: list,
    mtp_size: int,
    activation_silu: bool = True,
    run_config: dict = None,
):
    """
    Fused conv1d update for all MTP steps.

    Args:
        mixed_qkv: Input tensor [batch_size * mtp_size, dim] (interleaved)
        conv_states: Conv state buffer [num_buffers, dim, conv_width-1]
        conv_weight: Conv weights [dim, conv_width]
        conv_bias: Conv bias [dim]
        mtp_buffer_idx_list: List of buffer index tensors per step
        mtp_size: Number of MTP steps
        activation_silu: Whether to apply SiLU activation
        run_config: Auto-tuned configuration
    """
    total_tokens = mixed_qkv.shape[0]
    batch_size = total_tokens // mtp_size
    dim_size = mixed_qkv.shape[-1]
    conv_width = conv_weight.shape[-1]

    if run_config is None:
        BLOCK_DIM = triton.next_power_of_2(min(dim_size, 256))
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_DIM = run_config["BLOCK_DIM"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_dim = triton.cdiv(dim_size, BLOCK_DIM)

    for step_idx in range(mtp_size):
        is_last_step = step_idx == mtp_size - 1
        cur_indices = mtp_buffer_idx_list[step_idx]
        next_indices = mtp_buffer_idx_list[step_idx + 1] if not is_last_step else cur_indices

        grid = (batch_size, num_blocks_dim)

        _fused_conv1d_mtp_step_kernel[grid](
            mixed_qkv,
            conv_states,
            conv_weight,
            conv_bias,
            cur_indices,
            next_indices,
            batch_size,
            dim_size,
            conv_width,
            step_idx,
            mtp_size,
            is_last_step,
            mixed_qkv.stride(0),
            mixed_qkv.stride(-1) if mixed_qkv.ndim > 1 else 1,
            conv_states.stride(0),
            conv_states.stride(1),
            conv_states.stride(2),
            conv_weight.stride(0),
            conv_weight.stride(1),
            BLOCK_DIM=BLOCK_DIM,
            ACTIVATION_SILU=activation_silu,
            num_warps=num_warps,
            num_stages=num_stages,
        )


@triton.jit
def _copy_ssm_state_kernel(
    # SSM state buffer
    ssm_states_ptr,
    # Buffer indices
    src_indices_ptr,
    dst_indices_ptr,
    # Dimensions
    batch_size,
    num_heads,
    key_dim,
    value_dim,
    # Strides
    state_stride_buffer,
    state_stride_head,
    state_stride_key,
    state_stride_value,
    # Block sizes
    BLOCK_KEY: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
):
    """
    Copy SSM states from source indices to destination indices.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_idx = tl.program_id(2)

    # Calculate block positions
    num_value_blocks = tl.cdiv(value_dim, BLOCK_VALUE)
    block_key_idx = block_idx // num_value_blocks
    block_value_idx = block_idx % num_value_blocks

    key_start = block_key_idx * BLOCK_KEY
    value_start = block_value_idx * BLOCK_VALUE

    key_offsets = key_start + tl.arange(0, BLOCK_KEY)
    value_offsets = value_start + tl.arange(0, BLOCK_VALUE)

    key_mask = key_offsets < key_dim
    value_mask = value_offsets < value_dim
    mask = key_mask[:, None] & value_mask[None, :]

    # Load indices
    src_idx = tl.load(src_indices_ptr + batch_idx).to(tl.int64)
    dst_idx = tl.load(dst_indices_ptr + batch_idx).to(tl.int64)

    # Calculate offsets
    src_base = ssm_states_ptr + src_idx * state_stride_buffer + head_idx * state_stride_head
    dst_base = ssm_states_ptr + dst_idx * state_stride_buffer + head_idx * state_stride_head

    offsets = key_offsets[:, None] * state_stride_key + value_offsets[None, :] * state_stride_value

    # Copy data
    data = tl.load(src_base + offsets, mask=mask, other=0.0)
    tl.store(dst_base + offsets, data, mask=mask)


@triton.jit
def _copy_conv_state_kernel(
    # Conv state buffer [num_buffers, dim, conv_width-1]
    conv_states_ptr,
    # Buffer indices
    src_indices_ptr,
    dst_indices_ptr,
    # Dimensions
    batch_size,
    dim_size,
    width_size,
    num_width_blocks,  # Precomputed to avoid runtime division
    # Strides
    state_stride_buffer,
    state_stride_dim,
    state_stride_width,
    # Block sizes
    BLOCK_DIM: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    """
    Copy conv states from source indices to destination indices.

    Conv state shape: [num_buffers, dim, conv_width-1]
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Calculate block positions using precomputed num_width_blocks
    block_dim_idx = block_idx // num_width_blocks
    block_width_idx = block_idx % num_width_blocks

    dim_start = block_dim_idx * BLOCK_DIM
    width_start = block_width_idx * BLOCK_WIDTH

    dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
    width_offsets = width_start + tl.arange(0, BLOCK_WIDTH)

    dim_mask = dim_offsets < dim_size
    width_mask = width_offsets < width_size
    mask = dim_mask[:, None] & width_mask[None, :]

    # Load indices
    src_idx = tl.load(src_indices_ptr + batch_idx).to(tl.int64)
    dst_idx = tl.load(dst_indices_ptr + batch_idx).to(tl.int64)

    # Calculate offsets
    src_base = conv_states_ptr + src_idx * state_stride_buffer
    dst_base = conv_states_ptr + dst_idx * state_stride_buffer

    offsets = dim_offsets[:, None] * state_stride_dim + width_offsets[None, :] * state_stride_width

    # Copy data
    data = tl.load(src_base + offsets, mask=mask, other=0.0)
    tl.store(dst_base + offsets, data, mask=mask)


def _get_conv_copy_configs():
    """Generate candidate configurations for conv state copy."""
    configs = []
    for block_dim in [64, 128, 256]:
        for block_width in [2, 4, 8]:
            for num_warps in [2, 4]:
                configs.append(
                    {
                        "BLOCK_DIM": block_dim,
                        "BLOCK_WIDTH": block_width,
                        "num_warps": num_warps,
                        "num_stages": 2,
                    }
                )
    return configs


def _get_conv_copy_static_key(
    conv_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Static key for conv copy."""
    return {
        "dtype": str(conv_states.dtype),
        "dim_size": conv_states.shape[1],
        "width_size": conv_states.shape[2],
    }


def _get_conv_copy_run_key(
    conv_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Run key for conv copy."""
    return src_indices.shape[0]


@autotune(
    kernel_name="gdn_conv_state_copy:v1",
    configs_gen_func=_get_conv_copy_configs,
    static_key_func=_get_conv_copy_static_key,
    run_key_func=_get_conv_copy_run_key,
    mutates_args=["conv_states"],
)
def copy_conv_states(
    conv_states: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    run_config: dict = None,
):
    """
    Copy conv states from source indices to destination indices.

    Args:
        conv_states: Conv state buffer [num_buffers, dim, conv_width-1]
        src_indices: Source buffer indices [batch_size]
        dst_indices: Destination buffer indices [batch_size]
        run_config: Auto-tuned configuration
    """
    batch_size = src_indices.shape[0]
    dim_size = conv_states.shape[1]
    width_size = conv_states.shape[2]

    if run_config is None:
        BLOCK_DIM = triton.next_power_of_2(min(dim_size, 128))
        BLOCK_WIDTH = triton.next_power_of_2(min(width_size, 4))
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_DIM = run_config["BLOCK_DIM"]
        BLOCK_WIDTH = run_config["BLOCK_WIDTH"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_dim_blocks = triton.cdiv(dim_size, BLOCK_DIM)
    num_width_blocks = triton.cdiv(width_size, BLOCK_WIDTH)
    num_blocks_total = num_dim_blocks * num_width_blocks

    grid = (batch_size, num_blocks_total)

    _copy_conv_state_kernel[grid](
        conv_states,
        src_indices,
        dst_indices,
        batch_size,
        dim_size,
        width_size,
        num_width_blocks,  # Pass precomputed value
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_WIDTH=BLOCK_WIDTH,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _get_ssm_copy_configs():
    """Generate candidate configurations for SSM state copy."""
    configs = []
    for block_key in [16, 32, 64]:
        for block_value in [16, 32, 64, 128]:
            for num_warps in [2, 4, 8]:
                if block_key * block_value <= 4096:
                    configs.append(
                        {
                            "BLOCK_KEY": block_key,
                            "BLOCK_VALUE": block_value,
                            "num_warps": num_warps,
                            "num_stages": 2,
                        }
                    )
    return configs


def _get_ssm_copy_static_key(
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Static key for SSM copy."""
    return {
        "dtype": str(ssm_states.dtype),
        "num_heads": ssm_states.shape[1],
        "key_dim": ssm_states.shape[2],
        "value_dim": ssm_states.shape[3],
    }


def _get_ssm_copy_run_key(
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Run key for SSM copy."""
    return src_indices.shape[0]


@autotune(
    kernel_name="gdn_ssm_state_copy:v1",
    configs_gen_func=_get_ssm_copy_configs,
    static_key_func=_get_ssm_copy_static_key,
    run_key_func=_get_ssm_copy_run_key,
    mutates_args=["ssm_states"],
)
def copy_ssm_states(
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    run_config: dict = None,
):
    """
    Copy SSM states from source indices to destination indices.

    Args:
        ssm_states: SSM state buffer [num_buffers, num_heads, key_dim, value_dim]
        src_indices: Source buffer indices [batch_size]
        dst_indices: Destination buffer indices [batch_size]
        run_config: Auto-tuned configuration
    """
    batch_size = src_indices.shape[0]
    num_heads = ssm_states.shape[1]
    key_dim = ssm_states.shape[2]
    value_dim = ssm_states.shape[3]

    if run_config is None:
        BLOCK_KEY = triton.next_power_of_2(min(key_dim, 32))
        BLOCK_VALUE = triton.next_power_of_2(min(value_dim, 64))
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_KEY = run_config["BLOCK_KEY"]
        BLOCK_VALUE = run_config["BLOCK_VALUE"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_key_blocks = triton.cdiv(key_dim, BLOCK_KEY)
    num_value_blocks = triton.cdiv(value_dim, BLOCK_VALUE)
    num_blocks_total = num_key_blocks * num_value_blocks

    grid = (batch_size, num_heads, num_blocks_total)

    _copy_ssm_state_kernel[grid](
        ssm_states,
        src_indices,
        dst_indices,
        batch_size,
        num_heads,
        key_dim,
        value_dim,
        ssm_states.stride(0),
        ssm_states.stride(1),
        ssm_states.stride(2),
        ssm_states.stride(3),
        BLOCK_KEY=BLOCK_KEY,
        BLOCK_VALUE=BLOCK_VALUE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# =============================================================================
# Optimized Flat Copy Kernels (for contiguous memory)
# =============================================================================
# These kernels leverage the fact that both conv_states and ssm_states are
# contiguous in memory, allowing us to flatten the inner dimensions and use
# efficient 1D vectorized copy patterns.


@triton.jit
def _copy_state_flat_kernel(
    # State buffer pointer (flattened view)
    state_ptr,
    # Buffer indices
    src_indices_ptr,
    dst_indices_ptr,
    # Dimensions
    batch_size,
    flat_size,  # Total elements per buffer entry (flattened inner dims)
    # Strides
    stride_buffer,  # Stride to next buffer entry (in elements)
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized flat copy kernel for contiguous state buffers.

    Instead of using 2D/3D block patterns with stride calculations, this kernel
    treats each buffer entry as a flat 1D array and uses vectorized loads/stores
    for efficient memory transfer.

    Grid: (batch_size, num_blocks) where num_blocks = ceil(flat_size / BLOCK_SIZE)
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Calculate element range for this block
    elem_start = block_idx * BLOCK_SIZE
    elem_offsets = elem_start + tl.arange(0, BLOCK_SIZE)
    elem_mask = elem_offsets < flat_size

    # Load buffer indices for this batch item
    src_idx = tl.load(src_indices_ptr + batch_idx).to(tl.int64)
    dst_idx = tl.load(dst_indices_ptr + batch_idx).to(tl.int64)

    # Calculate source and destination base pointers
    src_base = state_ptr + src_idx * stride_buffer
    dst_base = state_ptr + dst_idx * stride_buffer

    # Vectorized copy
    data = tl.load(src_base + elem_offsets, mask=elem_mask, other=0.0)
    tl.store(dst_base + elem_offsets, data, mask=elem_mask)


@triton.jit
def _copy_states_fused_kernel(
    # Conv state buffer (flattened view)
    conv_state_ptr,
    # SSM state buffer (flattened view)
    ssm_state_ptr,
    # Buffer indices
    src_indices_ptr,
    dst_indices_ptr,
    # Dimensions
    batch_size,
    conv_flat_size,  # Total elements per conv buffer entry
    ssm_flat_size,  # Total elements per ssm buffer entry
    # Strides (in elements)
    conv_stride_buffer,
    ssm_stride_buffer,
    # Block sizes
    CONV_BLOCK_SIZE: tl.constexpr,
    SSM_BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel to copy both conv_states and ssm_states in a single launch.

    This reduces kernel launch overhead by processing both state copies together.
    Each thread block handles one batch item and copies both states sequentially.

    Grid: (batch_size, max(conv_blocks, ssm_blocks))
    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Load buffer indices (same for both conv and ssm)
    src_idx = tl.load(src_indices_ptr + batch_idx).to(tl.int64)
    dst_idx = tl.load(dst_indices_ptr + batch_idx).to(tl.int64)

    # ========== Copy Conv State ==========
    conv_num_blocks = tl.cdiv(conv_flat_size, CONV_BLOCK_SIZE)
    if block_idx < conv_num_blocks:
        conv_elem_start = block_idx * CONV_BLOCK_SIZE
        conv_elem_offsets = conv_elem_start + tl.arange(0, CONV_BLOCK_SIZE)
        conv_mask = conv_elem_offsets < conv_flat_size

        conv_src_base = conv_state_ptr + src_idx * conv_stride_buffer
        conv_dst_base = conv_state_ptr + dst_idx * conv_stride_buffer

        conv_data = tl.load(conv_src_base + conv_elem_offsets, mask=conv_mask, other=0.0)
        tl.store(conv_dst_base + conv_elem_offsets, conv_data, mask=conv_mask)

    # ========== Copy SSM State ==========
    ssm_num_blocks = tl.cdiv(ssm_flat_size, SSM_BLOCK_SIZE)
    if block_idx < ssm_num_blocks:
        ssm_elem_start = block_idx * SSM_BLOCK_SIZE
        ssm_elem_offsets = ssm_elem_start + tl.arange(0, SSM_BLOCK_SIZE)
        ssm_mask = ssm_elem_offsets < ssm_flat_size

        ssm_src_base = ssm_state_ptr + src_idx * ssm_stride_buffer
        ssm_dst_base = ssm_state_ptr + dst_idx * ssm_stride_buffer

        ssm_data = tl.load(ssm_src_base + ssm_elem_offsets, mask=ssm_mask, other=0.0)
        tl.store(ssm_dst_base + ssm_elem_offsets, ssm_data, mask=ssm_mask)


def _get_flat_copy_configs():
    """Generate candidate configurations for flat copy kernel."""
    configs = []
    # Larger block sizes for better memory throughput on contiguous data
    for block_size in [256, 512, 1024, 2048]:
        for num_warps in [4, 8]:
            configs.append(
                {
                    "BLOCK_SIZE": block_size,
                    "num_warps": num_warps,
                    "num_stages": 2,
                }
            )
    return configs


def _get_conv_flat_copy_static_key(
    conv_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Static key for conv flat copy."""
    return {
        "dtype": str(conv_states.dtype),
        "flat_size": conv_states.shape[1] * conv_states.shape[2],
    }


def _get_conv_flat_copy_run_key(
    conv_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Run key for conv flat copy."""
    return src_indices.shape[0]


@autotune(
    kernel_name="gdn_conv_state_flat_copy:v1",
    configs_gen_func=_get_flat_copy_configs,
    static_key_func=_get_conv_flat_copy_static_key,
    run_key_func=_get_conv_flat_copy_run_key,
    mutates_args=["conv_states"],
)
def copy_conv_states_flat(
    conv_states: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    run_config: dict = None,
):
    """
    Optimized flat copy for conv states leveraging contiguous memory.

    Args:
        conv_states: Conv state buffer [num_buffers, dim, conv_width-1] (MUST be contiguous)
        src_indices: Source buffer indices [batch_size]
        dst_indices: Destination buffer indices [batch_size]
        run_config: Auto-tuned configuration
    """
    assert conv_states.is_contiguous(), "conv_states must be contiguous for flat copy"

    batch_size = src_indices.shape[0]
    # Flatten inner dimensions
    flat_size = conv_states.shape[1] * conv_states.shape[2]
    stride_buffer = conv_states.stride(0)

    if run_config is None:
        BLOCK_SIZE = 1024
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_SIZE = run_config["BLOCK_SIZE"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks = triton.cdiv(flat_size, BLOCK_SIZE)
    grid = (batch_size, num_blocks)

    _copy_state_flat_kernel[grid](
        conv_states,
        src_indices,
        dst_indices,
        batch_size,
        flat_size,
        stride_buffer,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _get_ssm_flat_copy_static_key(
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Static key for ssm flat copy."""
    return {
        "dtype": str(ssm_states.dtype),
        "flat_size": ssm_states.shape[1] * ssm_states.shape[2] * ssm_states.shape[3],
    }


def _get_ssm_flat_copy_run_key(
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Run key for ssm flat copy."""
    return src_indices.shape[0]


@autotune(
    kernel_name="gdn_ssm_state_flat_copy:v1",
    configs_gen_func=_get_flat_copy_configs,
    static_key_func=_get_ssm_flat_copy_static_key,
    run_key_func=_get_ssm_flat_copy_run_key,
    mutates_args=["ssm_states"],
)
def copy_ssm_states_flat(
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    run_config: dict = None,
):
    """
    Optimized flat copy for SSM states leveraging contiguous memory.

    Args:
        ssm_states: SSM state buffer [num_buffers, num_heads, key_dim, value_dim] (MUST be contiguous)
        src_indices: Source buffer indices [batch_size]
        dst_indices: Destination buffer indices [batch_size]
        run_config: Auto-tuned configuration
    """
    assert ssm_states.is_contiguous(), "ssm_states must be contiguous for flat copy"

    batch_size = src_indices.shape[0]
    # Flatten inner dimensions (num_heads * key_dim * value_dim)
    flat_size = ssm_states.shape[1] * ssm_states.shape[2] * ssm_states.shape[3]
    stride_buffer = ssm_states.stride(0)

    if run_config is None:
        BLOCK_SIZE = 1024
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_SIZE = run_config["BLOCK_SIZE"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks = triton.cdiv(flat_size, BLOCK_SIZE)
    grid = (batch_size, num_blocks)

    _copy_state_flat_kernel[grid](
        ssm_states,
        src_indices,
        dst_indices,
        batch_size,
        flat_size,
        stride_buffer,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _get_fused_copy_configs():
    """Generate candidate configurations for fused copy kernel."""
    configs = []
    # Use power-of-2 block sizes for both conv and ssm
    for conv_block in [256, 512, 1024]:
        for ssm_block in [256, 512, 1024]:
            for num_warps in [4, 8]:
                configs.append(
                    {
                        "CONV_BLOCK_SIZE": conv_block,
                        "SSM_BLOCK_SIZE": ssm_block,
                        "num_warps": num_warps,
                        "num_stages": 2,
                    }
                )
    return configs


def _get_fused_copy_static_key(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Static key for fused copy."""
    return {
        "conv_dtype": str(conv_states.dtype),
        "ssm_dtype": str(ssm_states.dtype),
        "conv_flat_size": conv_states.shape[1] * conv_states.shape[2],
        "ssm_flat_size": ssm_states.shape[1] * ssm_states.shape[2] * ssm_states.shape[3],
    }


def _get_fused_copy_run_key(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
):
    """Run key for fused copy."""
    return src_indices.shape[0]


@autotune(
    kernel_name="gdn_states_fused_copy:v1",
    configs_gen_func=_get_fused_copy_configs,
    static_key_func=_get_fused_copy_static_key,
    run_key_func=_get_fused_copy_run_key,
    mutates_args=["conv_states", "ssm_states"],
)
def copy_states_fused(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    run_config: dict = None,
):
    """
    Fused copy for both conv and SSM states in a single kernel launch.

    This reduces kernel launch overhead by processing both state copies together.

    Args:
        conv_states: Conv state buffer [num_buffers, dim, conv_width-1] (MUST be contiguous)
        ssm_states: SSM state buffer [num_buffers, num_heads, key_dim, value_dim] (MUST be contiguous)
        src_indices: Source buffer indices [batch_size]
        dst_indices: Destination buffer indices [batch_size]
        run_config: Auto-tuned configuration
    """
    assert conv_states.is_contiguous(), "conv_states must be contiguous for fused copy"
    assert ssm_states.is_contiguous(), "ssm_states must be contiguous for fused copy"

    batch_size = src_indices.shape[0]

    # Flatten inner dimensions
    conv_flat_size = conv_states.shape[1] * conv_states.shape[2]
    ssm_flat_size = ssm_states.shape[1] * ssm_states.shape[2] * ssm_states.shape[3]

    conv_stride_buffer = conv_states.stride(0)
    ssm_stride_buffer = ssm_states.stride(0)

    if run_config is None:
        CONV_BLOCK_SIZE = 512
        SSM_BLOCK_SIZE = 512
        num_warps = 4
        num_stages = 2
    else:
        CONV_BLOCK_SIZE = run_config["CONV_BLOCK_SIZE"]
        SSM_BLOCK_SIZE = run_config["SSM_BLOCK_SIZE"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    # Grid covers both conv and ssm blocks
    conv_num_blocks = triton.cdiv(conv_flat_size, CONV_BLOCK_SIZE)
    ssm_num_blocks = triton.cdiv(ssm_flat_size, SSM_BLOCK_SIZE)
    max_blocks = max(conv_num_blocks, ssm_num_blocks)
    grid = (batch_size, max_blocks)

    _copy_states_fused_kernel[grid](
        conv_states,
        ssm_states,
        src_indices,
        dst_indices,
        batch_size,
        conv_flat_size,
        ssm_flat_size,
        conv_stride_buffer,
        ssm_stride_buffer,
        CONV_BLOCK_SIZE=CONV_BLOCK_SIZE,
        SSM_BLOCK_SIZE=SSM_BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
