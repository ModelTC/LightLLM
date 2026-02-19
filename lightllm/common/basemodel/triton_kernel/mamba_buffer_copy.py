"""
Optimized Mamba Buffer Copy Kernels with Autotune Support

This module provides auto-tuned Triton kernels for efficient buffer copying operations
in Mamba-style models, including support for MTP (Multi-Token Prediction) buffer broadcasting.
"""

import torch
import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _copy_buffer_p2p_1d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    pair_idx_offset,
    layer_idx_offset,
    stride_layer,
    stride_index,
    stride_d,
    d_size,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized kernel for 1D buffer copy.

    Grid: (num_pairs, layer_num, num_blocks_d)
    Each program copies one block of dimension d for one (pair, layer) combination.
    """
    pair_idx = tl.program_id(0) + pair_idx_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_d_idx = tl.program_id(2)

    # Load source and destination indices for this pair
    src_idx = tl.load(src_indexes_ptr + pair_idx).to(tl.int64)
    dst_idx = tl.load(dst_indexes_ptr + pair_idx).to(tl.int64)

    # Calculate offsets for this block
    d_start = block_d_idx * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)

    # Create mask for valid indices
    mask = d_offsets < d_size

    # Calculate source and destination pointers for this layer and pair
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_index
    base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_index

    src_ptr = base_src + d_offsets * stride_d
    dst_ptr = base_dst + d_offsets * stride_d

    # Load and store
    data = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(dst_ptr, data, mask=mask)


@triton.jit
def _copy_buffer_p2p_2d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    pair_idx_offset,
    layer_idx_offset,
    stride_layer,
    stride_index,
    stride_d1,
    stride_d2,
    d1_size,
    d2_size,
    num_blocks_d2,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    """
    Kernel to copy 2D buffer from source indices to destination indices.

    Grid: (num_pairs, layer_num, num_blocks_d1 * num_blocks_d2)
    Each program copies one 2D block for one (pair, layer) combination.
    """
    pair_idx = tl.program_id(0) + pair_idx_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_idx = tl.program_id(2)

    # Decompose block_idx into d1 and d2 block indices
    block_d1_idx = block_idx // num_blocks_d2
    block_d2_idx = block_idx % num_blocks_d2

    # Load source and destination indices
    src_idx = tl.load(src_indexes_ptr + pair_idx).to(tl.int64)
    dst_idx = tl.load(dst_indexes_ptr + pair_idx).to(tl.int64)

    # Calculate offsets for this block
    d1_start = block_d1_idx * BLOCK_D1
    d2_start = block_d2_idx * BLOCK_D2

    d1_offsets = d1_start + tl.arange(0, BLOCK_D1)
    d2_offsets = d2_start + tl.arange(0, BLOCK_D2)

    # Create mask for valid indices
    d1_mask = d1_offsets < d1_size
    d2_mask = d2_offsets < d2_size
    mask = d1_mask[:, None] & d2_mask[None, :]

    # Calculate base pointers
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_index
    base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_index

    # Calculate full offsets
    offsets = d1_offsets[:, None] * stride_d1 + d2_offsets[None, :] * stride_d2

    # Load and store
    data = tl.load(base_src + offsets, mask=mask, other=0.0)
    tl.store(base_dst + offsets, data, mask=mask)


@triton.jit
def _copy_buffer_broadcast_1d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    copy_idx_offset,
    layer_idx_offset,
    stride_layer,
    stride_index,
    stride_d,
    d_size,
    num_dst_per_src,
    BLOCK_D: tl.constexpr,
):
    """
    Broadcast kernel for 1D buffer copy (one source to multiple destinations).

    Grid: (num_src, layer_num, num_blocks_d)
    """
    src_idx_in_batch = tl.program_id(0) + copy_idx_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_d_idx = tl.program_id(2)

    # Load source index
    src_idx = tl.load(src_indexes_ptr + src_idx_in_batch).to(tl.int64)

    # Calculate offsets for this block
    d_start = block_d_idx * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask = d_offsets < d_size

    # Calculate source pointer
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_index
    src_ptr = base_src + d_offsets * stride_d

    # Load data once
    data = tl.load(src_ptr, mask=mask, other=0.0)

    # Broadcast to all destinations for this source
    for dst_offset in range(num_dst_per_src):
        dst_idx_in_batch = src_idx_in_batch * num_dst_per_src + dst_offset
        dst_idx = tl.load(dst_indexes_ptr + dst_idx_in_batch).to(tl.int64)

        base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_index
        dst_ptr = base_dst + d_offsets * stride_d

        tl.store(dst_ptr, data, mask=mask)


@triton.jit
def _copy_buffer_broadcast_2d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    copy_idx_offset,
    layer_idx_offset,
    stride_layer,
    stride_index,
    stride_d1,
    stride_d2,
    d1_size,
    d2_size,
    num_blocks_d2,
    num_dst_per_src,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
):
    """
    Broadcast kernel for 2D buffer copy (one source to multiple destinations).

    Grid: (num_src, layer_num, num_blocks_d1 * num_blocks_d2)
    """
    src_idx_in_batch = tl.program_id(0) + copy_idx_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_idx = tl.program_id(2)

    # Decompose block_idx
    block_d1_idx = block_idx // num_blocks_d2
    block_d2_idx = block_idx % num_blocks_d2

    # Load source index
    src_idx = tl.load(src_indexes_ptr + src_idx_in_batch).to(tl.int64)

    # Calculate offsets
    d1_start = block_d1_idx * BLOCK_D1
    d2_start = block_d2_idx * BLOCK_D2

    d1_offsets = d1_start + tl.arange(0, BLOCK_D1)
    d2_offsets = d2_start + tl.arange(0, BLOCK_D2)

    d1_mask = d1_offsets < d1_size
    d2_mask = d2_offsets < d2_size
    mask = d1_mask[:, None] & d2_mask[None, :]

    # Calculate source pointer and load data once
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_index
    offsets = d1_offsets[:, None] * stride_d1 + d2_offsets[None, :] * stride_d2
    data = tl.load(base_src + offsets, mask=mask, other=0.0)

    # Broadcast to all destinations
    for dst_offset in range(num_dst_per_src):
        dst_idx_in_batch = src_idx_in_batch * num_dst_per_src + dst_offset
        dst_idx = tl.load(dst_indexes_ptr + dst_idx_in_batch).to(tl.int64)

        base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_index
        tl.store(base_dst + offsets, data, mask=mask)


@triton.jit
def _copy_buffer_p2p_3d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    pair_idx_offset,
    layer_idx_offset,
    stride_layer,
    stride_index,
    stride_d1,
    stride_d2,
    stride_d3,
    d1_size,
    d2_size,
    d3_size,
    num_blocks_d2,
    num_blocks_d3,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
):
    """
    Optimized kernel for 3D data buffer copy (5D tensor: layer, buffer, d1, d2, d3).

    Grid: (num_pairs, layer_num, num_blocks_d1 * num_blocks_d2 * num_blocks_d3)
    Each program copies one 3D block for one (pair, layer) combination.
    """
    pair_idx = tl.program_id(0) + pair_idx_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_idx = tl.program_id(2)

    # Decompose block_idx into d1, d2, d3 block indices
    block_d1_idx = block_idx // (num_blocks_d2 * num_blocks_d3)
    temp = block_idx % (num_blocks_d2 * num_blocks_d3)
    block_d2_idx = temp // num_blocks_d3
    block_d3_idx = temp % num_blocks_d3

    # Load source and destination indices for this pair
    src_idx = tl.load(src_indexes_ptr + pair_idx).to(tl.int64)
    dst_idx = tl.load(dst_indexes_ptr + pair_idx).to(tl.int64)

    # Calculate offsets for this block
    d1_start = block_d1_idx * BLOCK_D1
    d2_start = block_d2_idx * BLOCK_D2
    d3_start = block_d3_idx * BLOCK_D3

    d1_offsets = d1_start + tl.arange(0, BLOCK_D1)
    d2_offsets = d2_start + tl.arange(0, BLOCK_D2)
    d3_offsets = d3_start + tl.arange(0, BLOCK_D3)

    # Create masks for valid indices
    d1_mask = d1_offsets < d1_size
    d2_mask = d2_offsets < d2_size
    d3_mask = d3_offsets < d3_size

    # 3D mask: [BLOCK_D1, BLOCK_D2, BLOCK_D3]
    mask = d1_mask[:, None, None] & d2_mask[None, :, None] & d3_mask[None, None, :]

    # Calculate base pointers
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_index
    base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_index

    # Calculate full 3D offsets
    offsets = (
        d1_offsets[:, None, None] * stride_d1
        + d2_offsets[None, :, None] * stride_d2
        + d3_offsets[None, None, :] * stride_d3
    )

    # Load and store
    data = tl.load(base_src + offsets, mask=mask, other=0.0)
    tl.store(base_dst + offsets, data, mask=mask)


@triton.jit
def _copy_buffer_broadcast_3d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    copy_idx_offset,
    layer_idx_offset,
    stride_layer,
    stride_index,
    stride_d1,
    stride_d2,
    stride_d3,
    d1_size,
    d2_size,
    d3_size,
    num_blocks_d2,
    num_blocks_d3,
    num_dst_per_src,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
):
    """
    Broadcast kernel for 3D data buffer copy (5D tensor: layer, buffer, d1, d2, d3).

    Grid: (num_src, layer_num, num_blocks_d1 * num_blocks_d2 * num_blocks_d3)
    Each program loads once from source and broadcasts to all destinations.
    """
    src_idx_in_batch = tl.program_id(0) + copy_idx_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_idx = tl.program_id(2)

    # Decompose block_idx into d1, d2, d3 block indices
    block_d1_idx = block_idx // (num_blocks_d2 * num_blocks_d3)
    temp = block_idx % (num_blocks_d2 * num_blocks_d3)
    block_d2_idx = temp // num_blocks_d3
    block_d3_idx = temp % num_blocks_d3

    # Load source index
    src_idx = tl.load(src_indexes_ptr + src_idx_in_batch).to(tl.int64)

    # Calculate offsets for this block
    d1_start = block_d1_idx * BLOCK_D1
    d2_start = block_d2_idx * BLOCK_D2
    d3_start = block_d3_idx * BLOCK_D3

    d1_offsets = d1_start + tl.arange(0, BLOCK_D1)
    d2_offsets = d2_start + tl.arange(0, BLOCK_D2)
    d3_offsets = d3_start + tl.arange(0, BLOCK_D3)

    # Create masks
    d1_mask = d1_offsets < d1_size
    d2_mask = d2_offsets < d2_size
    d3_mask = d3_offsets < d3_size

    mask = d1_mask[:, None, None] & d2_mask[None, :, None] & d3_mask[None, None, :]

    # Calculate source pointer and load data once
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_index

    offsets = (
        d1_offsets[:, None, None] * stride_d1
        + d2_offsets[None, :, None] * stride_d2
        + d3_offsets[None, None, :] * stride_d3
    )

    data = tl.load(base_src + offsets, mask=mask, other=0.0)

    # Broadcast to all destinations for this source
    for dst_offset in range(num_dst_per_src):
        dst_idx_in_batch = src_idx_in_batch * num_dst_per_src + dst_offset
        dst_idx = tl.load(dst_indexes_ptr + dst_idx_in_batch).to(tl.int64)

        base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_index
        tl.store(base_dst + offsets, data, mask=mask)


# ==================== Config Generation Functions ====================


def _get_buffer_copy_1d_configs():
    """Generate candidate configurations for 1D buffer copy."""
    configs = []
    for block_d in [32, 64, 128, 256, 512, 1024]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                configs.append(
                    {
                        "BLOCK_D": block_d,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                )
    return configs


def _get_buffer_copy_2d_configs():
    """Generate candidate configurations for 2D buffer copy."""
    configs = []
    for block_d1 in [16, 32, 64, 128]:
        for block_d2 in [16, 32, 64, 128, 256]:
            for num_warps in [2, 4, 8]:
                for num_stages in [2, 3, 4]:
                    configs.append(
                        {
                            "BLOCK_D1": block_d1,
                            "BLOCK_D2": block_d2,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                    )
    return configs


def _get_buffer_copy_3d_configs():
    """Generate candidate configurations for 3D buffer copy (5D tensor)."""
    configs = []
    for block_d1 in [8, 16, 32]:
        for block_d2 in [8, 16, 32, 64]:
            for block_d3 in [8, 16, 32, 64, 128]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3]:
                        # Skip configs that are too large for shared memory
                        if block_d1 * block_d2 * block_d3 > 32768:
                            continue
                        configs.append(
                            {
                                "BLOCK_D1": block_d1,
                                "BLOCK_D2": block_d2,
                                "BLOCK_D3": block_d3,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )
    return configs


# ==================== Static and Run Key Functions ====================


def _get_buffer_copy_static_key(src_buffer: torch.Tensor):
    """Static key based on buffer shape and dtype."""
    shape = src_buffer.shape
    return {
        "ndim": len(shape),
        "layer_num": shape[0],
        "d_sizes": str(shape[2:]),  # Dimension sizes
        "dtype": str(src_buffer.dtype),
    }


def _get_buffer_copy_run_key(src_indexes: torch.Tensor):
    """Run key based on number of copy pairs."""
    return src_indexes.shape[0]


# ==================== Auto-tuned Buffer Copy Functions ====================


@autotune(
    kernel_name="mamba_buffer_copy_p2p_1d:v1",
    configs_gen_func=_get_buffer_copy_1d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_buffer_p2p_1d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Auto-tuned 1D buffer copy."""
    num_pairs = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d_size = src_buffer.shape[2]

    if run_config is None:
        # Default config if autotune is disabled
        BLOCK_D = triton.next_power_of_2(min(d_size, 256))
        num_warps = 4 if BLOCK_D > 256 else 2
        num_stages = 2
    else:
        BLOCK_D = run_config["BLOCK_D"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_d = triton.cdiv(d_size, BLOCK_D)

    MAX_GRID_SIZE = 65535

    for pair_chunk_start in range(0, num_pairs, MAX_GRID_SIZE):
        pair_chunk_end = min(pair_chunk_start + MAX_GRID_SIZE, num_pairs)
        pair_chunk_size = pair_chunk_end - pair_chunk_start

        for layer_chunk_start in range(0, layer_num, MAX_GRID_SIZE):
            layer_chunk_end = min(layer_chunk_start + MAX_GRID_SIZE, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (pair_chunk_size, layer_chunk_size, num_blocks_d)

            _copy_buffer_p2p_1d_kernel[grid](
                src_buffer,
                dst_buffer,
                src_indexes,
                dst_indexes,
                pair_chunk_start,
                layer_chunk_start,
                src_buffer.stride(0),
                src_buffer.stride(1),
                src_buffer.stride(2),
                d_size,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )


@autotune(
    kernel_name="mamba_buffer_copy_p2p_2d:v1",
    configs_gen_func=_get_buffer_copy_2d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_buffer_p2p_2d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Auto-tuned 2D buffer copy."""
    num_pairs = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d1_size = src_buffer.shape[2]
    d2_size = src_buffer.shape[3]

    if run_config is None:
        # Default config if autotune is disabled
        BLOCK_D1 = triton.next_power_of_2(min(d1_size, 64))
        BLOCK_D2 = triton.next_power_of_2(min(d2_size, 128))
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_D1 = run_config["BLOCK_D1"]
        BLOCK_D2 = run_config["BLOCK_D2"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_d1 = triton.cdiv(d1_size, BLOCK_D1)
    num_blocks_d2 = triton.cdiv(d2_size, BLOCK_D2)
    num_blocks_total = num_blocks_d1 * num_blocks_d2

    MAX_GRID_SIZE = 65535

    for pair_chunk_start in range(0, num_pairs, MAX_GRID_SIZE):
        pair_chunk_end = min(pair_chunk_start + MAX_GRID_SIZE, num_pairs)
        pair_chunk_size = pair_chunk_end - pair_chunk_start

        for layer_chunk_start in range(0, layer_num, MAX_GRID_SIZE):
            layer_chunk_end = min(layer_chunk_start + MAX_GRID_SIZE, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (pair_chunk_size, layer_chunk_size, num_blocks_total)

            _copy_buffer_p2p_2d_kernel[grid](
                src_buffer,
                dst_buffer,
                src_indexes,
                dst_indexes,
                pair_chunk_start,
                layer_chunk_start,
                src_buffer.stride(0),
                src_buffer.stride(1),
                src_buffer.stride(2),
                src_buffer.stride(3),
                d1_size,
                d2_size,
                num_blocks_d2,
                BLOCK_D1=BLOCK_D1,
                BLOCK_D2=BLOCK_D2,
                num_warps=num_warps,
                num_stages=num_stages,
            )


@autotune(
    kernel_name="mamba_buffer_broadcast_1d:v1",
    configs_gen_func=_get_buffer_copy_1d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_buffer_broadcast_1d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Auto-tuned 1D buffer broadcast (one src to multiple dst)."""
    num_src = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d_size = src_buffer.shape[2]
    num_dst_per_src = dst_indexes.shape[0] // num_src

    if run_config is None:
        BLOCK_D = triton.next_power_of_2(min(d_size, 256))
        num_warps = 4 if BLOCK_D > 256 else 2
        num_stages = 2
    else:
        BLOCK_D = run_config["BLOCK_D"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_d = triton.cdiv(d_size, BLOCK_D)

    MAX_GRID_SIZE = 65535

    for src_chunk_start in range(0, num_src, MAX_GRID_SIZE):
        src_chunk_end = min(src_chunk_start + MAX_GRID_SIZE, num_src)
        src_chunk_size = src_chunk_end - src_chunk_start

        for layer_chunk_start in range(0, layer_num, MAX_GRID_SIZE):
            layer_chunk_end = min(layer_chunk_start + MAX_GRID_SIZE, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (src_chunk_size, layer_chunk_size, num_blocks_d)

            _copy_buffer_broadcast_1d_kernel[grid](
                src_buffer,
                dst_buffer,
                src_indexes,
                dst_indexes,
                src_chunk_start,
                layer_chunk_start,
                src_buffer.stride(0),
                src_buffer.stride(1),
                src_buffer.stride(2),
                d_size,
                num_dst_per_src,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )


@autotune(
    kernel_name="mamba_buffer_broadcast_2d:v1",
    configs_gen_func=_get_buffer_copy_2d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_buffer_broadcast_2d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Auto-tuned 2D buffer broadcast (one src to multiple dst)."""
    num_src = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d1_size = src_buffer.shape[2]
    d2_size = src_buffer.shape[3]
    num_dst_per_src = dst_indexes.shape[0] // num_src

    if run_config is None:
        BLOCK_D1 = triton.next_power_of_2(min(d1_size, 64))
        BLOCK_D2 = triton.next_power_of_2(min(d2_size, 128))
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_D1 = run_config["BLOCK_D1"]
        BLOCK_D2 = run_config["BLOCK_D2"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_d1 = triton.cdiv(d1_size, BLOCK_D1)
    num_blocks_d2 = triton.cdiv(d2_size, BLOCK_D2)
    num_blocks_total = num_blocks_d1 * num_blocks_d2

    MAX_GRID_SIZE = 65535

    for src_chunk_start in range(0, num_src, MAX_GRID_SIZE):
        src_chunk_end = min(src_chunk_start + MAX_GRID_SIZE, num_src)
        src_chunk_size = src_chunk_end - src_chunk_start

        for layer_chunk_start in range(0, layer_num, MAX_GRID_SIZE):
            layer_chunk_end = min(layer_chunk_start + MAX_GRID_SIZE, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (src_chunk_size, layer_chunk_size, num_blocks_total)

            _copy_buffer_broadcast_2d_kernel[grid](
                src_buffer,
                dst_buffer,
                src_indexes,
                dst_indexes,
                src_chunk_start,
                layer_chunk_start,
                src_buffer.stride(0),
                src_buffer.stride(1),
                src_buffer.stride(2),
                src_buffer.stride(3),
                d1_size,
                d2_size,
                num_blocks_d2,
                num_dst_per_src,
                BLOCK_D1=BLOCK_D1,
                BLOCK_D2=BLOCK_D2,
                num_warps=num_warps,
                num_stages=num_stages,
            )


@autotune(
    kernel_name="mamba_buffer_copy_p2p_3d:v1",
    configs_gen_func=_get_buffer_copy_3d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_buffer_p2p_3d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Auto-tuned 3D data buffer copy (5D tensor)."""
    num_pairs = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d1_size = src_buffer.shape[2]
    d2_size = src_buffer.shape[3]
    d3_size = src_buffer.shape[4]

    if run_config is None:
        BLOCK_D1 = triton.next_power_of_2(min(d1_size, 16))
        BLOCK_D2 = triton.next_power_of_2(min(d2_size, 32))
        BLOCK_D3 = triton.next_power_of_2(min(d3_size, 64))
        num_warps = 4 if BLOCK_D1 * BLOCK_D2 * BLOCK_D3 > 4096 else 8
        num_stages = 2
    else:
        BLOCK_D1 = run_config["BLOCK_D1"]
        BLOCK_D2 = run_config["BLOCK_D2"]
        BLOCK_D3 = run_config["BLOCK_D3"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_d1 = triton.cdiv(d1_size, BLOCK_D1)
    num_blocks_d2 = triton.cdiv(d2_size, BLOCK_D2)
    num_blocks_d3 = triton.cdiv(d3_size, BLOCK_D3)
    num_blocks_total = num_blocks_d1 * num_blocks_d2 * num_blocks_d3

    MAX_GRID_SIZE = 65535

    for pair_chunk_start in range(0, num_pairs, MAX_GRID_SIZE):
        pair_chunk_end = min(pair_chunk_start + MAX_GRID_SIZE, num_pairs)
        pair_chunk_size = pair_chunk_end - pair_chunk_start

        for layer_chunk_start in range(0, layer_num, MAX_GRID_SIZE):
            layer_chunk_end = min(layer_chunk_start + MAX_GRID_SIZE, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (pair_chunk_size, layer_chunk_size, num_blocks_total)

            _copy_buffer_p2p_3d_kernel[grid](
                src_buffer,
                dst_buffer,
                src_indexes,
                dst_indexes,
                pair_chunk_start,
                layer_chunk_start,
                src_buffer.stride(0),
                src_buffer.stride(1),
                src_buffer.stride(2),
                src_buffer.stride(3),
                src_buffer.stride(4),
                d1_size,
                d2_size,
                d3_size,
                num_blocks_d2,
                num_blocks_d3,
                BLOCK_D1=BLOCK_D1,
                BLOCK_D2=BLOCK_D2,
                BLOCK_D3=BLOCK_D3,
                num_warps=num_warps,
                num_stages=num_stages,
            )


@autotune(
    kernel_name="mamba_buffer_broadcast_3d:v1",
    configs_gen_func=_get_buffer_copy_3d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_buffer_broadcast_3d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Auto-tuned 3D data buffer broadcast (5D tensor, one src to multiple dst)."""
    num_src = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d1_size = src_buffer.shape[2]
    d2_size = src_buffer.shape[3]
    d3_size = src_buffer.shape[4]
    num_dst_per_src = dst_indexes.shape[0] // num_src

    if run_config is None:
        BLOCK_D1 = triton.next_power_of_2(min(d1_size, 16))
        BLOCK_D2 = triton.next_power_of_2(min(d2_size, 32))
        BLOCK_D3 = triton.next_power_of_2(min(d3_size, 64))
        num_warps = 4 if BLOCK_D1 * BLOCK_D2 * BLOCK_D3 > 4096 else 8
        num_stages = 2
    else:
        BLOCK_D1 = run_config["BLOCK_D1"]
        BLOCK_D2 = run_config["BLOCK_D2"]
        BLOCK_D3 = run_config["BLOCK_D3"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_d1 = triton.cdiv(d1_size, BLOCK_D1)
    num_blocks_d2 = triton.cdiv(d2_size, BLOCK_D2)
    num_blocks_d3 = triton.cdiv(d3_size, BLOCK_D3)
    num_blocks_total = num_blocks_d1 * num_blocks_d2 * num_blocks_d3

    MAX_GRID_SIZE = 65535

    for src_chunk_start in range(0, num_src, MAX_GRID_SIZE):
        src_chunk_end = min(src_chunk_start + MAX_GRID_SIZE, num_src)
        src_chunk_size = src_chunk_end - src_chunk_start

        for layer_chunk_start in range(0, layer_num, MAX_GRID_SIZE):
            layer_chunk_end = min(layer_chunk_start + MAX_GRID_SIZE, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (src_chunk_size, layer_chunk_size, num_blocks_total)

            _copy_buffer_broadcast_3d_kernel[grid](
                src_buffer,
                dst_buffer,
                src_indexes,
                dst_indexes,
                src_chunk_start,
                layer_chunk_start,
                src_buffer.stride(0),
                src_buffer.stride(1),
                src_buffer.stride(2),
                src_buffer.stride(3),
                src_buffer.stride(4),
                d1_size,
                d2_size,
                d3_size,
                num_blocks_d2,
                num_blocks_d3,
                num_dst_per_src,
                BLOCK_D1=BLOCK_D1,
                BLOCK_D2=BLOCK_D2,
                BLOCK_D3=BLOCK_D3,
                num_warps=num_warps,
                num_stages=num_stages,
            )


# ==================== Unified Interface ====================


def copy_buffer_p2p(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """
    Copy buffers from source indices to destination indices with auto-tuning.

    Supports 3D (conv states), 4D (standard buffers), and 5D (SSM states) buffers.

    Args:
        src_buffer: Source buffer tensor [layer_num, buffer_size, ...]
        dst_buffer: Destination buffer tensor [layer_num, buffer_size, ...]
        src_indexes: Source buffer indices [num_pairs]
        dst_indexes: Destination buffer indices [num_pairs]
    """
    assert src_buffer.shape == dst_buffer.shape
    assert src_indexes.shape == dst_indexes.shape
    assert len(src_indexes.shape) == 1

    if len(src_buffer.shape) == 3:
        # 1D case: (layer_num, buffer_size, d)
        _copy_buffer_p2p_1d_autotuned(src_buffer, dst_buffer, src_indexes, dst_indexes)

    elif len(src_buffer.shape) == 4:
        # 2D case: (layer_num, buffer_size, d1, d2)
        _copy_buffer_p2p_2d_autotuned(src_buffer, dst_buffer, src_indexes, dst_indexes)

    elif len(src_buffer.shape) == 5:
        # 5D case: (layer_num, buffer_size, d1, d2, d3) - Use Triton kernel for zero extra memory
        _copy_buffer_p2p_3d_autotuned(src_buffer, dst_buffer, src_indexes, dst_indexes)

    else:
        raise ValueError(f"Unsupported buffer shape: {src_buffer.shape}")


def copy_buffer_broadcast(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """
    Broadcast buffers from source indices to multiple destination indices (MTP use case).

    Each source buffer is copied to multiple destination buffers.

    Args:
        src_buffer: Source buffer tensor [layer_num, buffer_size, ...]
        dst_buffer: Destination buffer tensor [layer_num, buffer_size, ...]
        src_indexes: Source buffer indices [num_src]
        dst_indexes: Destination buffer indices [num_src, num_dst_per_src] (2D tensor)
    """
    assert src_buffer.shape == dst_buffer.shape
    assert len(src_indexes.shape) == 1
    assert len(dst_indexes.shape) == 2, f"dst_indexes must be 2D, got shape {dst_indexes.shape}"

    num_src = src_indexes.shape[0]

    assert num_src == dst_indexes.shape[0], f"Mismatch: src_indexes {num_src} vs dst_indexes {dst_indexes.shape[0]}"

    # Flatten dst_indexes for kernel
    dst_indexes_flat = dst_indexes.reshape(-1).contiguous()

    if len(src_buffer.shape) == 3:
        # 1D case
        _copy_buffer_broadcast_1d_autotuned(src_buffer, dst_buffer, src_indexes, dst_indexes_flat)

    elif len(src_buffer.shape) == 4:
        # 2D case
        _copy_buffer_broadcast_2d_autotuned(src_buffer, dst_buffer, src_indexes, dst_indexes_flat)

    elif len(src_buffer.shape) == 5:
        # 5D case: (layer_num, buffer_size, d1, d2, d3) - Use Triton kernel for zero extra memory
        _copy_buffer_broadcast_3d_autotuned(src_buffer, dst_buffer, src_indexes, dst_indexes_flat)

    else:
        raise ValueError(f"Unsupported buffer shape: {src_buffer.shape}")
