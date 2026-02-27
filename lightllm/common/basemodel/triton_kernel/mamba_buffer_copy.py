import torch
import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import autotune

_MAX_GRID_DIM = 65535


@triton.jit
def _copy_mamba_buffer_1d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    chunk_offset,
    layer_idx_offset,
    stride_layer,
    stride_slot,
    stride_d,
    d_size,
    BLOCK_D: tl.constexpr,
):
    """
    Indexed 1:1 copy kernel for Mamba recurrent state buffers.

    Grid: (num_pairs, layer_num, num_blocks_d)
    Each program copies one block of dimension d for one (pair, layer) combination.
    """
    pair_idx = tl.program_id(0) + chunk_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_d_idx = tl.program_id(2)

    # Cast strides to int64 to prevent overflow in pointer arithmetic
    stride_layer = stride_layer.to(tl.int64)
    stride_slot = stride_slot.to(tl.int64)

    # Load source and destination slot indices for this pair
    src_idx = tl.load(src_indexes_ptr + pair_idx).to(tl.int64)
    dst_idx = tl.load(dst_indexes_ptr + pair_idx).to(tl.int64)

    # Calculate offsets for this block
    d_start = block_d_idx * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)

    # Create mask for valid indices
    mask = d_offsets < d_size

    # Calculate source and destination pointers for this layer and pair
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_slot
    base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_slot

    src_ptr = base_src + d_offsets * stride_d
    dst_ptr = base_dst + d_offsets * stride_d

    # Load and store
    data = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(dst_ptr, data, mask=mask)


@triton.jit
def _fork_mamba_buffer_1d_kernel(
    src_buffer_ptr,
    dst_buffer_ptr,
    src_indexes_ptr,
    dst_indexes_ptr,
    chunk_offset,
    layer_idx_offset,
    stride_layer,
    stride_slot,
    stride_d,
    d_size,
    num_dst_per_src,
    BLOCK_D: tl.constexpr,
):
    """
    Fork kernel for Mamba recurrent state buffers: one source slot → N destination slots.

    Used for MTP speculation where one parent state is copied to multiple child slots.
    Grid: (num_src, layer_num, num_blocks_d)
    """
    src_chunk_idx = tl.program_id(0) + chunk_offset
    layer_idx = tl.program_id(1) + layer_idx_offset
    block_d_idx = tl.program_id(2)

    # Cast strides to int64 to prevent overflow in pointer arithmetic
    stride_layer = stride_layer.to(tl.int64)
    stride_slot = stride_slot.to(tl.int64)

    # Load source slot index
    src_idx = tl.load(src_indexes_ptr + src_chunk_idx).to(tl.int64)

    # Calculate offsets for this block
    d_start = block_d_idx * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask = d_offsets < d_size

    # Calculate source pointer and load data once
    base_src = src_buffer_ptr + layer_idx * stride_layer + src_idx * stride_slot
    src_ptr = base_src + d_offsets * stride_d
    data = tl.load(src_ptr, mask=mask, other=0.0)

    # Write to each destination slot for this source
    for dst_offset in range(num_dst_per_src):
        dst_idx_in_batch = src_chunk_idx * num_dst_per_src + dst_offset
        dst_idx = tl.load(dst_indexes_ptr + dst_idx_in_batch).to(tl.int64)

        base_dst = dst_buffer_ptr + layer_idx * stride_layer + dst_idx * stride_slot
        dst_ptr = base_dst + d_offsets * stride_d

        tl.store(dst_ptr, data, mask=mask)


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


# ==================== Static and Run Key Functions ====================


def _get_buffer_copy_static_key(src_buffer: torch.Tensor):
    """Static key based on buffer shape and dtype."""
    shape = src_buffer.shape
    return {
        "ndim": len(shape),
        "layer_num": shape[0],
        "d_sizes": str(shape[2:]),
        "dtype": str(src_buffer.dtype),
    }


def _get_buffer_copy_run_key(src_indexes: torch.Tensor):
    """Run key based on number of copy pairs."""
    return src_indexes.shape[0]


# ==================== Auto-tuned Buffer Copy Functions ====================


@autotune(
    kernel_name="mamba_buffer_copy_1d:v1",
    configs_gen_func=_get_buffer_copy_1d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_mamba_buffer_1d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Auto-tuned indexed 1:1 copy of Mamba recurrent state buffer slots."""
    num_pairs = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d_size = src_buffer.shape[2]

    if run_config is None:
        BLOCK_D = triton.next_power_of_2(min(d_size, 256))
        num_warps = 4 if BLOCK_D > 256 else 2
        num_stages = 2
    else:
        BLOCK_D = run_config["BLOCK_D"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    num_blocks_d = triton.cdiv(d_size, BLOCK_D)

    for pair_chunk_start in range(0, num_pairs, _MAX_GRID_DIM):
        pair_chunk_end = min(pair_chunk_start + _MAX_GRID_DIM, num_pairs)
        pair_chunk_size = pair_chunk_end - pair_chunk_start

        for layer_chunk_start in range(0, layer_num, _MAX_GRID_DIM):
            layer_chunk_end = min(layer_chunk_start + _MAX_GRID_DIM, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (pair_chunk_size, layer_chunk_size, num_blocks_d)

            _copy_mamba_buffer_1d_kernel[grid](
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
    kernel_name="mamba_buffer_fork_1d:v1",
    configs_gen_func=_get_buffer_copy_1d_configs,
    static_key_func=_get_buffer_copy_static_key,
    run_key_func=_get_buffer_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _fork_mamba_buffer_1d_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,  # flat 1D: [num_src * num_dst_per_src]
    run_config: dict = None,
):
    """Auto-tuned fork: copy each source Mamba slot to N destination slots."""
    num_src = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d_size = src_buffer.shape[2]
    assert (
        dst_indexes.shape[0] % num_src == 0
    ), f"dst_indexes length {dst_indexes.shape[0]} must be divisible by num_src {num_src}"
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

    for src_chunk_start in range(0, num_src, _MAX_GRID_DIM):
        src_chunk_end = min(src_chunk_start + _MAX_GRID_DIM, num_src)
        src_chunk_size = src_chunk_end - src_chunk_start

        for layer_chunk_start in range(0, layer_num, _MAX_GRID_DIM):
            layer_chunk_end = min(layer_chunk_start + _MAX_GRID_DIM, layer_num)
            layer_chunk_size = layer_chunk_end - layer_chunk_start

            grid = (src_chunk_size, layer_chunk_size, num_blocks_d)

            _fork_mamba_buffer_1d_kernel[grid](
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


# ==================== Unified Interface ====================


def _flatten_trailing_dims(buffer: torch.Tensor) -> torch.Tensor:
    """Flatten all dimensions after [layer_num, buffer_size] into one.

    For a contiguous buffer of shape [L, B, d1, d2, ...], returns a view
    of shape [L, B, d1*d2*...]. This is a zero-copy operation.
    """
    if buffer.ndim == 3:
        return buffer
    L, B = buffer.shape[:2]
    return buffer.view(L, B, -1)


def copy_mamba_buffer(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """
    Indexed 1:1 copy of Mamba recurrent state buffer slots.

    Copies slot src_indexes[i] → dst_indexes[i] for all layers simultaneously.
    Used for cache eviction/restore and normal token state management.

    Args:
        src_buffer: [layer_num, num_slots, ...]
        dst_buffer: [layer_num, num_slots, ...]
        src_indexes: source slot indices [num_pairs]
        dst_indexes: destination slot indices [num_pairs]
    """
    assert src_buffer.shape == dst_buffer.shape
    assert src_indexes.shape == dst_indexes.shape
    assert len(src_indexes.shape) == 1

    src_flat = _flatten_trailing_dims(src_buffer)
    dst_flat = _flatten_trailing_dims(dst_buffer)
    _copy_mamba_buffer_1d_autotuned(src_flat, dst_flat, src_indexes, dst_indexes)


def fork_mamba_buffer(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """
    Fork Mamba recurrent state slots: copy one source slot to N destination slots.

    Used for MTP (Multi-Token Prediction) speculation, where a parent token's
    recurrent state must be replicated into each speculative child slot.

    Args:
        src_buffer: [layer_num, num_slots, ...]
        dst_buffer: [layer_num, num_slots, ...]
        src_indexes: source slot indices [num_src]
        dst_indexes: destination slot indices [num_src, num_dst_per_src]
    """
    assert src_buffer.shape == dst_buffer.shape
    assert len(src_indexes.shape) == 1
    assert len(dst_indexes.shape) == 2, f"dst_indexes must be 2D [num_src, num_dst_per_src], got {dst_indexes.shape}"

    num_src = src_indexes.shape[0]
    assert (
        num_src == dst_indexes.shape[0]
    ), f"Mismatch: src_indexes {num_src} vs dst_indexes rows {dst_indexes.shape[0]}"

    # Flatten dst_indexes to 1D for kernel; kernel reconstructs the 2D layout via num_dst_per_src
    dst_indexes_flat = dst_indexes.reshape(-1).contiguous()

    src_flat = _flatten_trailing_dims(src_buffer)
    dst_flat = _flatten_trailing_dims(dst_buffer)
    _fork_mamba_buffer_1d_autotuned(src_flat, dst_flat, src_indexes, dst_indexes_flat)
