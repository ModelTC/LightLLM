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

    # Cast strides to int64 to prevent overflow in pointer arithmetic
    stride_layer = stride_layer.to(tl.int64)
    stride_index = stride_index.to(tl.int64)

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

    # Cast strides to int64 to prevent overflow in pointer arithmetic
    stride_layer = stride_layer.to(tl.int64)
    stride_index = stride_index.to(tl.int64)

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


def copy_buffer_p2p(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """
    Copy buffers from source indices to destination indices with auto-tuning.

    Supports any buffer shape [layer_num, buffer_size, ...] as long as the
    trailing dimensions are contiguous (which is the default for torch.zeros).

    Args:
        src_buffer: Source buffer tensor [layer_num, buffer_size, ...]
        dst_buffer: Destination buffer tensor [layer_num, buffer_size, ...]
        src_indexes: Source buffer indices [num_pairs]
        dst_indexes: Destination buffer indices [num_pairs]
    """
    assert src_buffer.shape == dst_buffer.shape
    assert src_indexes.shape == dst_indexes.shape
    assert len(src_indexes.shape) == 1

    src_flat = _flatten_trailing_dims(src_buffer)
    dst_flat = _flatten_trailing_dims(dst_buffer)
    _copy_buffer_p2p_1d_autotuned(src_flat, dst_flat, src_indexes, dst_indexes)


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

    src_flat = _flatten_trailing_dims(src_buffer)
    dst_flat = _flatten_trailing_dims(dst_buffer)
    _copy_buffer_broadcast_1d_autotuned(src_flat, dst_flat, src_indexes, dst_indexes_flat)
