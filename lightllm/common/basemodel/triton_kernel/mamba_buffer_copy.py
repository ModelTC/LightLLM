import torch
import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import autotune

_MAX_GRID_DIM = 65535


@triton.jit
def _copy_buffer_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    stride_layer,
    stride_slot,
    d_size,
    BLOCK_D: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    layer_idx = tl.program_id(1)
    block_d = tl.program_id(2)

    stride_layer = stride_layer.to(tl.int64)
    stride_slot = stride_slot.to(tl.int64)

    src_slot = tl.load(src_idx_ptr + pair_idx).to(tl.int64)
    dst_slot = tl.load(dst_idx_ptr + pair_idx).to(tl.int64)

    offs = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < d_size

    base = layer_idx * stride_layer
    tl.store(
        dst_ptr + base + dst_slot * stride_slot + offs,
        tl.load(src_ptr + base + src_slot * stride_slot + offs, mask=mask),
        mask=mask,
    )


@triton.jit
def _fork_buffer_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    stride_layer,
    stride_slot,
    d_size,
    num_dst_per_src,
    BLOCK_D: tl.constexpr,
):
    flat_pair = tl.program_id(0)
    layer_idx = tl.program_id(1)
    block_d = tl.program_id(2)

    src_chunk = flat_pair // num_dst_per_src

    stride_layer = stride_layer.to(tl.int64)
    stride_slot = stride_slot.to(tl.int64)

    src_slot = tl.load(src_idx_ptr + src_chunk).to(tl.int64)
    dst_slot = tl.load(dst_idx_ptr + flat_pair).to(tl.int64)

    offs = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < d_size

    base = layer_idx * stride_layer
    tl.store(
        dst_ptr + base + dst_slot * stride_slot + offs,
        tl.load(src_ptr + base + src_slot * stride_slot + offs, mask=mask),
        mask=mask,
    )


def _get_buffer_copy_configs():
    configs = []
    for block_d in [128, 256, 512, 1024, 2048, 4096]:
        for num_warps in [1, 2, 4, 8]:
            for num_stages in [1, 2]:
                configs.append({"BLOCK_D": block_d, "num_warps": num_warps, "num_stages": num_stages})
    return configs


def _get_copy_static_key(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """Static key for copy kernel cache: dtype, d_size, layer_num.

    Different models (35B vs 397B) have different optimal configs, so each
    should get its own cache file.
    """
    d_size = (
        src_buffer.shape[2]
        if src_buffer.ndim == 3
        else src_buffer.numel() // (src_buffer.shape[0] * src_buffer.shape[1])
    )
    return {
        "dtype": str(src_buffer.dtype),
        "d_size": d_size,
        "layer_num": src_buffer.shape[0],
        "ndim": src_buffer.ndim,
    }


def _get_copy_run_key(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """Run key: constant since static_key already uniquely identifies config."""
    return 0


def _get_fork_static_key(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes_flat: torch.Tensor,
    num_dst_per_src: int,
):
    """Static key for fork kernel cache: dtype, d_size, layer_num."""
    d_size = (
        src_buffer.shape[2]
        if src_buffer.ndim == 3
        else src_buffer.numel() // (src_buffer.shape[0] * src_buffer.shape[1])
    )
    return {
        "dtype": str(src_buffer.dtype),
        "d_size": d_size,
        "layer_num": src_buffer.shape[0],
        "ndim": src_buffer.ndim,
    }


def _get_fork_run_key(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes_flat: torch.Tensor,
    num_dst_per_src: int,
):
    """Run key: constant since static_key already uniquely identifies config."""
    return 0


# ─── Helper functions ─────────────────────────────────────────────────────────


def _flatten_trailing_dims(buffer: torch.Tensor) -> torch.Tensor:
    """Flatten dims after [layer_num, buffer_size] into one. Zero-copy for contiguous tensors."""
    if buffer.ndim == 3:
        return buffer
    L, B = buffer.shape[:2]
    return buffer.view(L, B, -1)


# ─── Autotuned implementations ────────────────────────────────────────────────


@autotune(
    kernel_name="mamba_buffer_copy_1d:v1",
    configs_gen_func=_get_buffer_copy_configs,
    static_key_func=_get_copy_static_key,
    run_key_func=_get_copy_run_key,
    mutates_args=["dst_buffer"],
)
def _copy_mamba_buffer_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    """Autotuned indexed copy implementation."""
    # Default heuristic when autotune is disabled or no config cached
    if not run_config:
        d_size = src_buffer.shape[2]
        # For memory-bound copy, larger BLOCK_D is better (reduces grid size)
        BLOCK_D = min(4096, triton.next_power_of_2(d_size))
        num_warps = 4 if BLOCK_D >= 1024 else 2
        run_config = {"BLOCK_D": BLOCK_D, "num_warps": num_warps, "num_stages": 1}

    config = run_config
    BLOCK_D = config["BLOCK_D"]
    num_pairs = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d_size = src_buffer.shape[2]

    num_blocks_d = triton.cdiv(d_size, BLOCK_D)

    assert num_pairs <= _MAX_GRID_DIM, f"num_pairs={num_pairs} exceeds grid limit {_MAX_GRID_DIM}"
    assert layer_num <= _MAX_GRID_DIM, f"layer_num={layer_num} exceeds grid limit {_MAX_GRID_DIM}"

    grid = (num_pairs, layer_num, num_blocks_d)
    _copy_buffer_kernel[grid](
        src_buffer,
        dst_buffer,
        src_indexes,
        dst_indexes,
        src_buffer.stride(0),
        src_buffer.stride(1),
        d_size,
        BLOCK_D=BLOCK_D,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


@autotune(
    kernel_name="mamba_buffer_fork_1d:v1",
    configs_gen_func=_get_buffer_copy_configs,
    static_key_func=_get_fork_static_key,
    run_key_func=_get_fork_run_key,
    mutates_args=["dst_buffer"],
)
def _fork_mamba_buffer_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes_flat: torch.Tensor,
    num_dst_per_src: int,
    run_config: dict = None,
):
    """Autotuned fork implementation."""
    # Default heuristic when autotune is disabled or no config cached
    if not run_config:
        d_size = src_buffer.shape[2]
        BLOCK_D = min(4096, triton.next_power_of_2(d_size))
        num_warps = 4 if BLOCK_D >= 1024 else 2
        run_config = {"BLOCK_D": BLOCK_D, "num_warps": num_warps, "num_stages": 1}

    config = run_config
    BLOCK_D = config["BLOCK_D"]
    num_src = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d_size = src_buffer.shape[2]

    num_blocks_d = triton.cdiv(d_size, BLOCK_D)
    total_pairs = num_src * num_dst_per_src

    assert total_pairs <= _MAX_GRID_DIM, f"total_pairs={total_pairs} exceeds grid limit {_MAX_GRID_DIM}"
    assert layer_num <= _MAX_GRID_DIM, f"layer_num={layer_num} exceeds grid limit {_MAX_GRID_DIM}"

    grid = (total_pairs, layer_num, num_blocks_d)
    _fork_buffer_kernel[grid](
        src_buffer,
        dst_buffer,
        src_indexes,
        dst_indexes_flat,
        src_buffer.stride(0),
        src_buffer.stride(1),
        d_size,
        num_dst_per_src,
        BLOCK_D=BLOCK_D,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


# ─── Public API ───────────────────────────────────────────────────────────────


def copy_mamba_buffer(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """
    Indexed 1:1 copy of Mamba recurrent state buffer slots.

    Copies slot src_indexes[i] -> dst_indexes[i] for all layers simultaneously.

    Args:
        src_buffer: [layer_num, num_slots, ...]
        dst_buffer: [layer_num, num_slots, ...]
        src_indexes: source slot indices [num_pairs]
        dst_indexes: destination slot indices [num_pairs]
    """
    assert src_buffer.shape == dst_buffer.shape
    assert src_indexes.shape == dst_indexes.shape and src_indexes.ndim == 1

    src_flat = _flatten_trailing_dims(src_buffer)
    dst_flat = _flatten_trailing_dims(dst_buffer)
    _copy_mamba_buffer_autotuned(src_flat, dst_flat, src_indexes, dst_indexes)


def fork_mamba_buffer(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
    """
    Fork Mamba recurrent state slots: one source -> N destinations.

    Used for MTP speculation where parent state is replicated to child slots.

    Args:
        src_buffer: [layer_num, num_slots, ...]
        dst_buffer: [layer_num, num_slots, ...]
        src_indexes: source slot indices [num_src]
        dst_indexes: destination slot indices [num_src, num_dst_per_src]
    """
    assert src_buffer.shape == dst_buffer.shape
    assert src_indexes.ndim == 1
    assert dst_indexes.ndim == 2, f"dst_indexes must be 2D [num_src, num_dst_per_src], got {dst_indexes.shape}"
    assert (
        dst_indexes.shape[0] == src_indexes.shape[0]
    ), f"Mismatch: src_indexes {src_indexes.shape[0]} vs dst_indexes rows {dst_indexes.shape[0]}"

    num_dst_per_src = dst_indexes.shape[1]
    dst_indexes_flat = dst_indexes.reshape(-1).contiguous()

    src_flat = _flatten_trailing_dims(src_buffer)
    dst_flat = _flatten_trailing_dims(dst_buffer)
    _fork_mamba_buffer_autotuned(src_flat, dst_flat, src_indexes, dst_indexes_flat, num_dst_per_src)
