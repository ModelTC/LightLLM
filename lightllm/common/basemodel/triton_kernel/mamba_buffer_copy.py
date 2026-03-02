import torch
import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import autotune


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
):
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


def _get_copy_run_key(src_buffer: torch.Tensor):
    return 0


def _get_fork_static_key(src_buffer: torch.Tensor):
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


def _get_fork_run_key(src_buffer: torch.Tensor):
    return 0


def _flatten_trailing_dims(buffer: torch.Tensor) -> torch.Tensor:
    if buffer.ndim == 3:
        return buffer
    L, B = buffer.shape[:2]
    return buffer.view(L, B, -1)


@autotune(
    kernel_name="mamba_buffer_copy_1d:v1",
    configs_gen_func=_get_buffer_copy_configs,
    static_key_func=_get_copy_static_key,
    run_key_func=_get_copy_run_key,
)
def _copy_mamba_buffer_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
    run_config: dict = None,
):
    if not run_config:
        d_size = src_buffer.shape[2]
        BLOCK_D = min(4096, triton.next_power_of_2(d_size))
        num_warps = 4 if BLOCK_D >= 1024 else 2
        run_config = {"BLOCK_D": BLOCK_D, "num_warps": num_warps, "num_stages": 1}

    config = run_config
    BLOCK_D = config["BLOCK_D"]
    num_pairs = src_indexes.shape[0]
    layer_num = src_buffer.shape[0]
    d_size = src_buffer.shape[2]

    num_blocks_d = triton.cdiv(d_size, BLOCK_D)

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
)
def _fork_mamba_buffer_autotuned(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes_flat: torch.Tensor,
    num_dst_per_src: int,
    run_config: dict = None,
):
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


def copy_mamba_buffer(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_indexes: torch.Tensor,
    dst_indexes: torch.Tensor,
):
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
