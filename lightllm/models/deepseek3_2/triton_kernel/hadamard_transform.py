import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _butterfly_stage(x, GROUPS: tl.constexpr, STEP: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    x_grouped = tl.reshape(x, (BLOCK_R, GROUPS, 2, STEP))
    x_grouped = tl.permute(x_grouped, (0, 1, 3, 2))
    left, right = tl.split(x_grouped)
    x_pair = tl.join(left + right, left - right)
    x_pair = tl.permute(x_pair, (0, 1, 3, 2))
    return tl.reshape(x_pair, (BLOCK_R, BLOCK_N))


@triton.jit
def _hadamard_transform_kernel(
    X,
    Y,
    n_rows,
    scale: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = rows[:, None] < n_rows
    offsets = rows[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)

    x = _butterfly_stage(x, 64, 1, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 32, 2, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 16, 4, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 8, 8, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 4, 16, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 2, 32, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 1, 64, BLOCK_R, BLOCK_N)

    tl.store(Y + offsets, x * scale, mask=mask)


@functools.lru_cache(maxsize=None)
def _target_programs(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count * 2


def _pick_block_r(rows: int, device_index: int) -> int:
    block_r = triton.next_power_of_2(max(1, rows // _target_programs(device_index)))
    return max(1, min(128, block_r))


def _hadamard_transform_triton(x: torch.Tensor, scale: float) -> torch.Tensor:
    original_shape = x.shape
    hidden_size = x.size(-1)
    if not x.is_contiguous():
        x = x.contiguous()
    rows = x.numel() // hidden_size
    out = torch.empty_like(x)
    BLOCK_R = _pick_block_r(rows, x.device.index)
    grid = (triton.cdiv(rows, BLOCK_R),)
    _hadamard_transform_kernel[grid](
        x,
        out,
        rows,
        scale,
        BLOCK_R=BLOCK_R,
        BLOCK_N=hidden_size,
        num_warps=4,
    )
    return out.view(original_shape)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    assert x.is_cuda, "hadamard_transform only supports CUDA tensors"
    assert x.dtype == torch.bfloat16, "hadamard_transform expects bfloat16 input"
    assert x.size(-1) == 128, "DeepSeek-V3.2 Hadamard transform expects hidden size 128"

    return _hadamard_transform_triton(x, scale)
