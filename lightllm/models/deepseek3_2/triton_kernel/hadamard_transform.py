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


@triton.jit
def _hadamard_transform_quant_fp8_kernel(
    X,
    Y,
    S,
    n_rows,
    scale: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < n_rows
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * BLOCK_N + cols[None, :]
    x = tl.load(X + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)

    x = _butterfly_stage(x, 64, 1, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 32, 2, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 16, 4, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 8, 8, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 4, 16, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 2, 32, BLOCK_R, BLOCK_N)
    x = _butterfly_stage(x, 1, 64, BLOCK_R, BLOCK_N)

    # Match the unfused path's bf16 Hadamard output before FP8 quantization.
    x = (x * scale).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    quant_scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / quant_scale[:, None], -448.0), 448.0)

    tl.store(Y + offsets, y, mask=row_mask[:, None])
    tl.store(S + rows, quant_scale, mask=row_mask)


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


def hadamard_transform_quant_fp8(x: torch.Tensor, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Hadamard-128 with the following ue8m0 FP8 quantization."""

    assert x.is_cuda, "hadamard_transform_quant_fp8 only supports CUDA tensors"
    assert x.dtype == torch.bfloat16, "Hadamard transform expects bfloat16 input"
    assert x.size(-1) == 128, "Hadamard transform expects hidden size 128"
    if not x.is_contiguous():
        x = x.contiguous()

    original_shape = x.shape
    rows = x.numel() // 128
    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty((*original_shape[:-1], 1), dtype=torch.float32, device=x.device)
    if rows == 0:
        return output, output_scale

    block_r = 32
    _hadamard_transform_quant_fp8_kernel[(triton.cdiv(rows, block_r),)](
        x,
        output,
        output_scale,
        rows,
        scale,
        BLOCK_R=block_r,
        BLOCK_N=128,
        num_warps=2,
    )
    return output.view(original_shape), output_scale
