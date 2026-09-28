import torch
import triton
import triton.language as tl
from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import _butterfly_stage


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
