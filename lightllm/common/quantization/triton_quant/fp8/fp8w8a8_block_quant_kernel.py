import torch
import triton
import triton.language as tl
from lightllm.utils.dist_utils import get_current_device_id


@triton.jit
def weight_quant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x))

    max_fp8e4m3_val = 448.0
    scale = amax / max_fp8e4m3_val
    y = (x / (scale + 1e-6)).to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n_blocks + pid_n, scale)


def mm_weight_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.size()

    y_quant = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)

    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    s_scales = torch.empty((num_blocks_m, num_blocks_n), dtype=torch.float32, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    weight_quant_kernel[grid](x, s_scales, y_quant, M, N, BLOCK_SIZE=block_size)
    return y_quant, s_scales


def weight_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    x = x.cuda(get_current_device_id())
    if x.dim() == 3:
        y_quant = torch.empty((x.shape[0], x.shape[1], x.shape[2]), dtype=torch.float8_e4m3fn, device=x.device)
        num_blocks_m = triton.cdiv(x.shape[1], block_size)
        num_blocks_n = triton.cdiv(x.shape[2], block_size)
        s_scales = torch.empty((x.shape[0], num_blocks_m, num_blocks_n), dtype=torch.float32, device=x.device)
        for i in range(x.shape[0]):
            y_quant[i], s_scales[i] = mm_weight_quant(x[i], block_size)
        return y_quant, s_scales
    else:
        y_quant, s_scales = mm_weight_quant(x, block_size)
        return y_quant, s_scales
