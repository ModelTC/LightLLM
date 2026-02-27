import torch
import triton
import triton.language as tl
from lightllm.utils.dist_utils import get_current_device_id


@triton.jit
def weight_quant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_N: tl.constexpr):
    m_index = tl.program_id(axis=0)

    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x = tl.load(x_ptr + m_index * N + offs_n, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x))

    max_fp8e4m3_val = 448.0
    scale = amax / max_fp8e4m3_val
    y = (x / (scale + 1e-6)).to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + m_index * N + offs_n, y, mask=mask)
    tl.store(s_ptr + m_index, scale)


def mm_weight_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    M, N = x.size()

    y_quant = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    s_scales = torch.empty((M, 1), dtype=torch.float32, device=x.device)

    grid = (M,)
    weight_quant_kernel[grid](x, s_scales, y_quant, M, N, BLOCK_N=triton.next_power_of_2(N), num_warps=16)
    return y_quant, s_scales


def weight_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    x = x.cuda(get_current_device_id())
    if x.dim() == 3:
        y_quant = torch.empty((x.shape[0], x.shape[1], x.shape[2]), dtype=torch.float8_e4m3fn, device=x.device)
        s_scales = torch.empty((x.shape[0], x.shape[1], 1), dtype=torch.float32, device=x.device)
        for i in range(x.shape[0]):
            y_quant[i], s_scales[i] = mm_weight_quant(x[i])
        return y_quant, s_scales
    else:
        y_quant, s_scales = mm_weight_quant(x)
        return y_quant, s_scales
