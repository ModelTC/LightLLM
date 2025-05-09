import torch
import triton
import triton.language as tl
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


@triton.jit
def gelu(x):
    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + tl.math.erf(x_fp32 * 0.7071067811))
    return x_gelu


@triton.jit
def gelu_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = gelu(input)
    tl.store(output_ptr + offsets, output, mask=mask)


def gelu_fwd(input, use_custom_tensor_mananger=False):
    if use_custom_tensor_mananger:
        shape = input.shape
        dtype = input.dtype
        device = input.device
        output = g_cache_manager.alloc_tensor(shape, dtype, device=device)
    else:
        output = torch.empty_like(input)
    assert input.is_contiguous(), "Input tensor must be contiguous"
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](output, input, n_elements, BLOCK_SIZE=1024)
    return output
