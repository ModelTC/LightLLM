import torch

import triton
import triton.language as tl


@triton.jit
def _add_in_place(
    input_ptr,
    other_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    x = x + y * alpha
    tl.store(input_ptr + offsets, x, mask=mask)


@torch.no_grad()
def add_in_place(input: torch.Tensor, other: torch.Tensor, *, alpha=1):
    assert input.is_contiguous(), "input tensor must be contiguous"
    assert other.is_contiguous(), "other tensor must be contiguous"
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_in_place[grid](
        input,
        other,
        n_elements,
        alpha,
        BLOCK_SIZE=1024,
    )
    return input
