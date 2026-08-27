"""Tiled transpose kernels used by the post-layer logits path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_convert_2d_kernel(
    input_ptr,
    output_ptr,
    rows,
    cols,
    input_stride_0,
    input_stride_1,
    output_stride_0,
    output_stride_1,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    input_offsets = row_offsets[:, None] * input_stride_0 + col_offsets[None, :] * input_stride_1
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    values = tl.load(input_ptr + input_offsets, mask=mask)

    output_offsets = col_offsets[:, None] * output_stride_0 + row_offsets[None, :] * output_stride_1
    tl.store(output_ptr + output_offsets, tl.trans(values), mask=tl.trans(mask))


@torch.no_grad()
def transpose_convert_2d(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    block_rows: int = 64,
    block_cols: int = 64,
    num_warps: int = 8,
    num_stages: int = 1,
) -> torch.Tensor:
    """Transpose a contiguous 2-D CUDA tensor while converting its dtype."""

    assert input_tensor.is_cuda and output_tensor.is_cuda
    assert input_tensor.device == output_tensor.device
    assert input_tensor.ndim == 2 and output_tensor.ndim == 2
    assert output_tensor.shape == (input_tensor.shape[1], input_tensor.shape[0])
    assert input_tensor.is_contiguous() and output_tensor.is_contiguous()

    rows, cols = input_tensor.shape
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(cols, block_cols))
    _transpose_convert_2d_kernel[grid](
        input_tensor,
        output_tensor,
        rows,
        cols,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output_tensor
