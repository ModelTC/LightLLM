import torch
import triton
import triton.language as tl
from lightllm.utils.sgl_utils import sgl_ops


@triton.jit
def load_awq_tile(
    weight,
    scales,
    zeros,
    rows,
    cols,
    valid_rows,
    packed_stride,
    scale_stride,
    zero_stride,
    group_size,
    base_row=None,
    base_col=None,
):
    if base_row is not None and base_col is not None:
        block_rows: tl.constexpr = rows.shape[0]
        block_cols: tl.constexpr = cols.shape[0]
        packed_col = base_col // 8 + tl.arange(0, block_cols // 8)
        nibbles = tl.arange(0, 8)
        shift = (nibbles // 2 + (nibbles % 2) * 4) * 4
        packed = tl.load(weight + rows[:, None] * packed_stride + packed_col[None, :], valid_rows[:, None], 0)
        group = base_row // group_size
        zero = tl.load(zeros + group * zero_stride + packed_col, tl.sum(valid_rows.to(tl.int32), 0) != 0, 0)
        scale = tl.load(scales + group * scale_stride + cols, tl.sum(valid_rows.to(tl.int32), 0) != 0, 0)
        unpacked = tl.reshape((packed[:, :, None] >> shift[None, None, :]) & 15, (block_rows, block_cols))
        zero = tl.reshape((zero[:, None] >> shift[None, :]) & 15, (block_cols,))
        return ((unpacked - zero[None, :]).to(tl.float32) * scale[None, :]).to(scale.dtype)
    groups = rows // group_size
    pack_cols = cols // 8
    nibble = cols % 8
    shift = (nibble // 2 + (nibble % 2) * 4) * 4
    packed = tl.load(weight + rows[:, None] * packed_stride + pack_cols[None, :], valid_rows[:, None], 0)
    zero = tl.load(zeros + groups[:, None] * zero_stride + pack_cols[None, :], valid_rows[:, None], 0)
    scale = tl.load(scales + groups[:, None] * scale_stride + cols[None, :], valid_rows[:, None], 0)
    quantized = (packed >> shift[None, :]) & 15
    zero = (zero >> shift[None, :]) & 15
    return ((quantized - zero).to(tl.float32) * scale).to(scale.dtype)


@triton.jit
def _awq_gemm(
    inputs,
    weight,
    scales,
    zeros,
    out,
    tokens,
    k: tl.constexpr,
    n: tl.constexpr,
    input_stride,
    out_stride,
    weight_stride: tl.constexpr,
    scale_stride: tl.constexpr,
    zero_stride: tl.constexpr,
    group_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    split_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    safe_cols = cols if n % block_n == 0 else cols % n
    inner = tl.arange(0, block_k)
    accum = tl.zeros((block_m, block_n), tl.float32)
    split = tl.program_id(2)
    span: tl.constexpr = triton.cdiv(k, block_k * split_k) * block_k
    for offset in tl.range(0, span, block_k, num_stages=3):
        start = split * span + offset
        indices = start + inner
        activations = tl.load(
            inputs + rows[:, None] * input_stride + indices[None, :],
            (rows[:, None] < tokens) & (indices[None, :] < k),
            0,
        )
        if n % block_n == 0 and group_size % block_k == 0 and k % block_k == 0:
            weights = load_awq_tile(
                weight,
                scales,
                zeros,
                indices,
                safe_cols,
                indices < k,
                weight_stride,
                scale_stride,
                zero_stride,
                group_size,
                start,
                tl.program_id(1) * block_n,
            )
        else:
            weights = load_awq_tile(
                weight,
                scales,
                zeros,
                indices,
                safe_cols,
                indices < k,
                weight_stride,
                scale_stride,
                zero_stride,
                group_size,
            )
        accum = tl.dot(activations, weights.to(activations.dtype), accum)
    tl.store(
        out + split * tokens * out_stride + rows[:, None] * out_stride + cols[None, :],
        accum,
        (rows[:, None] < tokens) & (cols[None, :] < n),
    )


@triton.jit
def _dequantize_awq(
    weight,
    scales,
    zeros,
    output,
    k: tl.constexpr,
    n: tl.constexpr,
    weight_stride: tl.constexpr,
    scale_stride: tl.constexpr,
    zero_stride: tl.constexpr,
    group_size: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    values = load_awq_tile(
        weight, scales, zeros, rows, cols % n, rows < k, weight_stride, scale_stride, zero_stride, group_size
    )
    tl.store(output + rows[:, None] * n + cols[None, :], values, (rows[:, None] < k) & (cols[None, :] < n))


def dequantize_awq(weight, scales, zeros):
    width = scales.shape[-1]
    height = weight.shape[0]
    group_size = height // scales.shape[0]
    if (
        hasattr(sgl_ops, "awq_dequantize")
        and scales.dtype == torch.float16
        and group_size in (32, 64, 128, height)
        and width % 64 == 0
        and weight.is_contiguous()
        and scales.is_contiguous()
        and zeros.is_contiguous()
    ):
        return sgl_ops.awq_dequantize(weight, scales, zeros)
    output = torch.empty((height, width), device=scales.device, dtype=scales.dtype)
    _dequantize_awq[(triton.cdiv(height, 16), triton.cdiv(width, 128))](
        weight, scales, zeros, output, height, width, weight.stride(0), scales.stride(0), zeros.stride(0), group_size
    )
    return output


@triton.jit
def _awq_gemv(
    inputs,
    weight,
    scales,
    zeros,
    partial,
    height: tl.constexpr,
    width: tl.constexpr,
    weight_stride: tl.constexpr,
    scale_stride: tl.constexpr,
    zero_stride: tl.constexpr,
    group_size: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    base_row = tl.program_id(0) * block_rows
    base_col = tl.program_id(1) * block_cols
    tile_rows = base_row + tl.arange(0, block_rows)
    tile_cols = base_col + tl.arange(0, block_cols)
    safe_cols = tile_cols if width % block_cols == 0 else tile_cols % width
    activations = tl.load(inputs + tile_rows, tile_rows < height, 0).to(tl.float32)
    if width % block_cols == 0 and group_size % block_rows == 0 and height % block_rows == 0:
        values = load_awq_tile(
            weight,
            scales,
            zeros,
            tile_rows,
            safe_cols,
            tile_rows < height,
            weight_stride,
            scale_stride,
            zero_stride,
            group_size,
            base_row,
            base_col,
        )
    else:
        values = load_awq_tile(
            weight,
            scales,
            zeros,
            tile_rows,
            safe_cols,
            tile_rows < height,
            weight_stride,
            scale_stride,
            zero_stride,
            group_size,
        )
    products = activations[:, None] * values.to(tl.float32)
    tl.store(partial + tl.program_id(0) * width + tile_cols, tl.sum(products, axis=0), tile_cols < width)


def awq_gemm(inputs, weight, scales, zeros, out=None):
    n = scales.shape[-1]
    matrix = inputs.reshape(-1, inputs.shape[-1]).contiguous()
    result = out if out is not None else torch.empty((*inputs.shape[:-1], n), dtype=inputs.dtype, device=inputs.device)
    flat = result.view(-1, n)
    if matrix.shape[0] == 0:
        return result
    if matrix.shape[0] == 1 and matrix.shape[1] >= 1024:
        height = matrix.shape[1]
        group_size = height // scales.shape[0]
        block_rows, block_cols = (64, 128) if n % 128 == 0 and group_size % 64 == 0 else (128, 64)
        blocks = triton.cdiv(height, block_rows)
        partial = torch.empty((blocks, n), device=matrix.device, dtype=torch.float32)
        _awq_gemv[(blocks, triton.cdiv(n, block_cols))](
            matrix,
            weight,
            scales,
            zeros,
            partial,
            height,
            n,
            weight.stride(0),
            scales.stride(0),
            zeros.stride(0),
            group_size,
            block_rows,
            block_cols,
        )
        torch.sum(partial, dim=0, out=flat[0])
        return result
    if matrix.shape[0] >= 32:
        unpacked = dequantize_awq(weight, scales, zeros).to(matrix.dtype)
        torch.mm(matrix, unpacked, out=flat)
        return result
    split_k = 4 if matrix.shape[1] >= 1024 else 1
    partial = (
        flat if split_k == 1 else torch.empty((split_k, matrix.shape[0], n), device=matrix.device, dtype=torch.float32)
    )
    _awq_gemm[(triton.cdiv(matrix.shape[0], 16), triton.cdiv(n, 64), split_k)](
        matrix,
        weight,
        scales,
        zeros,
        partial,
        matrix.shape[0],
        matrix.shape[1],
        n,
        matrix.stride(0),
        partial.stride(-2),
        weight.stride(0),
        scales.stride(0),
        zeros.stride(0),
        matrix.shape[1] // scales.shape[0],
        16,
        64,
        64,
        split_k,
    )
    if split_k > 1:
        torch.sum(partial, dim=0, out=flat)
    return result
