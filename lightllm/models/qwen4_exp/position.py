import torch
import triton
import triton.language as tl


@triton.jit
def _combine_mrope(
    Cos,
    Sin,
    OutCos,
    OutSin,
    ROWS: tl.constexpr,
    DIM: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    AXES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // (2 * DIM), index % DIM
    axis = tl.full((BLOCK,), 0, tl.int32)
    if AXES:
        height = (col % 3 == 1) & (col <= 3 * HEIGHT)
        width = (col % 3 == 2) & (col <= 3 * WIDTH)
        axis = tl.where(height, 1, tl.where(width, 2, 0))
    cosine = tl.load(Cos + axis * C0 + row * C1 + col * C2, row < ROWS, 0)
    sine = tl.load(Sin + axis * S0 + row * S1 + col * S2, row < ROWS, 0)
    tl.store(OutCos + index, cosine, row < ROWS)
    tl.store(OutSin + index, sine, row < ROWS)


def combine_mrope(cos: torch.Tensor, sin: torch.Tensor, sections):
    if cos.shape != sin.shape or cos.device != sin.device or cos.dtype != sin.dtype:
        raise ValueError("Rotary cosine and sine shapes, devices and dtypes must match")
    if cos.ndim != 2 and (cos.ndim != 3 or cos.shape[0] != 3 or len(sections) != 3):
        raise ValueError("Expected three rotary axes and three mRoPE sections")
    if cos.is_cuda:
        rows, dim = cos.shape[-2:]
        out_cos = torch.empty((rows, 2 * dim), dtype=cos.dtype, device=cos.device)
        out_sin = torch.empty_like(out_cos)
        axes = cos.ndim == 3
        if rows and dim:
            _combine_mrope[(triton.cdiv(rows * 2 * dim, 256),)](
                cos,
                sin,
                out_cos,
                out_sin,
                rows,
                dim,
                *(cos.stride() if axes else (0, *cos.stride())),
                *(sin.stride() if axes else (0, *sin.stride())),
                sections[1] if axes else 0,
                sections[2] if axes else 0,
                axes,
                256,
            )
        return out_cos, out_sin
    if cos.ndim == 2:
        return cos.repeat(1, 2).contiguous(), sin.repeat(1, 2).contiguous()
    offsets = torch.arange(cos.shape[-1], device=cos.device)
    height = ((offsets % 3) == 1) & (offsets <= 3 * sections[1])
    width = ((offsets % 3) == 2) & (offsets <= 3 * sections[2])
    combined = []
    for tensor in (cos, sin):
        values = torch.where(height, tensor[1], torch.where(width, tensor[2], tensor[0]))
        combined.append(values.repeat(1, 2).contiguous())
    return tuple(combined)
