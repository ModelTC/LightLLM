import torch

import triton
import triton.language as tl
from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu


@triton.jit
def _silu_and_mul_group_quant_kernel(
    input_ptr,
    stride_input_m,
    out_q_ptr,
    stride_out_q_m,
    out_s_ptr,
    stride_out_s_m,
    size_n,  # width of the silu output (= down-proj K), a multiple of GROUP_SIZE
    limit,
    alpha,
    fp8_min,
    fp8_max,
    eps,
    GROUP_SIZE: tl.constexpr,
    layout: tl.constexpr,  # "blocked" or "interleaved"
    USE_LIMIT_AND_ALPHA: tl.constexpr,
    USE_TANH_APPROXIMATE_GELU: tl.constexpr,
):
    # One program per (token row, quant group of GROUP_SIZE columns). Computes silu(gate)*up for
    # the group, then a per-group fp8 quant — byte-identical layout to
    # silu_and_mul_fwd followed by per_token_group_quant_fp8 (row-major scales), in one launch.
    m_index = tl.program_id(0).to(tl.int64)
    group_index = tl.program_id(1)
    cols = group_index * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    if layout == "interleaved":
        # [gate0, up0, gate1, up1, ...]
        gate_off = m_index * stride_input_m + cols * 2
        up_off = gate_off + 1
    else:
        # [gate0, gate1, ..., up0, up1, ...]
        gate_off = m_index * stride_input_m + cols
        up_off = m_index * stride_input_m + cols + size_n
    gate = tl.load(input_ptr + gate_off).to(tl.float32)
    up = tl.load(input_ptr + up_off)

    if USE_LIMIT_AND_ALPHA:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        gate = 1 / (1 + tl.exp(-gate * alpha)) * gate
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = (up + 1) * gate
    else:
        if USE_TANH_APPROXIMATE_GELU:
            gate_cubed = gate * gate * gate
            tanh_arg = 0.7978845608028654 * (gate + 0.044715 * gate_cubed)
            tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
            gate = 0.5 * gate * (1.0 + tanh_val)
        else:
            gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate

    # quantize the (bf16-rounded) silu output per group, matching per_token_group_quant_fp8.
    gate_up_f = gate_up.to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(gate_up_f)), eps)
    out_s = _absmax / fp8_max
    out_q = tl.clamp(gate_up_f / out_s, fp8_min, fp8_max).to(out_q_ptr.dtype.element_ty)
    tl.store(out_q_ptr + m_index * stride_out_q_m + cols, out_q)
    tl.store(out_s_ptr + m_index * stride_out_s_m + group_index, out_s)


def silu_and_mul_group_quant_fwd(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    layout: str = "blocked",
    limit=None,
    alpha=None,
):
    """Fused silu_and_mul + per-token-group fp8 quant.

    ``input`` [M, 2*N] (gate|up) -> ``output_q`` [M, N] fp8 + ``output_s`` [M, N//group_size]
    float32 row-major. Equivalent to ``silu_and_mul_fwd(input, tmp); per_token_group_quant_fp8(tmp)``
    but in one kernel, so the down-projection's activation quant disappears as a separate launch.
    """
    assert input.is_contiguous()
    assert output_q.is_contiguous() and output_q.dtype == torch.float8_e4m3fn
    assert (limit is None and alpha is None) or (limit is not None and alpha is not None)
    size_m = input.shape[0]
    size_n = input.shape[-1] // 2
    assert size_n % group_size == 0, f"silu output width {size_n} not divisible by group_size {group_size}"
    assert output_q.shape[0] == size_m and output_q.shape[1] == size_n
    assert output_s.shape[0] == size_m and output_s.shape[1] == size_n // group_size

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    # grid: token rows on dim-0 (up to 2**31, may be large for prefill), groups on dim-1 (small).
    grid = (size_m, size_n // group_size)
    _silu_and_mul_group_quant_kernel[grid](
        input,
        input.stride(0),
        output_q,
        output_q.stride(0),
        output_s,
        output_s.stride(0),
        size_n,
        limit if limit is not None else 0.0,
        alpha if alpha is not None else 0.0,
        fp8_min,
        fp8_max,
        1e-10,
        GROUP_SIZE=group_size,
        layout=layout,
        USE_LIMIT_AND_ALPHA=limit is not None and alpha is not None,
        USE_TANH_APPROXIMATE_GELU=ffn_use_tanh_approximate_gelu(),
        num_warps=1,
    )
    return
