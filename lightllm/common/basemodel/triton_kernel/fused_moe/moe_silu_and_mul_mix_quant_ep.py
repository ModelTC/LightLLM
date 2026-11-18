import torch

import triton
import triton.language as tl

from lightllm.common.triton_utils.autotuner import AutotuneKernelType, autotune
from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import _ceil_to_ue8m0


@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    USE_LIMIT_ONLY: tl.constexpr = False,
    USE_TANH_APPROXIMATE_GELU: tl.constexpr = False,
    USE_LIMIT_AND_ALPHA: tl.constexpr = False,
    alpha: tl.constexpr = None,
    limit: tl.constexpr = None,
    CLAMP_UP_ADD_ONE: tl.constexpr = True,
    EXPERT_ALIGNMENT: tl.constexpr = None,
    USE_UE8M0_SCALES: tl.constexpr = False,
    BLOCK_M: tl.constexpr = 1,
    GROUPS: tl.constexpr = 1,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    if EXPERT_ALIGNMENT is None:
        start = 0
        end = tl.load(masked_m_ptr + expert_id)
    else:
        previous = tl.load(masked_m_ptr + expert_id - 1, expert_id > 0, 0)
        start = tl.cdiv(previous, EXPERT_ALIGNMENT) * EXPERT_ALIGNMENT
        end = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d[None, :]
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d[None, :]
    group_offsets = hidden_dim_block_index * GROUPS + tl.arange(0, GROUPS)
    output_scale_offs = (
        output_scale_ptr + expert_id * stride_output_scale_0 + group_offsets[None, :] * stride_output_scale_2
    )

    for first in tl.range(start + token_id * BLOCK_M, end, block_num_per_expert * BLOCK_M, num_stages=NUM_STAGE):
        token_index = first + tl.arange(0, BLOCK_M)
        row_mask = (BLOCK_M == 1) | (token_index[:, None] < end)
        mask = row_mask & (offs_in_d[None, :] < size_n)
        offsets = token_index.to(tl.int64)[:, None] * stride_input_1
        gate = tl.load(input_ptr_offs + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(input_ptr_offs + offsets + size_n, mask=mask, other=0.0)
        if USE_LIMIT_ONLY:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
        if USE_LIMIT_AND_ALPHA:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
            gate = gate / (1 + tl.exp(-gate * alpha))
            if CLAMP_UP_ADD_ONE:
                up += 1
        elif USE_TANH_APPROXIMATE_GELU:
            gate_cubed = gate * gate * gate
            tanh_arg = 0.7978845608028654 * (gate + 0.044715 * gate_cubed)
            tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
            gate = 0.5 * gate * (1.0 + tanh_val)
        else:
            gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        if USE_LIMIT_AND_ALPHA:
            # Match the BF16/FP16 activation stored before prefill's FP8 quantization.
            gate_up = gate_up.to(input_ptr.dtype.element_ty).to(tl.float32)
        grouped = tl.reshape(gate_up, (BLOCK_M, GROUPS, BLOCK_N // GROUPS))
        _absmax = tl.maximum(tl.max(tl.abs(grouped), 2), 1e-10)
        output_s = _absmax / fp8_max
        if USE_UE8M0_SCALES:
            output_s = _ceil_to_ue8m0(output_s)
        output_q = tl.reshape(tl.clamp(grouped / output_s[:, :, None], fp8_min, fp8_max), (BLOCK_M, BLOCK_N))
        tl.store(
            output_ptr_offs + token_index.to(tl.int64)[:, None] * stride_output_1,
            output_q,
            mask=mask,
        )
        tl.store(
            output_scale_offs + token_index.to(tl.int64)[:, None] * stride_output_scale_1,
            output_s,
            mask=row_mask & ((GROUPS == 1) | (group_offsets[None, :] < size_n // (BLOCK_N // GROUPS))),
        )


@triton.jit
def _expanded_swiglu_quant_1x32_kernel(
    x,
    quant,
    scales,
    expert_ends,
    width: tl.constexpr,
    stride_sm,
    stride_sn,
    limit: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS: tl.constexpr,
):
    expert = tl.program_id(2)
    previous = tl.load(expert_ends + expert - 1, expert > 0, 0)
    start = tl.cdiv(previous, 32) * 32
    end = tl.load(expert_ends + expert)
    cols = tl.program_id(0) * (GROUPS * 32) + tl.arange(0, GROUPS * 32)
    groups = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    for first in range(start + tl.program_id(1) * BLOCK_M, end, tl.num_programs(1) * BLOCK_M):
        rows = first + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < end) & (cols[None, :] < width)
        row_offsets = rows.to(tl.int64)[:, None]
        offsets = row_offsets * (2 * width) + cols[None, :]
        gate = tl.load(x + offsets, mask, 0).to(tl.float32)
        up = tl.load(x + offsets + width, mask, 0).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.clamp(up, -limit, limit)
        gate = (gate / (1 + tl.exp(-gate))).to(x.dtype.element_ty)
        values = tl.reshape(up * gate, (BLOCK_M, GROUPS, 32))
        scale = _ceil_to_ue8m0(tl.maximum(tl.max(tl.abs(values), 2), 1e-10) / 448.0)
        q = tl.reshape(tl.clamp(values / scale[:, :, None], -448.0, 448.0), (BLOCK_M, GROUPS * 32))
        tl.store(quant + row_offsets * width + cols[None, :], q, mask)
        tl.store(
            scales + row_offsets * stride_sm + groups[None, :] * stride_sn,
            scale,
            (rows[:, None] < end) & (groups[None, :] < width // 32),
        )


def _expanded_swiglu_tuning_inputs(x, quant, scales, expert_ends, expected_m, limit):
    # Capture warmup can send every dummy row to the same expert. Use separate
    # buffers so tuning never populates padding in the caller's expanded layout.
    # Keep the mean of the graph bucket while exercising different loop and tail lengths.
    experts = expert_ends.numel()
    light_m = expected_m // 2
    counts = [light_m, expected_m, 2 * expected_m - light_m] * (experts // 3) + [expected_m] * (experts % 3)
    ends, rows = [], 0
    for count in counts:
        ends.append(rows + count)
        rows += triton.cdiv(count, 32) * 32
    if scales.stride(0) == 1:
        scale_strides = (1, max(rows, scales.stride(1)))
    else:
        scale_strides = (scales.shape[1], 1)
    return (
        x.new_zeros((rows, x.shape[1])),
        quant.new_empty((rows, quant.shape[1])),
        torch.empty_strided((rows, scales.shape[1]), scale_strides, dtype=scales.dtype, device=x.device),
        expert_ends.new_tensor(ends),
        expected_m,
        limit,
    ), {}


@autotune(
    kernel_name="expanded_swiglu_quant_1x32:v1",
    configs_gen_func=lambda: [
        {"BLOCK_M": m, "GROUPS": g, "row_programs": r, "num_warps": w}
        for m, g, r, w in (
            (1, 4, 4, 1),
            (2, 4, 8, 4),
            (4, 4, 8, 4),
            (4, 8, 8, 4),
            (4, 8, 4, 4),
            (8, 4, 4, 4),
            (8, 8, 4, 4),
            (4, 16, 8, 4),
        )
    ],
    static_key_func=lambda x, scales, expert_ends: {
        "n": x.shape[1] // 2,
        "experts": expert_ends.numel(),
        "dtype": str(x.dtype),
        "column_major_scales": scales.stride(0) == 1,
    },
    run_key_func=lambda expected_m: expected_m,
    kernel_type=AutotuneKernelType.DECODE,
    rebuild_input_func=_expanded_swiglu_tuning_inputs,
    warmup_all_exist_config=False,
    benchmark_repeats=32,
)
def _expanded_swiglu_quant_1x32(x, quant, scales, expert_ends, expected_m, limit, run_config=None):
    if run_config is None:
        run_config = {"BLOCK_M": 4, "GROUPS": 8, "row_programs": 8, "num_warps": 4}
    width = quant.shape[1]
    groups = run_config["GROUPS"]
    grid = (triton.cdiv(width, groups * 32), run_config["row_programs"], expert_ends.numel())
    _expanded_swiglu_quant_1x32_kernel[grid](
        x,
        quant,
        scales,
        expert_ends,
        width,
        *scales.stride(),
        limit,
        BLOCK_M=run_config["BLOCK_M"],
        GROUPS=groups,
        num_warps=run_config["num_warps"],
    )


def _masked_swiglu_tuning_inputs(
    input,
    output,
    output_scale,
    quant_group_size,
    masked_m,
    alpha,
    limit,
    clamp_up_add_one,
    expert_alignment,
    use_ue8m0_scales,
    expected_m,
):
    experts = masked_m.numel()
    mean = max(1, expected_m)
    counts = [mean // 2, mean, 2 * mean - mean // 2] * (experts // 3) + [mean] * (experts % 3)
    width = output.shape[-1]
    if expert_alignment is None:
        rows = min(input.shape[1], max(counts))
        counts = [min(c, rows) for c in counts]
        shape = (experts, rows)
        bounds = counts
    else:
        rows, bounds = 0, []
        for count in counts:
            bounds.append(rows + count)
            rows += triton.cdiv(count, expert_alignment) * expert_alignment
        shape = (rows,)
    if output_scale.stride(-1) == 1:
        scales = output_scale.new_empty((*shape, output_scale.shape[-1]))
    else:
        # Keep the graph capacity's column pitch while rebuilding only the valid row work.
        pitch = max(output_scale.stride(-1), triton.cdiv(shape[-1], 4) * 4)
        strides = (1, pitch) if len(shape) == 1 else (output_scale.shape[-1] * pitch, 1, pitch)
        scales = torch.empty_strided(
            (*shape, output_scale.shape[-1]),
            strides,
            dtype=output_scale.dtype,
            device=input.device,
        )
    return (
        input.new_zeros((*shape, input.shape[-1])),
        output.new_empty((*shape, width)),
        scales,
        quant_group_size,
        masked_m.new_tensor(bounds),
        alpha,
        limit,
        clamp_up_add_one,
        expert_alignment,
        use_ue8m0_scales,
        expected_m,
    ), {}


def _masked_swiglu_key(
    input, output_scale, masked_m, quant_group_size, expert_alignment, use_ue8m0_scales, alpha, limit, clamp_up_add_one
):
    return {
        "n": input.shape[-1] // 2,
        "experts": masked_m.numel(),
        "dtype": str(input.dtype),
        "group_size": quant_group_size,
        "alignment": expert_alignment,
        "scale_strides": tuple(output_scale.stride()),
        "ue8m0": use_ue8m0_scales,
        "alpha": alpha,
        "clamp": limit is not None,
        "add_one": clamp_up_add_one,
        "gelu": ffn_use_tanh_approximate_gelu(),
    }


@autotune(
    kernel_name="masked_swiglu_quant:v1",
    configs_gen_func=lambda: [
        {"BLOCK_M": m, "GROUPS": g, "row_programs": r, "num_warps": w, "NUM_STAGES": s}
        for m, g, r, w, s in (
            (1, 1, 32, 1, 6),
            (1, 1, 8, 1, 3),
            (2, 1, 8, 4, 2),
            (4, 1, 4, 4, 2),
            (2, 2, 4, 4, 2),
            (4, 2, 4, 4, 2),
            (4, 4, 4, 4, 2),
            (8, 1, 4, 4, 2),
        )
    ],
    static_key_func=_masked_swiglu_key,
    run_key_func=lambda expected_m: expected_m,
    kernel_type=AutotuneKernelType.DECODE,
    rebuild_input_func=_masked_swiglu_tuning_inputs,
    warmup_all_exist_config=False,
    benchmark_repeats=32,
)
def _masked_swiglu_quant(
    input,
    output,
    output_scale,
    quant_group_size,
    masked_m,
    alpha,
    limit,
    clamp_up_add_one,
    expert_alignment,
    use_ue8m0_scales,
    expected_m,
    run_config=None,
):
    if run_config is None:
        run_config = {
            "BLOCK_M": 1,
            "GROUPS": 1,
            "row_programs": 32 if masked_m.numel() >= 4 else 64,
            "num_warps": 1,
            "NUM_STAGES": 6,
        }
    groups = run_config["GROUPS"]
    block_n = quant_group_size * groups
    if expert_alignment is None:
        input_strides, output_strides, scale_strides = input.stride(), output.stride(), output_scale.stride()
    else:
        input_strides = (0, *input.stride())
        output_strides = (0, *output.stride())
        scale_strides = (0, *output_scale.stride())
    size_n = input.shape[-1] // 2
    grid = (triton.cdiv(size_n, block_n), run_config["row_programs"], masked_m.numel())
    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input_strides,
        output,
        *output_strides,
        output_scale,
        *scale_strides,
        masked_m,
        size_n,
        448.0,
        -448.0,
        BLOCK_N=block_n,
        NUM_STAGE=run_config["NUM_STAGES"],
        USE_LIMIT_ONLY=limit is not None and alpha is None,
        USE_TANH_APPROXIMATE_GELU=ffn_use_tanh_approximate_gelu(),
        USE_LIMIT_AND_ALPHA=alpha is not None,
        alpha=alpha,
        limit=limit,
        CLAMP_UP_ADD_ONE=clamp_up_add_one,
        EXPERT_ALIGNMENT=expert_alignment,
        USE_UE8M0_SCALES=use_ue8m0_scales,
        BLOCK_M=run_config["BLOCK_M"],
        GROUPS=groups,
        num_warps=run_config["num_warps"],
    )


def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    alpha=None,
    limit=None,
    clamp_up_add_one=True,
    *,
    expert_alignment=None,
    use_ue8m0_scales=False,
    expected_m=None,
):
    """
    Quantize the valid SwiGLU rows in each expert segment. For 3D input,
    masked_m holds row counts. For 2D expanded input it holds exclusive ends
    whose starts are aligned to expert_alignment. expected_m selects a decode
    tuning configuration. The kernel reads the row bounds from masked_m.
    """

    assert alpha is None or limit is not None
    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    if expert_alignment is None:
        assert input.ndim == 3 and input.shape[0] == masked_m.shape[0]
    else:
        assert input.ndim == 2
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0
    if (
        expected_m is not None
        and quant_group_size == expert_alignment == 32
        and use_ue8m0_scales
        and limit is not None
        and alpha is None
        and not ffn_use_tanh_approximate_gelu()
    ):
        return _expanded_swiglu_quant_1x32(input, output, output_scale, masked_m, expected_m, limit)

    return _masked_swiglu_quant(
        input,
        output,
        output_scale,
        quant_group_size,
        masked_m,
        alpha,
        limit,
        clamp_up_add_one,
        expert_alignment,
        use_ue8m0_scales,
        expected_m if expected_m is not None else input.shape[-2],
    )
    return
