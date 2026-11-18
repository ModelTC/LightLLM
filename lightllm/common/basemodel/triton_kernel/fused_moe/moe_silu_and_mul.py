import torch

import triton
import triton.language as tl
from lightllm.common.triton_utils.autotuner import autotune
from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu
from lightllm.utils.sgl_utils import HAS_SGL_KERNEL
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import _ceil_to_ue8m0, _pack_ue8m0_scales


@triton.jit
def _silu_and_mul_quant_fp8(
    input_ptr,
    quant_ptr,
    scale_ptr,
    input_stride: tl.constexpr,
    size_n: tl.constexpr,
    groups,
    scale_stride_n: tl.constexpr,
    limit,
    eps: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_CTA: tl.constexpr,
    USE_LIMIT: tl.constexpr,
    USE_GELU: tl.constexpr,
    USE_UE8M0_SCALE: tl.constexpr,
    USE_PACKED_UE8M0: tl.constexpr,
    USE_SGL_FP32_SCALE: tl.constexpr,
):
    groups_per_row: tl.constexpr = size_n // GROUP_SIZE
    lane = tl.arange(0, GROUPS_PER_CTA)
    if USE_PACKED_UE8M0:
        row = tl.program_id(0)
        rows = row + tl.zeros((GROUPS_PER_CTA,), tl.int32)
        group_cols = tl.program_id(1) * GROUPS_PER_CTA + lane
        ids = rows * groups_per_row + group_cols
        valid = group_cols < groups_per_row
    else:
        ids = tl.program_id(0) * GROUPS_PER_CTA + lane
        rows = ids // groups_per_row
        group_cols = ids % groups_per_row
        valid = ids < groups
    cols = tl.arange(0, GROUP_SIZE)[None, :]
    offsets = rows[:, None] * input_stride + group_cols[:, None] * GROUP_SIZE + cols
    mask = valid[:, None]
    gate = tl.load(input_ptr + offsets, mask, 0).to(tl.float32)
    up = tl.load(input_ptr + offsets + size_n, mask, 0).to(tl.float32)
    if USE_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    if USE_GELU:
        cubed = gate * gate * gate
        arg = 0.7978845608028654 * (gate + 0.044715 * cubed)
        gate = 0.5 * gate * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * arg)) - 1.0))
    else:
        gate = gate / (1 + tl.exp(-gate))
    gate = gate.to(input_ptr.dtype.element_ty)
    # Preserve both rounding boundaries of activation -> BF16 buffer -> FP8.
    value = (up * gate.to(tl.float32)).to(input_ptr.dtype.element_ty).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), axis=1), eps)
    scale = amax / fp8_max
    if USE_UE8M0_SCALE or USE_PACKED_UE8M0:
        scale = _ceil_to_ue8m0(scale)
    if USE_SGL_FP32_SCALE:
        # SGL v2 computes this multiplier independently of its stored FP32 scale.
        scaled = value * (fp8_max / amax)[:, None]
    else:
        scaled = value / scale[:, None]
    quant = tl.clamp(scaled, -fp8_max, fp8_max).to(quant_ptr.dtype.element_ty)
    tl.store(quant_ptr + ids[:, None] * GROUP_SIZE + cols, quant, mask)
    if USE_PACKED_UE8M0:
        packs_per_cta: tl.constexpr = GROUPS_PER_CTA // 4
        packed_cols = tl.program_id(1) * packs_per_cta + tl.arange(0, packs_per_cta)
        packed = _pack_ue8m0_scales(scale, valid, GROUPS_PER_CTA)
        tl.store(scale_ptr + row + packed_cols * scale_stride_n, packed, packed_cols < tl.cdiv(groups_per_row, 4))
    else:
        tl.store(scale_ptr + rows + group_cols * scale_stride_n, scale, valid)


@autotune(
    kernel_name="silu_and_mul_quant_fp8:v1",
    configs_gen_func=lambda: [
        {"GROUPS_PER_CTA": g, "num_warps": w} for g, w in [(1, 1), (4, 1), (4, 4), (8, 4), (16, 4)]
    ],
    static_key_func=lambda input, group_size, quant, limit, use_ue8m0_scales, use_packed_ue8m0: {
        "N": input.shape[1] // 2,
        "input_dtype": str(input.dtype),
        "output_dtype": str(quant.dtype),
        "group_size": group_size,
        "clamp": limit is not None,
        "gelu": ffn_use_tanh_approximate_gelu(),
        "ue8m0": use_ue8m0_scales,
        **({"packed_ue8m0": True} if use_packed_ue8m0 else {}),
    },
    run_key_func=lambda input: input.shape[0],
)
def silu_and_mul_quant_fp8_out(
    input,
    quant,
    scale,
    group_size,
    limit=None,
    use_ue8m0_scales=True,
    use_packed_ue8m0=False,
    eps=1e-10,
    run_config=None,
):
    """Write the FP8 activation and its column-major scales into caller-owned buffers."""
    m, n = quant.shape
    groups_per_row = n // group_size
    if run_config is None:
        run_config = {"GROUPS_PER_CTA": 16 if use_packed_ue8m0 else 4, "num_warps": 4}
    groups_per_cta = run_config["GROUPS_PER_CTA"]
    if use_packed_ue8m0:
        groups_per_cta = max(groups_per_cta, 4)
        grid = (m, triton.cdiv(groups_per_row, groups_per_cta))
    else:
        grid = (triton.cdiv(m * groups_per_row, groups_per_cta),)
    _silu_and_mul_quant_fp8[grid](
        input,
        quant,
        scale,
        input.stride(0),
        n,
        m * groups_per_row,
        scale.stride(1),
        limit,
        eps,
        torch.finfo(quant.dtype).max,
        group_size,
        USE_LIMIT=limit is not None,
        USE_GELU=ffn_use_tanh_approximate_gelu(),
        USE_UE8M0_SCALE=use_ue8m0_scales,
        USE_PACKED_UE8M0=use_packed_ue8m0,
        USE_SGL_FP32_SCALE=HAS_SGL_KERNEL and not (use_ue8m0_scales or use_packed_ue8m0),
        GROUPS_PER_CTA=groups_per_cta,
        num_warps=run_config["num_warps"],
    )


def silu_and_mul_quant_fp8(
    input,
    group_size,
    dtype,
    limit=None,
    alloc_func=torch.empty,
    run_config=None,
    use_ue8m0_scales=True,
    use_packed_ue8m0=False,
    eps=1e-10,
):
    """Fuse blocked activation with FP8 quantization and TMA-aligned column-major scales."""
    assert input.ndim == 2 and input.stride(1) == 1
    assert input.shape[1] % (2 * group_size) == 0 and group_size == triton.next_power_of_2(group_size)
    m, n = input.shape[0], input.shape[1] // 2
    quant = alloc_func((m, n), device=input.device, dtype=dtype)
    aligned_m = triton.cdiv(m, 4) * 4
    groups_per_row = n // group_size
    scale_cols = triton.cdiv(groups_per_row, 4) if use_packed_ue8m0 else groups_per_row
    scale_dtype = torch.int32 if use_packed_ue8m0 else torch.float32
    scale = alloc_func((scale_cols, aligned_m), device=input.device, dtype=scale_dtype).T[:m, :]
    silu_and_mul_quant_fp8_out(
        input, quant, scale, group_size, limit, use_ue8m0_scales, use_packed_ue8m0, eps, run_config=run_config
    )
    return quant, scale


@triton.jit
def _silu_and_mul_kernel_fast(
    input_ptr,
    output_ptr,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    size_m,
    size_n,
    limit: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NEED_MASK: tl.constexpr,
    layout: tl.constexpr = "blocked",  # "blocked" or "interleaved"
    USE_LIMIT_AND_ALPHA: tl.constexpr = False,
    USE_LIMIT_ONLY: tl.constexpr = False,
    CLAMP_UP_ADD_ONE: tl.constexpr = True,
    USE_TANH_APPROXIMATE_GELU: tl.constexpr = False,
):
    stride_input_m = tl.cast(stride_input_m, dtype=tl.int64)
    stride_output_m = tl.cast(stride_output_m, dtype=tl.int64)

    n_block_index = tl.program_id(0)
    m_block_index = tl.program_id(1)
    n_offsets = n_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    m_start_index = m_block_index * BLOCK_M
    m_end_index = (m_block_index + 1) * BLOCK_M
    m_end_index = tl.where(m_end_index < size_m, m_end_index, size_m)
    if NEED_MASK:
        mask = n_offsets[None, :] < size_n
        other = 0.0
    else:
        mask = None
        other = None

    for m_index in tl.range(m_start_index, m_end_index, num_stages=NUM_STAGES):
        if layout == "interleaved":
            # [gate0, up0, gate1, up1, ...]
            base_offsets = m_index * stride_input_m + n_offsets[None, :] * 2
            gate_offsets = base_offsets
            up_offsets = base_offsets + 1
        elif layout == "blocked":
            # [gate0, gate1, ..., up0, up1, ...]
            gate_offsets = m_index * stride_input_m + n_offsets[None, :]
            up_offsets = m_index * stride_input_m + (n_offsets[None, :] + size_n)
        out_offsets = m_index * stride_output_m + n_offsets[None, :]

        up = tl.load(
            input_ptr + up_offsets,
            mask=mask,
            other=other,
        )
        gate = tl.load(
            input_ptr + gate_offsets,
            mask=mask,
            other=other,
        ).to(tl.float32)

        if USE_LIMIT_AND_ALPHA:
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
            gate = 1 / (1 + tl.exp(-gate * alpha)) * gate
            gate = gate.to(input_ptr.dtype.element_ty)
            if CLAMP_UP_ADD_ONE:
                up += 1
            tl.store(output_ptr + out_offsets, up * gate, mask=mask)
        else:
            if USE_LIMIT_ONLY:
                # clamped swiglu (DeepSeek-V4 swiglu_limit): clamp 后接标准 silu，
                # 无 gpt-oss 的 alpha 缩放与 (up+1)。
                gate = tl.minimum(gate, limit)
                up = tl.minimum(tl.maximum(up, -limit), limit)
            if USE_TANH_APPROXIMATE_GELU:
                # tanh-approx GELU, matching Gemma's gelu_pytorch_tanh MLP.
                gate_cubed = gate * gate * gate
                tanh_arg = 0.7978845608028654 * (gate + 0.044715 * gate_cubed)
                tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
                gate = 0.5 * gate * (1.0 + tanh_val)
            else:
                gate = gate / (1 + tl.exp(-gate))
            gate = gate.to(input_ptr.dtype.element_ty)

            tl.store(
                output_ptr + out_offsets,
                up * gate,
                mask=mask,
            )


def _get_silu_and_mul_configs():
    return [
        {"BLOCK_M": bm, "BLOCK_N": bn, "num_warps": nw, "NUM_STAGES": ns}
        for ns in [1, 2, 4]
        for nw in [1, 4, 8]
        for bm in [1, 8, 32, 64, 128, 256]
        for bn in [32, 64, 128, 256]
    ]


def _get_silu_and_mul_static_key(input: torch.Tensor, output: torch.Tensor):
    return {"N": input.shape[-1] // 2, "out_dtype": str(output.dtype)}


@autotune(
    kernel_name="silu_and_mul_fwd:v1",
    configs_gen_func=_get_silu_and_mul_configs,
    static_key_func=_get_silu_and_mul_static_key,
    run_key_func=lambda input: input.shape[0],
    mutates_args=["output"],
)
def silu_and_mul_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    layout="blocked",
    limit=None,
    alpha=None,
    run_config=None,
    clamp_up_add_one=True,
):
    assert input.stride(-1) == 1
    assert output.is_contiguous()
    # limit+alpha: gpt-oss 语义 (up+1)*silu(alpha*gate); 仅 limit: clamp 后标准 silu (DeepSeek-V4)
    assert alpha is None or limit is not None

    stride_input_m = input.stride(0)
    stride_input_n = input.stride(1)
    stride_output_m = output.stride(0)
    stride_output_n = output.stride(1)
    size_m = input.shape[0]
    size_n = input.shape[-1] // 2

    if not run_config:
        if size_m < 256:
            run_config = {"BLOCK_M": 1, "BLOCK_N": 128, "num_warps": 1, "NUM_STAGES": 1}
        else:
            run_config = {"BLOCK_M": 16, "BLOCK_N": 128, "num_warps": 4, "NUM_STAGES": 5}

    BLOCK_M = run_config["BLOCK_M"]
    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    NUM_STAGES = run_config["NUM_STAGES"]
    # limit the grid size to avoid the invalid argument error of triton
    while triton.cdiv(size_m, BLOCK_M) > 8192:
        BLOCK_M *= 2
    USE_LIMIT_AND_ALPHA = limit is not None and alpha is not None
    USE_LIMIT_ONLY = limit is not None and alpha is None

    grid = (
        triton.cdiv(size_n, BLOCK_N),
        triton.cdiv(size_m, BLOCK_M),
    )
    NEED_MASK = (size_n % BLOCK_N) != 0
    _silu_and_mul_kernel_fast[grid](
        input_ptr=input,
        output_ptr=output,
        stride_input_m=stride_input_m,
        stride_input_n=stride_input_n,
        stride_output_m=stride_output_m,
        stride_output_n=stride_output_n,
        size_m=size_m,
        size_n=size_n,
        limit=limit,
        alpha=alpha,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_STAGES=NUM_STAGES,
        NEED_MASK=NEED_MASK,
        num_warps=num_warps,
        layout=layout,
        USE_LIMIT_AND_ALPHA=USE_LIMIT_AND_ALPHA,
        USE_LIMIT_ONLY=USE_LIMIT_ONLY,
        CLAMP_UP_ADD_ONE=clamp_up_add_one,
        USE_TANH_APPROXIMATE_GELU=ffn_use_tanh_approximate_gelu(),
    )
    return
