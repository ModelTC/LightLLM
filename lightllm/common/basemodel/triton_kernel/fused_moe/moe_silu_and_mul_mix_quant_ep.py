import torch

import triton
import triton.language as tl

from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu
from lightllm.utils.envs_utils import get_env_use_ue8m0_scales


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
    USE_TANH_APPROXIMATE_GELU: tl.constexpr = False,
    USE_LIMIT_AND_ALPHA: tl.constexpr = False,
    alpha: tl.constexpr = None,
    limit: tl.constexpr = None,
    CLAMP_UP_ADD_ONE: tl.constexpr = True,
    USE_UE8M0_SCALE: tl.constexpr = False,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr + expert_id * stride_output_scale_0 + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE):
        gate = tl.load(input_ptr_offs + token_index * stride_input_1, mask=offs_in_d < size_n, other=0.0).to(tl.float32)
        up = tl.load(input_ptr_offs + token_index * stride_input_1 + size_n, mask=offs_in_d < size_n, other=0.0)
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
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        if USE_UE8M0_SCALE:
            output_s = tl.exp2(tl.ceil(tl.log2(output_s)))
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
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
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert (limit is None and alpha is None) or (limit is not None and alpha is not None)
    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        USE_TANH_APPROXIMATE_GELU=ffn_use_tanh_approximate_gelu(),
        USE_LIMIT_AND_ALPHA=limit is not None,
        alpha=alpha,
        limit=limit,
        CLAMP_UP_ADD_ONE=clamp_up_add_one,
        USE_UE8M0_SCALE=get_env_use_ue8m0_scales(),
        num_warps=num_warps,
    )
    return


@triton.jit
def _silu_and_mul_psum_post_quant_kernel(
    input_ptr,
    output_ptr,
    output_scale_ptr,
    expert_token_psum_ptr,
    input_stride_m,
    output_stride_m,
    output_scale_stride_m,
    output_scale_stride_k,
    size_n,
    fp8_max,
    fp8_min,
    EXPERT_ALIGNMENT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    NEED_MASK: tl.constexpr,
    USE_TANH_APPROXIMATE_GELU: tl.constexpr = False,
    USE_LIMIT_AND_ALPHA: tl.constexpr = False,
    alpha: tl.constexpr = None,
    limit: tl.constexpr = None,
    CLAMP_UP_ADD_ONE: tl.constexpr = True,
    USE_UE8M0_SCALE: tl.constexpr = False,
):
    hidden_block_id = tl.program_id(0).to(tl.int64)
    row_offset_in_expert = tl.program_id(1).to(tl.int64)
    row_step = tl.num_programs(1).to(tl.int64)
    expert_id = tl.program_id(2).to(tl.int64)

    # psum 保存每个 expert 最后一个有效行的后一位置；下一个 expert 的
    # 起点是前一个结束位置向上对齐到 EXPERT_ALIGNMENT 后的位置。
    expert_end = tl.load(expert_token_psum_ptr + expert_id).to(tl.int64)
    previous_expert_end = tl.load(
        expert_token_psum_ptr + expert_id - 1,
        mask=expert_id > 0,
        other=0,
    ).to(tl.int64)
    expert_start = tl.cdiv(previous_expert_end, EXPERT_ALIGNMENT) * EXPERT_ALIGNMENT

    # 行号与 stride 相乘后才用于地址计算；必须在乘法前提升为 int64，
    # 否则大 tensor 的元素偏移可能先在 int32 中溢出。
    input_stride_m = tl.cast(input_stride_m, tl.int64)
    output_stride_m = tl.cast(output_stride_m, tl.int64)
    output_scale_stride_m = tl.cast(output_scale_stride_m, tl.int64)
    output_scale_stride_k = tl.cast(output_scale_stride_k, tl.int64)

    hidden_offsets = hidden_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    input_offsets = hidden_offsets
    output_offsets = hidden_offsets
    scale_offsets = hidden_block_id * output_scale_stride_k
    if NEED_MASK:
        hidden_mask = hidden_offsets < size_n
        other = 0.0
    else:
        hidden_mask = None
        other = None

    first_row = expert_start + row_offset_in_expert
    for row_index in tl.range(first_row, expert_end, row_step, num_stages=NUM_STAGE):
        input_row_offsets = row_index * input_stride_m
        gate = tl.load(
            input_ptr + input_row_offsets + input_offsets,
            mask=hidden_mask,
            other=other,
        ).to(tl.float32)
        up = tl.load(
            input_ptr + input_row_offsets + input_offsets + size_n,
            mask=hidden_mask,
            other=other,
        )
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
            gate_up = gate_up.to(input_ptr.dtype.element_ty).to(tl.float32)
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        if USE_UE8M0_SCALE:
            output_s = tl.exp2(tl.ceil(tl.log2(output_s)))
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(output_ptr.dtype.element_ty)
        tl.store(
            output_ptr + row_index * output_stride_m + output_offsets,
            output_q,
            mask=hidden_mask,
        )
        tl.store(output_scale_ptr + row_index * output_scale_stride_m + scale_offsets, output_s)


def silu_and_mul_psum_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    expert_token_psum: torch.Tensor,
    expert_alignment: int,
    quant_group_size: int,
    alpha=None,
    limit=None,
    clamp_up_add_one=True,
):
    """按 expert 的 psum 区间处理有效行，跳过 expert 之间的对齐填充。"""
    assert (limit is None and alpha is None) or (limit is not None and alpha is not None)
    assert input.ndim == 2 and input.is_contiguous()
    assert output.ndim == 2 and output.is_contiguous()
    assert output_scale.ndim == 2
    assert output.dtype == torch.float8_e4m3fn
    assert expert_token_psum.ndim == 1 and expert_token_psum.numel() > 0
    assert expert_alignment > 0

    assert input.shape[0] == output.shape[0] == output_scale.shape[0]
    assert input.shape[1] == output.shape[1] * 2

    size_n = output.shape[1]
    assert size_n % quant_group_size == 0
    assert output_scale.shape[1] == size_n // quant_group_size

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N == quant_group_size
    NEED_MASK = (size_n % BLOCK_N) != 0

    num_experts = expert_token_psum.numel()
    # 行方向总并行度以 256 为目标，并平均分给所有 expert。常见配置为：
    # 4/8/16/32/64/128/256 个 expert 分别得到 64/32/16/8/4/2/1。
    # 单 expert 最多使用 64 个 program；expert 超过 256 时仍至少使用 1 个。
    target_row_parallelism = 256
    max_row_parallelism_per_expert = 64
    row_parallelism = target_row_parallelism // num_experts
    row_parallelism = max(row_parallelism, 1)
    row_parallelism = min(row_parallelism, max_row_parallelism_per_expert)
    grid = (hidden_dim_split_block_num, row_parallelism, num_experts)

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max
    _silu_and_mul_psum_post_quant_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        output_scale_ptr=output_scale,
        expert_token_psum_ptr=expert_token_psum,
        input_stride_m=input.stride(0),
        output_stride_m=output.stride(0),
        output_scale_stride_m=output_scale.stride(0),
        output_scale_stride_k=output_scale.stride(1),
        size_n=size_n,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
        EXPERT_ALIGNMENT=expert_alignment,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        NEED_MASK=NEED_MASK,
        USE_TANH_APPROXIMATE_GELU=ffn_use_tanh_approximate_gelu(),
        USE_LIMIT_AND_ALPHA=limit is not None,
        alpha=alpha,
        limit=limit,
        CLAMP_UP_ADD_ONE=clamp_up_add_one,
        USE_UE8M0_SCALE=get_env_use_ue8m0_scales(),
        num_warps=num_warps,
    )
