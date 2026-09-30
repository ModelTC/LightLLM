import torch

import triton
import triton.language as tl

from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu


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
        if USE_TANH_APPROXIMATE_GELU:
            gate_cubed = gate * gate * gate
            tanh_arg = 0.7978845608028654 * (gate + 0.044715 * gate_cubed)
            tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
            gate = 0.5 * gate * (1.0 + tanh_val)
        else:
            gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
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
):
    """按 expert 的 psum 区间处理有效行，跳过 expert 之间的对齐填充。"""
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
    # TODO: 根据 expert 数量、预期 token 数和 GPU 型号增加 autotune 配置。
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
        num_warps=num_warps,
    )
