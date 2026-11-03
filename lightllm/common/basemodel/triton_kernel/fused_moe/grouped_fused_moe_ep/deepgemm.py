import torch
import triton
from typing import List, Optional, Tuple

from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul_mix_quant_ep import (
    silu_and_mul_masked_post_quant_fwd,
    silu_and_mul_psum_post_quant_fwd,
)
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
    ep_build_m_indices,
    ep_compact_metadata,
    ep_gather_chunk,
    ep_zero_padding,
)
from lightllm.utils.envs_utils import get_env_use_ue8m0_scales
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.utils.sgl_utils import HAS_SGL_KERNEL
from lightllm.utils.tensor_buffer_manager import TensorBufferManager

logger = init_logger(__name__)

deep_gemm = None
try:
    import deep_gemm

    HAS_DEEPGEMM = True
except Exception:
    logger.warning("no deepep or deep_gemm")
    HAS_DEEPGEMM = False


def get_mk_alignment_for_contiguous_layout() -> int:
    """返回 DeepGEMM contiguous grouped GEMM 使用的 M 对齐值。"""
    assert HAS_DEEPGEMM
    return deep_gemm.get_mk_alignment_for_contiguous_layout()


def set_mk_alignment_for_contiguous_layout(alignment: int) -> None:
    """设置 DeepGEMM contiguous grouped GEMM 使用的 M 对齐值。"""
    assert HAS_DEEPGEMM
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)


def deepgemm_grouped_fp8_nt_contiguous(
    input_tuple: Tuple[torch.Tensor, torch.Tensor],
    w_tuple: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    use_psum_layout: bool = False,
    expected_m_for_psum_layout: Optional[int] = None,
):
    """调用不同版本的 DeepGEMM contiguous grouped GEMM。

    ``m_indices`` 在两种布局下的含义不同：

    - 普通布局：shape 为 ``[M]``，逐行记录输入使用的 expert ID。
    - psum 布局：shape 为 ``[expert_num]``，第 i 项记录 expert i 最后一个
      有效行的后一位置。expert 0 的起点为 0；其余 expert 的起点是前一个
      expert 的结束位置向上对齐到 DeepGEMM M 对齐边界后的结果。
    """
    if not HAS_DEEPGEMM:
        raise RuntimeError("deep_gemm does not provide grouped_gemm_fp8 NT contiguous GEMM kernel in this version")

    if hasattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous"):
        if not use_psum_layout:
            return deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                lhs=input_tuple,
                rhs=w_tuple,
                out=out,
                m_indices=m_indices,
            )
        return deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            lhs=input_tuple,
            rhs=w_tuple,
            out=out,
            m_indices=m_indices,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_for_psum_layout,
        )

    if hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous"):
        return deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a=input_tuple,
            b=w_tuple,
            d=out,
            grouped_layout=m_indices,
            use_psum_layout=use_psum_layout,
            expected_m_for_psum_layout=expected_m_for_psum_layout,
        )

    raise RuntimeError("deep_gemm does not provide grouped_gemm_fp8 NT contiguous GEMM kernel in this version")


def decode_masked_group_gemm(
    recv_x: Tuple[torch.Tensor, torch.Tensor],
    expert_token_psum: torch.Tensor,
    expert_alignment: int,
    dtype: torch.dtype,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expected_m: int,
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
    clamp_up_add_one: bool = True,
):
    """在 ElasticBuffer 的 expert 连续布局上执行两次 decode grouped GEMM。

    ``recv_x`` 的容量按最坏情况固定，便于 CUDA Graph 重放；GPU 上的
    ``expert_token_psum`` 给出每个 expert 的有效结束位置，DeepGEMM 和中间
    SwiGLU 只处理这些位置覆盖的行，不需要把计数同步回 CPU。
    """
    assert (limit is None and alpha is None) or (limit is not None and alpha is not None)
    max_rows, hidden_size = recv_x[0].shape
    intermediate_twice = w1.shape[1]
    intermediate_size = intermediate_twice // 2
    block_size = 128

    gemm_out_a = torch.empty((max_rows, intermediate_twice), device=recv_x[0].device, dtype=dtype)
    deepgemm_grouped_fp8_nt_contiguous(
        recv_x,
        (w1, w1_scale),
        gemm_out_a,
        expert_token_psum,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )

    qsilu_out = torch.empty((max_rows, intermediate_size), dtype=w1.dtype, device=recv_x[0].device)
    scale_storage = torch.empty(
        (intermediate_size // block_size, triton.cdiv(max_rows, 4) * 4),
        dtype=torch.float32,
        device=recv_x[0].device,
    )
    qsilu_out_scale = scale_storage[:, :max_rows].T
    silu_and_mul_psum_post_quant_fwd(
        input=gemm_out_a,
        output=qsilu_out,
        output_scale=qsilu_out_scale,
        expert_token_psum=expert_token_psum,
        expert_alignment=expert_alignment,
        quant_group_size=block_size,
        alpha=alpha,
        limit=limit,
        clamp_up_add_one=clamp_up_add_one,
    )
    del gemm_out_a

    gemm_out_b = torch.empty((max_rows, hidden_size), device=recv_x[0].device, dtype=dtype)
    deepgemm_grouped_fp8_nt_contiguous(
        (qsilu_out, qsilu_out_scale),
        (w2, w2_scale),
        gemm_out_b,
        expert_token_psum,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
    )
    return gemm_out_b


def masked_group_gemm(
    recv_x: Tuple[torch.Tensor, torch.Tensor],
    masked_m: torch.Tensor,
    dtype: torch.dtype,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expected_m: int,
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
    clamp_up_add_one: bool = True,
):
    padded_m = recv_x[0].shape[1]
    E, N, _ = w1.shape
    block_size = 128
    # groupgemm (masked layout)
    gemm_out_a = torch.empty((E, padded_m, N), device=recv_x[0].device, dtype=dtype)
    expected_m = min(expected_m, padded_m)
    qsilu_out_scale = torch.empty((E, padded_m, N // 2 // block_size), device=recv_x[0].device, dtype=torch.float32)
    qsilu_out = torch.empty((E, padded_m, N // 2), dtype=w1.dtype, device=recv_x[0].device)
    _deepgemm_grouped_fp8_nt_masked(recv_x, (w1, w1_scale), gemm_out_a, masked_m, expected_m)

    silu_and_mul_masked_post_quant_fwd(
        gemm_out_a,
        qsilu_out,
        qsilu_out_scale,
        block_size,
        masked_m,
        alpha=alpha,
        limit=limit,
        clamp_up_add_one=clamp_up_add_one,
    )
    del gemm_out_a
    gemm_out_b = torch.empty_like(recv_x[0], device=recv_x[0].device, dtype=dtype)
    _deepgemm_grouped_fp8_nt_masked((qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, masked_m, expected_m)
    return gemm_out_b


def _get_max_chunk_rows(
    workspace: torch.Tensor,
    gather_rows: int,
    hidden_size: int,
    intermediate_size: int,
    intermediate_twice: int,
    scale_cols: int,
    hidden_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    expert_alignment: int,
) -> int:
    """计算并缓存当前 workspace 配置能够容纳的最大 chunk 行数。"""
    if not hasattr(_get_max_chunk_rows, "cache"):
        _get_max_chunk_rows.cache = {}
    max_chunk_rows_cache = _get_max_chunk_rows.cache

    # 同一 1024 行区间共用一个缓存项，并按区间上界探测，保证复用结果不会高估可用空间。
    cached_gather_rows = (gather_rows + 1023) // 1024 * 1024
    cache_key = (
        workspace.numel(),
        workspace.device,
        cached_gather_rows,
        hidden_size,
        intermediate_size,
        intermediate_twice,
        scale_cols,
        hidden_dtype,
        quant_dtype,
        expert_alignment,
        HAS_SGL_KERNEL,
    )
    if cache_key in max_chunk_rows_cache:
        return max_chunk_rows_cache[cache_key]

    def can_allocate(chunk_rows: int) -> bool:
        """按实际计算阶段的生命周期申请 buffer，探测该 chunk 是否能够执行。"""
        try:
            probe_manager = TensorBufferManager(workspace)
            probe_manager.alloc((cached_gather_rows, hidden_size), hidden_dtype)

            # W1 阶段同时保存 GEMM 输出和 SwiGLU 输出。
            silu_out = probe_manager.alloc((chunk_rows, intermediate_size), hidden_dtype)
            gemm_out_a = probe_manager.alloc((chunk_rows, intermediate_twice), hidden_dtype)
            probe_manager.free(gemm_out_a)

            # 量化阶段复用 W1 输出空间，并继续保留 SwiGLU 输出。
            probe_manager.alloc((chunk_rows, intermediate_size), quant_dtype)
            aligned_chunk_rows = (chunk_rows + 3) // 4 * 4
            if HAS_SGL_KERNEL:
                probe_manager.alloc((scale_cols, aligned_chunk_rows), torch.float32)
            else:
                # LightLLM fallback 通过 alloc_func 申请 row-major scale；后续 TMA 转置不占用 workspace。
                probe_manager.alloc((chunk_rows, scale_cols), torch.float32)
            probe_manager.free(silu_out)

            # W2 阶段释放 SwiGLU 输出后申请最终 GEMM 输出。
            probe_manager.alloc((chunk_rows, hidden_size), hidden_dtype)
        except MemoryError:
            return False
        return True

    # W1 阶段必须同时保存 gemm_out_a 和 silu_out，可据此得到 chunk 数量的绝对上界。
    w1_row_bytes = (intermediate_twice + intermediate_size) * hidden_dtype.itemsize
    left = 1
    right = workspace.numel() // w1_row_bytes // expert_alignment
    max_chunk_rows = 0

    while left <= right:
        chunk_count = (left + right) // 2
        chunk_rows = chunk_count * expert_alignment
        if can_allocate(chunk_rows):
            max_chunk_rows = chunk_rows
            left = chunk_count + 1
        else:
            right = chunk_count - 1

    max_chunk_rows_cache[cache_key] = max_chunk_rows
    logger.info("cache DeepEP max_chunk_rows: key=%s, max_chunk_rows=%s", cache_key, max_chunk_rows)
    return max_chunk_rows


def chunked_expanded_moe_forward(
    num_recv_tokens_per_expert_list: List[int],  # [num_local_experts], alignment-padded token counts
    num_unaligned_recv_tokens_per_expert: torch.Tensor,  # [num_local_experts], actual token counts
    recv_x: Tuple[
        torch.Tensor, torch.Tensor  # [fp8, scale]
    ],  # ([num_expanded_tokens, hidden_size], [num_expanded_tokens, hidden_size // block_size_k])
    recv_topk_weights: torch.Tensor,  # [num_expanded_tokens]
    recv_src_metadata: torch.Tensor,  # [num_recv_tokens, topk + 2]
    w1: torch.Tensor,  # [num_local_experts, 2 * intermediate_size, hidden_size]
    w1_scale: torch.Tensor,  # [num_local_experts, 2 * intermediate_size // block_size_k, hidden_size // block_size_k]
    w2: torch.Tensor,  # [num_local_experts, hidden_size, intermediate_size]
    w2_scale: torch.Tensor,  # [num_local_experts, hidden_size // block_size_k, intermediate_size // block_size_k]
    block_size_k: int,
    workspace: torch.Tensor,  # [workspace_bytes], uint8
    hidden_dtype: torch.dtype,  # scalar dtype descriptor
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
    clamp_up_add_one: bool = True,
):
    """Run bounded expanded MoE and rewrite metadata for dense DeepEP combine."""
    alignment = get_mk_alignment_for_contiguous_layout()
    all_tokens, intermediate_twice = recv_x[0].shape[0], w1.shape[1]
    intermediate_size, hidden_size = intermediate_twice // 2, w2.shape[1]
    assert all_tokens == sum(num_recv_tokens_per_expert_list) and all_tokens % alignment == 0
    assert all_tokens > 0, "chunked_expanded_moe_forward requires non-empty input"
    assert workspace.dtype == torch.uint8 and workspace.ndim == 1 and workspace.is_contiguous()

    m_indices = torch.empty(all_tokens, device=recv_x[0].device, dtype=torch.int32)
    # 与 m_indices 一一对应：0 表示真实 token，1 表示 expert 对齐产生的 padding 行。
    # padding 行必须在 grouped GEMM 前清零，避免无效数据参与计算。
    padding_mask = torch.empty_like(m_indices)
    ep_build_m_indices(num_unaligned_recv_tokens_per_expert, m_indices, padding_mask, alignment)
    ep_zero_padding(
        recv_x[0],
        recv_x[1],
        recv_topk_weights,
        padding_mask,
    )
    del padding_mask

    gather_rows = recv_src_metadata.shape[0]
    scale_cols = intermediate_size // block_size_k
    max_chunk_rows = _get_max_chunk_rows(
        workspace=workspace,
        gather_rows=gather_rows,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        intermediate_twice=intermediate_twice,
        scale_cols=scale_cols,
        hidden_dtype=hidden_dtype,
        quant_dtype=w2.dtype,
        expert_alignment=alignment,
    )

    if max_chunk_rows == 0:
        raise RuntimeError(
            f"DeepEP workspace with {workspace.numel()} bytes cannot hold the dense output and "
            f"one {alignment}-row temporary chunk"
        )
    max_chunk_rows = min(all_tokens, max_chunk_rows)

    workspace_manager = TensorBufferManager(workspace)
    gather_out = workspace_manager.alloc((gather_rows, hidden_size), hidden_dtype)
    gather_out.zero_()

    # 不同 rank 接收到的 token 数不同，因此实际 chunk 数也可能不同。Autotuner warmup
    # 中的分布式通信要求各 rank 进入 autotuning 的次数一致，否则容易发生通信错位。
    # 所以只允许第一个 chunk 保持 autotuning；从第二个 chunk 开始临时关闭，循环结束
    # 后再恢复进入函数时的 warmup 状态。零 token rank 的首次调用由外层特殊分支补齐。
    is_autotune_warmup = Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL)
    try:
        for chunk_index, chunk_start in enumerate(range(0, all_tokens, max_chunk_rows)):
            if is_autotune_warmup and chunk_index == 1:
                Autotuner.end_autotune_warmup()

            chunk_end = min(chunk_start + max_chunk_rows, all_tokens)
            chunk_rows = chunk_end - chunk_start
            silu_out = workspace_manager.alloc((chunk_rows, intermediate_size), hidden_dtype)
            gemm_out_a = workspace_manager.alloc((chunk_rows, intermediate_twice), hidden_dtype)
            deepgemm_grouped_fp8_nt_contiguous(
                (recv_x[0][chunk_start:chunk_end], recv_x[1][chunk_start:chunk_end]),
                (w1, w1_scale),
                gemm_out_a,
                m_indices[chunk_start:chunk_end],
            )
            silu_and_mul_fwd(gemm_out_a, silu_out, alpha=alpha, limit=limit, clamp_up_add_one=clamp_up_add_one)
            workspace_manager.free(gemm_out_a)
            del gemm_out_a

            quant_buffers = []

            def workspace_quant_alloc(shape, dtype, device):
                if device != workspace.device:
                    raise RuntimeError(f"quant buffer must be allocated on {workspace.device}, got {device}")
                quant_buffer = workspace_manager.alloc(shape, dtype)
                quant_buffers.append(quant_buffer)
                return quant_buffer

            qsilu_out, qsilu_out_scale = per_token_group_quant_fp8(
                silu_out,
                block_size_k,
                dtype=w2.dtype,
                column_major_scales=True,
                scale_tma_aligned=True,
                alloc_func=workspace_quant_alloc,
                use_ue8m0_scales=get_env_use_ue8m0_scales(),
            )
            workspace_manager.free(silu_out)
            del silu_out

            gemm_out_b = workspace_manager.alloc((chunk_rows, hidden_size), hidden_dtype)
            deepgemm_grouped_fp8_nt_contiguous(
                (qsilu_out, qsilu_out_scale),
                (w2, w2_scale),
                gemm_out_b,
                m_indices[chunk_start:chunk_end],
            )
            del qsilu_out, qsilu_out_scale
            for quant_buffer in quant_buffers:
                workspace_manager.free(quant_buffer)

            ep_gather_chunk(gemm_out_b, chunk_start, recv_topk_weights, recv_src_metadata, gather_out)
            workspace_manager.free(gemm_out_b)
    finally:
        if is_autotune_warmup:
            Autotuner.start_autotune_warmup(AutotuneKernelType.GENERAL)

    ep_compact_metadata(recv_src_metadata)
    return gather_out


def _deepgemm_grouped_fp8_nt_masked(
    input_tuple: Tuple[torch.Tensor, torch.Tensor],
    w_tuple: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    if HAS_DEEPGEMM:
        if hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_masked"):
            return deep_gemm.m_grouped_fp8_gemm_nt_masked(input_tuple, w_tuple, out, masked_m, expected_m)
        if hasattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_masked"):
            return deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(input_tuple, w_tuple, out, masked_m, expected_m)
    raise RuntimeError("deep_gemm does not provide grouped_gemm_fp8 NT contiguous GEMM kernel in this version")


def deepgemm_grouped_fp8_fp4_nt_contiguous(
    input_tuple: Tuple[torch.Tensor, torch.Tensor],
    w_tuple: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    grouped_layout: torch.Tensor,
    use_psum_layout: bool = False,
):
    if HAS_DEEPGEMM and hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        return deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            input_tuple,
            w_tuple,
            out,
            grouped_layout,
            use_psum_layout=use_psum_layout,
            recipe=(1, 1, 32),
        )
    raise RuntimeError("deep_gemm does not provide grouped fp8-fp4 NT contiguous GEMM kernel")
