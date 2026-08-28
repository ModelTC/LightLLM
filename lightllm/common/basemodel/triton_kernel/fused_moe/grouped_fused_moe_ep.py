"""Fused MoE kernel."""

import torch
import triton
import triton.language as tl
from typing import Any, Callable, Dict, List, Optional, Tuple
from lightllm.distributed import dist_group_manager
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul_mix_quant_ep import (
    silu_and_mul_masked_post_quant_fwd,
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
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.ep_balance import PrefillEPBalanceCounters
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
)
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.utils.device_utils import is_sm100_gpu
from lightllm.utils.sgl_utils import HAS_SGL_KERNEL
from lightllm.utils.tensor_buffer_manager import TensorBufferManager

logger = init_logger(__name__)
_MEGA_MOE_STATES: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
SUPPORTED_EP_EXPERT_DTYPES = ("fp8w8a8-b128-deepgemm", "fp4fp8-b32-deepgemm")


try:
    from deep_ep import Buffer, EventOverlap
    import deep_gemm

    HAS_DEEPGEMM = True
except:
    logger.warning("no deepep or deep_gemm")
    HAS_DEEPGEMM = False


def get_ep_num_sms() -> int:
    return getattr(dist_group_manager, "ep_num_sms", None) or 0


def use_sm100_mega_moe(quant_method: Any) -> bool:
    return is_sm100_gpu() and quant_method.method_name == "fp4fp8-b32-deepgemm"


def check_ep_expert_dtype(quant_method: Any):
    expert_dtype = getattr(quant_method, "method_name", None)
    if expert_dtype not in SUPPORTED_EP_EXPERT_DTYPES:
        raise ValueError(
            "EP MoE requires --expert_dtype to be one of ['fp8', 'fp4'], "
            f"but the resolved fused_moe quant method is `{expert_dtype}`. "
            "Please start with --expert_dtype fp8 or --expert_dtype fp4. "
            "Note that --expert_dtype fp4 with EP MoE is only supported on SM100 GPUs."
        )
    if expert_dtype == "fp4fp8-b32-deepgemm" and not is_sm100_gpu():
        raise RuntimeError(
            "--expert_dtype fp4 requires an SM100 GPU for EP MoE; " "please use --expert_dtype fp8 on non-SM100 GPUs."
        )


def masked_group_gemm(
    recv_x: Tuple[torch.Tensor, torch.Tensor],
    masked_m: torch.Tensor,
    dtype: torch.dtype,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expected_m: int,
    clamp_limit: Optional[float] = None,
    alloc_tensor_func: Callable = torch.empty,
):
    padded_m = recv_x[0].shape[1]
    E, N, _ = w1.shape
    block_size = 128
    # groupgemm (masked layout)
    gemm_out_a = alloc_tensor_func((E, padded_m, N), device=recv_x[0].device, dtype=dtype)
    expected_m = min(expected_m, padded_m)
    qsilu_out_scale = alloc_tensor_func(
        (E, padded_m, N // 2 // block_size), device=recv_x[0].device, dtype=torch.float32
    )
    qsilu_out = alloc_tensor_func((E, padded_m, N // 2), dtype=w1.dtype, device=recv_x[0].device)
    _deepgemm_grouped_fp8_nt_masked(recv_x, (w1, w1_scale), gemm_out_a, masked_m, expected_m)

    silu_and_mul_masked_post_quant_fwd(gemm_out_a, qsilu_out, qsilu_out_scale, block_size, masked_m, limit=clamp_limit)
    del gemm_out_a
    gemm_out_b = alloc_tensor_func(recv_x[0].shape, device=recv_x[0].device, dtype=dtype)
    _deepgemm_grouped_fp8_nt_masked((qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, masked_m, expected_m)
    return gemm_out_b


def _get_mega_moe_cache_state(w13: Any, w2: Any):
    state_key = (
        w13.weight.data_ptr(),
        w13.weight_scale.data_ptr(),
        w2.weight.data_ptr(),
        w2.weight_scale.data_ptr(),
    )
    return _MEGA_MOE_STATES.setdefault(state_key, {})


def _get_mega_moe_weights(w13: Any, w2: Any, state: Dict[str, Any]):
    if "weight_cache" not in state:
        state["weight_cache"] = deep_gemm.transform_weights_for_mega_moe(
            (w13.weight, w13.weight_scale),
            (w2.weight, w2.weight_scale),
        )
    return state["weight_cache"]


def _get_mega_moe_cumulative_stats(num_local_experts: int, device: torch.device, state: Dict[str, Any]):
    stats = state.get("stats")
    if stats is None or stats.numel() != num_local_experts or stats.device != device:
        stats = torch.zeros((num_local_experts,), device=device, dtype=torch.int32)
        state["stats"] = stats
    return stats


def mega_moe_impl(
    hidden_states: torch.Tensor,
    w13: Any,
    w2: Any,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_method: Any,
    clamp_limit: Optional[float] = None,
    alloc_tensor_func: Callable = torch.empty,
):
    if not (HAS_DEEPGEMM and hasattr(deep_gemm, "fp8_fp4_mega_moe")):
        raise RuntimeError("deep_gemm does not provide fp8-fp4 Mega MoE kernel")

    from deep_gemm.utils import per_token_cast_to_fp8

    buffer = getattr(dist_group_manager, "ep_mega_moe_buffer", None)
    if buffer is None:
        raise RuntimeError("SM100 Mega MoE requires dist_group_manager.ep_mega_moe_buffer to be initialized")

    num_tokens = hidden_states.shape[0]
    if num_tokens > buffer.num_max_tokens_per_rank:
        raise RuntimeError(
            f"Mega MoE got {num_tokens} tokens, exceeding num_max_tokens_per_rank={buffer.num_max_tokens_per_rank}"
        )

    qinput_tensor = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=quant_method.block_size,
        use_packed_ue8m0=True,
    )
    state = _get_mega_moe_cache_state(w13, w2)
    l1_weights, l2_weights = _get_mega_moe_weights(w13, w2, state)
    stats = _get_mega_moe_cumulative_stats(w13.weight.shape[0], hidden_states.device, state)
    buffer.x[:num_tokens].copy_(qinput_tensor[0])
    buffer.x_sf[:num_tokens].copy_(qinput_tensor[1])
    buffer.topk_idx[:num_tokens].copy_(topk_ids)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)

    output = alloc_tensor_func(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    deep_gemm.fp8_fp4_mega_moe(
        output,
        l1_weights,
        l2_weights,
        buffer,
        cumulative_local_expert_recv_stats=stats,
        activation_clamp=clamp_limit,
    )
    return output


def quantize_fused_experts_input(
    hidden_states: torch.Tensor,
    w13: Any,
    quant_method: Any,
):
    check_ep_expert_dtype(quant_method)
    if use_sm100_mega_moe(quant_method):
        from deep_gemm.utils import per_token_cast_to_fp8

        return per_token_cast_to_fp8(
            hidden_states,
            use_ue8m0=True,
            gran_k=quant_method.block_size,
            use_packed_ue8m0=True,
        )

    block_size_k = 0
    if w13.weight.ndim == 3:
        block_size_k = w13.weight.shape[2] // w13.weight_scale.shape[2]
    assert block_size_k == 128, "block_size_k must be 128"
    return per_token_group_quant_fp8(hidden_states, block_size_k, dtype=w13.weight.dtype)


def fused_experts(
    hidden_states: torch.Tensor,
    w13: Any,
    w2: Any,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    num_experts: int,
    quant_method: Any,
    is_prefill: Optional[bool],
    previous_event: Optional[Any] = None,
    clamp_limit: Optional[float] = None,
    alloc_tensor_func: Callable = torch.empty,
    ep_balance_counters: Optional[PrefillEPBalanceCounters] = None,
):
    check_ep_expert_dtype(quant_method)
    if use_sm100_mega_moe(quant_method):
        return mega_moe_impl(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_idx,
            quant_method,
            clamp_limit=clamp_limit,
            alloc_tensor_func=alloc_tensor_func,
        )

    buffer = dist_group_manager.ep_buffer if is_prefill else dist_group_manager.ep_low_latency_buffer
    return fused_experts_impl(
        hidden_states=hidden_states,
        w1=w13.weight,
        w2=w2.weight,
        topk_weights=topk_weights,
        topk_idx=topk_idx,
        num_experts=num_experts,
        buffer=buffer,
        is_prefill=is_prefill,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        use_int8_w8a16=False,
        w1_scale=w13.weight_scale,
        w2_scale=w2.weight_scale,
        previous_event=previous_event,
        clamp_limit=clamp_limit,
        alloc_tensor_func=alloc_tensor_func,
        ep_balance_counters=ep_balance_counters,
    )


def fused_experts_impl(
    hidden_states: torch.Tensor,  # [M, K]
    w1: torch.Tensor,  # [group, N, K]
    w2: torch.Tensor,  # [group, K, N/2]
    topk_weights: torch.Tensor,  # [M, topk]
    topk_idx: torch.Tensor,  # [M, topk]
    num_experts: int,
    buffer: Any,
    is_prefill: bool,
    use_fp8_w8a8: bool = False,
    use_fp8_all2all: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    previous_event: Optional[Any] = None,
    clamp_limit: Optional[float] = None,
    alloc_tensor_func: Callable = torch.empty,
    ep_balance_counters: Optional[PrefillEPBalanceCounters] = None,
):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_idx.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    # qaunt hidden_states
    assert use_fp8_w8a8 and use_fp8_all2all, "use_fp8_w8a8 and use_fp8_all2all must be True"

    block_size_k = 0

    if w1.ndim == 3:
        block_size_k = w1.shape[2] // w1_scale.shape[2]

    assert block_size_k == 128, "block_size_k must be 128"

    combined_x = None
    if is_prefill:
        qinput_tensor, input_scale = per_token_group_quant_fp8(
            hidden_states, block_size_k, dtype=w1.dtype, alloc_func=alloc_tensor_func
        )
        allocate_on_comm_stream = previous_event is not None
        # Expanded dispatch directly produces expert-contiguous, alignment-padded inputs:
        #   recv_x[0]: [num_expanded_tokens, hidden]
        #   recv_x[1]: [num_expanded_tokens, hidden // block_size_k], with a
        #              TMA-aligned column-major physical layout
        #   recv_topk_weights: [num_expanded_tokens]
        # Here, num_expanded_tokens is the sum of each local expert's token count padded to expert_alignment.
        # handle.num_recv_tokens_per_expert_list: a Python list of length num_local_experts;
        #     each value is the expert's token count padded to expert_alignment, and
        #     their sum is num_expanded_tokens
        # handle.num_unaligned_recv_tokens_per_expert: [num_local_experts], the actual
        #     token counts before alignment padding
        # handle.recv_src_metadata: [num_recv_tokens, topk + 2]; the last topk columns
        #     map each deduplicated receive token to rows in the expanded tensors
        recv_x, _, recv_topk_weights, handle, _ = buffer.dispatch(
            (qinput_tensor, input_scale),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=get_deepep_num_max_dispatch_tokens_per_rank_prefill(),
            expert_alignment=128,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
            do_cpu_sync=True,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=True,
        )
        if ep_balance_counters is not None:
            # Sent routes are globally conserved by all-to-all; recv_x[0] is the 128-aligned expanded compute load.
            ep_balance_counters.accumulate(
                route_load=topk_idx.numel(),
                compute_load=recv_x[0].shape[0],
            )
        # Dispatch is synchronous in this path.  Its FP8 source is no longer
        # needed once the received tensors have been produced.
        del qinput_tensor, input_scale

        all_tokens = sum(handle.num_recv_tokens_per_expert_list)
        if all_tokens > 0:
            gather_out = chunked_expanded_moe_forward(
                num_recv_tokens_per_expert_list=handle.num_recv_tokens_per_expert_list,
                num_unaligned_recv_tokens_per_expert=handle.num_unaligned_recv_tokens_per_expert,
                recv_x=recv_x,
                recv_topk_weights=recv_topk_weights,
                recv_src_metadata=handle.recv_src_metadata,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                block_size_k=block_size_k,
                workspace=dist_group_manager.get_deep_ep_prefill_moe_workspace(),
                hidden_dtype=hidden_states.dtype,
                clamp_limit=clamp_limit,
            )
        else:
            gather_out = torch.empty(
                (handle.recv_src_metadata.shape[0], w2.shape[1]),
                device=recv_x[0].device,
                dtype=hidden_states.dtype,
            )
            ######################################## warning ##################################################
            # A rank may receive no tokens during autotune warmup. Run one dummy token through
            # silu_and_mul_fwd so the empty rank matches the first kernel call made by non-empty ranks.
            # This branch does not synchronize additional calls caused by different positive chunk counts.
            if Autotuner.is_autotune_warmup():
                N = w1.shape[1]
                _gemm_out_a = torch.zeros((1, N), device=hidden_states.device, dtype=hidden_states.dtype)
                _silu_out = torch.zeros((1, N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
                silu_and_mul_fwd(_gemm_out_a.view(-1, N), _silu_out, limit=clamp_limit)
                _gemm_out_a, _silu_out = None, None
        del recv_x

        # normal combine
        combined_x, _, event = buffer.combine(
            gather_out,
            handle,
            topk_weights=None,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    else:
        # low latency dispatch
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank_decode()
        expected_m = triton.cdiv(hidden_states.shape[0] * buffer.group_size * topk_idx.shape[1], num_experts)
        recv_x, masked_m, handle, event, hook = buffer.low_latency_dispatch(
            hidden_states,
            topk_idx,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=use_fp8_w8a8,
            async_finish=False,
            return_recv_hook=False,
        )
        # deepgemm
        gemm_out_b = masked_group_gemm(
            recv_x,
            masked_m,
            hidden_states.dtype,
            w1,
            w1_scale,
            w2,
            w2_scale,
            expected_m,
            clamp_limit=clamp_limit,
            alloc_tensor_func=alloc_tensor_func,
        )
        # low latency combine
        combined_x, event_overlap, hook = buffer.low_latency_combine(
            gemm_out_b, topk_idx, topk_weights, handle, async_finish=False, return_recv_hook=False
        )
    return combined_x


def deepgemm_grouped_fp8_nt_contiguous(
    input_tuple: Tuple[torch.Tensor, torch.Tensor],
    w_tuple: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    if HAS_DEEPGEMM:
        if hasattr(deep_gemm, "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous"):
            return deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(input_tuple, w_tuple, out, m_indices)
        if hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous"):
            return deep_gemm.m_grouped_fp8_gemm_nt_contiguous(input_tuple, w_tuple, out, m_indices)
    raise RuntimeError("deep_gemm does not provide grouped_gemm_fp8 NT contiguous GEMM kernel in this version")


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
    num_recv_tokens_per_expert_list: List[int],  # [num_local_experts], 128-aligned token counts
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
    clamp_limit: Optional[float] = None,
):
    """Run bounded expanded MoE and rewrite metadata for dense DeepEP combine."""
    alignment = 128
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
            "RDMA workspace sizing invariant violated: "
            f"workspace_bytes={workspace.numel()}, gather_rows={gather_rows}, "
            f"hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
            f"chunk_rows={alignment}"
        )
    max_chunk_rows = min(all_tokens, max_chunk_rows)

    workspace_manager = TensorBufferManager(workspace)
    gather_out = workspace_manager.alloc((gather_rows, hidden_size), hidden_dtype)
    gather_out.zero_()

    # 不同 rank 接收到的 token 数不同，因此实际 chunk 数也可能不同。Autotuner warmup
    # 中的分布式通信要求各 rank 进入 autotuning 的次数一致，否则容易发生通信错位。
    # 所以只允许第一个 chunk 保持 autotuning；从第二个 chunk 开始临时关闭，循环结束
    # 后再恢复进入函数时的 warmup 状态。零 token rank 的首次调用由外层特殊分支补齐。
    is_autotune_warmup = Autotuner.is_autotune_warmup()
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
            silu_and_mul_fwd(gemm_out_a, silu_out, limit=clamp_limit)
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
            Autotuner.start_autotune_warmup()

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
