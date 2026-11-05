import torch
import triton
from typing import Any, Callable, Optional

from lightllm.distributed import dist_group_manager
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
    ep_reduce_decode_output,
)
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
    get_env_use_ue8m0_scales,
)
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType

from .common import check_ep_expert_dtype, get_ep_num_sms
from .deepgemm import (
    chunked_expanded_moe_forward,
    decode_masked_group_gemm,
    get_mk_alignment_for_contiguous_layout,
)
from .fp4_mega import mega_moe_impl, use_sm100_mega_moe


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
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
    clamp_up_add_one: bool = True,
    shared_expert_func: Optional[Callable[[], torch.Tensor]] = None,
):
    assert (limit is None and alpha is None) or (limit is not None and alpha is not None)
    check_ep_expert_dtype(quant_method)
    if use_sm100_mega_moe(quant_method):
        if limit is not None:
            raise NotImplementedError("FP4 Mega MoE does not support clamped SwiGLU")
        # Mega MoE has no split combine, so the shared expert cannot overlap it.
        shared = shared_expert_func() if shared_expert_func is not None else None
        routed = mega_moe_impl(hidden_states, w13, w2, topk_weights, topk_idx, quant_method)
        return routed if shared is None else routed + shared

    return fused_experts_impl(
        hidden_states=hidden_states,
        w1=w13.weight,
        w2=w2.weight,
        topk_weights=topk_weights,
        topk_idx=topk_idx,
        num_experts=num_experts,
        block_size_k=quant_method.block_size,
        buffer=dist_group_manager.ep_buffer,
        is_prefill=is_prefill,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        use_int8_w8a16=False,
        w1_scale=w13.weight_scale,
        w2_scale=w2.weight_scale,
        previous_event=previous_event,
        alpha=alpha,
        limit=limit,
        clamp_up_add_one=clamp_up_add_one,
        shared_expert_func=shared_expert_func,
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
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
    clamp_up_add_one: bool = True,
    block_size_k: int = 128,
    shared_expert_func: Optional[Callable[[], torch.Tensor]] = None,
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
    assert block_size_k == 128, "block_size_k must be 128"
    # TODO: 共享专家目前整段只和 combine 重叠。可以拆开：dispatch 期间算 up，
    # combine 期间算 down，把两段通信的空闲都盖住。

    if is_prefill:
        qinput_tensor, input_scale = per_token_group_quant_fp8(
            hidden_states, block_size_k, dtype=w1.dtype, use_ue8m0_scales=get_env_use_ue8m0_scales()
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
            expert_alignment=get_mk_alignment_for_contiguous_layout(),
            # 当前 prefill 路径同步等待通信完成，不与 grouped GEMM 重叠，
            # 因此使用非 overlap 配额，让通信可以占用更多 SM。
            num_sms=get_ep_num_sms(overlap_with_compute=False),
            previous_event=previous_event,
            async_with_compute_stream=False,
            allocate_on_comm_stream=allocate_on_comm_stream,
            do_cpu_sync=True,
            do_handle_copy=False,
            do_expand=True,
            use_tma_aligned_col_major_sf=True,
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
                hidden_dtype=hidden_states.dtype,
                alpha=alpha,
                limit=limit,
                clamp_up_add_one=clamp_up_add_one,
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
            if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL):
                N = w1.shape[1]
                _gemm_out_a = torch.zeros((1, N), device=hidden_states.device, dtype=hidden_states.dtype)
                _silu_out = torch.zeros((1, N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
                silu_and_mul_fwd(
                    _gemm_out_a.view(-1, N),
                    _silu_out,
                    alpha=alpha,
                    limit=limit,
                    clamp_up_add_one=clamp_up_add_one,
                )
                _gemm_out_a, _silu_out = None, None
        del recv_x

        combined_x = _combine_with_optional_shared_expert(
            buffer,
            gather_out,
            handle,
            previous_event,
            allocate_on_comm_stream,
            shared_expert_func,
        )
        return combined_x

    qinput_tensor, input_scale = per_token_group_quant_fp8(
        hidden_states, block_size_k, dtype=w1.dtype, use_ue8m0_scales=get_env_use_ue8m0_scales()
    )
    # 同一 token 命中目标 rank 上的多个 expert 时，hidden vector 只跨卡
    # 发送一次；接收端再将其展开成 expert 连续布局。这里的对齐值必须与
    # DeepGEMM contiguous grouped GEMM 的 M 对齐要求完全一致。
    recv_x, _, recv_topk_weights, ep_handle, _ = buffer.dispatch(
        (qinput_tensor, input_scale),
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=get_deepep_num_max_dispatch_tokens_per_rank_decode(),
        expert_alignment=get_mk_alignment_for_contiguous_layout(),
        # 串行 decode 会在 dispatch 内部等待通信完成，因此输出 tensor
        # 直接归属计算流，并使用非 overlap SM 配额缩短通信时间。
        num_sms=get_ep_num_sms(overlap_with_compute=False),
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
        do_cpu_sync=False,
        do_handle_copy=False,
        do_expand=True,
        do_zero_padding=True,
        use_tma_aligned_col_major_sf=True,
    )
    del qinput_tensor, input_scale

    expected_m = triton.cdiv(hidden_states.shape[0] * buffer.num_ranks * topk_idx.shape[1], num_experts)
    expert_output = decode_masked_group_gemm(
        recv_x=recv_x,
        expert_token_psum=ep_handle.psum_num_recv_tokens_per_expert,
        expert_alignment=ep_handle.expert_alignment,
        dtype=hidden_states.dtype,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        expected_m=expected_m,
        alpha=alpha,
        limit=limit,
        clamp_up_add_one=clamp_up_add_one,
    )
    # 阶段 1：把按 expert 展开的输出归约回去重接收 token 布局。
    dense_output, compact_metadata = ep_reduce_decode_output(
        expert_output=expert_output,
        route_weights=recv_topk_weights,
        recv_src_metadata=ep_handle.recv_src_metadata,
        num_valid_recv_tokens=ep_handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
    )
    del expert_output
    ep_handle.recv_src_metadata = compact_metadata

    # compact metadata 的第一个 top-k 槽位指向同序 dense row，其余槽位置为 -1。
    return _combine_with_optional_shared_expert(
        buffer,
        dense_output,
        ep_handle,
        previous_event=None,
        allocate_on_comm_stream=False,
        shared_expert_func=shared_expert_func,
    )


def _combine_with_optional_shared_expert(
    buffer,
    expert_output: torch.Tensor,
    handle,
    previous_event,
    allocate_on_comm_stream: bool,
    shared_expert_func: Optional[Callable[[], torch.Tensor]],
):
    """Launch combine, optionally running the shared expert while the transfer is in flight."""
    overlap = shared_expert_func is not None
    if overlap:
        # Combine waits for the routed GEMM, then the shared expert overlaps the transfer.
        from deep_ep import ElasticBuffer

        previous_event = ElasticBuffer.capture()
        allocate_on_comm_stream = True
    combined_x, _, event = buffer.combine(
        expert_output,
        handle,
        topk_weights=None,
        num_sms=get_ep_num_sms(overlap_with_compute=overlap),
        previous_event=previous_event,
        async_with_compute_stream=overlap,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    if allocate_on_comm_stream and combined_x.is_cuda:
        # 输出在通信流上分配，后续在计算流上使用。再登记一次计算流，
        # 引用归零时分配器才会等这条流上已经排队的 kernel 结束。
        combined_x.record_stream(torch.cuda.current_stream())
    if overlap:
        shared = shared_expert_func()
        event.current_stream_wait()
        combined_x.add_(shared)
    return combined_x
