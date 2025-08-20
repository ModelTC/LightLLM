"""Fused MoE kernel."""
import os
import torch
import triton
import triton.language as tl
from typing import Any, Callable, Dict, Optional, Tuple
import torch.distributed as dist
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.fused_moe.moe_silu_and_mul_mix_quant_ep import silu_and_mul_masked_post_quant_fwd
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
    tma_align_input_scale,
)
from lightllm.common.fused_moe.deepep_scatter_gather import ep_scatter, ep_gather
from lightllm.utils.envs_utils import get_deepep_num_max_dispatch_tokens_per_rank
import numpy as np

logger = init_logger(__name__)

try:
    from deep_ep import Buffer, EventOverlap
    import deep_gemm

except:
    logger.warning("no deepep or deep_gemm")


def tma_aligned_quantize(
    input_tensor: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, k = input_tensor.shape
    input_scale = torch.empty((m, k // block_size), dtype=torch.float32, device=input_tensor.device)
    qinput_tensor = torch.empty((m, k), dtype=dtype, device=input_tensor.device)
    per_token_group_quant_fp8(input_tensor, block_size, qinput_tensor, input_scale)
    input_scale = tma_align_input_scale(input_scale)
    return qinput_tensor, input_scale


def masked_group_gemm(
    recv_x: Tuple[torch.Tensor],
    masked_m: torch.Tensor,
    dtype: torch.dtype,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    expected_m: int,
):
    padded_m = recv_x[0].shape[1]
    E, N, _ = w1.shape
    block_size = 128
    # groupgemm (masked layout)
    gemm_out_a = torch.empty((E, padded_m, N), device=recv_x[0].device, dtype=dtype)
    expected_m = min(expected_m, padded_m)
    qsilu_out_scale = torch.empty((E, padded_m, N // 2 // block_size), device=recv_x[0].device, dtype=torch.float32)
    qsilu_out = torch.empty((E, padded_m, N // 2), dtype=w1.dtype, device=recv_x[0].device)
    # groupgemm (masked layout)
    gemm_out_b = torch.empty_like(recv_x[0], device=recv_x[0].device, dtype=dtype)

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(recv_x, (w1, w1_scale), gemm_out_a, masked_m, expected_m)

    silu_and_mul_masked_post_quant_fwd(gemm_out_a, qsilu_out, qsilu_out_scale, block_size, masked_m)
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
        (qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, masked_m, expected_m
    )
    return gemm_out_b


def fused_experts_impl(
    hidden_states: torch.Tensor,  # [M, K]
    w1: torch.Tensor,  # [group, N, K]
    w2: torch.Tensor,  # [group, K, N/2]
    topk_weights: torch.Tensor,  # [M, topk]
    topk_idx: torch.Tensor,  # [M, topk]
    num_experts: int,
    buffer: "Buffer",
    is_prefill: bool,
    use_fp8_w8a8: bool = False,
    use_fp8_all2all: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    previous_event: Optional["EventOverlap"] = None,
):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_idx.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    M, K = hidden_states.shape
    E, N, _ = w1.shape

    # qaunt hidden_states
    assert use_fp8_w8a8 and use_fp8_all2all, "use_fp8_w8a8 and use_fp8_all2all must be True"

    block_size_k = 0

    if w1.ndim == 3:
        block_size_k = w1.shape[2] // w1_scale.shape[2]

    assert block_size_k == 128, "block_size_k must be 128"

    combined_x = None
    if is_prefill:
        input_scale = torch.empty((M, K // block_size_k), dtype=torch.float32, device=hidden_states.device)
        qinput_tensor = torch.empty((M, K), dtype=w1.dtype, device=hidden_states.device)
        per_token_group_quant_fp8(hidden_states, block_size_k, qinput_tensor, input_scale)

        # get_dispatch_layout
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_idx, num_experts, previous_event=previous_event, async_finish=False, allocate_on_comm_stream=False
        )

        # normal dispatch
        # recv_x [recive_num_tokens, hidden] recv_x_scale [recive_num_tokens, hidden // block_size]
        # recv_topk_idx [recive_num_tokens, topk_num]
        # recv_topk_weights [recive_num_tokens, topk_num]
        # num_recv_tokens_per_expert_list list [cur_node_expert_num] padding with expert_alignment=128
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = buffer.dispatch(
            (qinput_tensor, input_scale),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
            expert_alignment=128,
        )

        # scatter
        all_tokens = sum(num_recv_tokens_per_expert_list)  # calcu padding all nums.

        if get_env_start_args().enable_ep_fake_balance:
            rank = dist.get_rank()
            if rank == 0:
                logger.info(
                    f"prefill, [{rank}], all_tokens = {all_tokens}, "
                    f"num_recv_tokens_per_expert_list: {num_recv_tokens_per_expert_list}"
                )

        # gather_out shape [recive_num_tokens, hidden]
        gather_out = torch.empty_like(recv_x[0], device=hidden_states.device, dtype=hidden_states.dtype)
        if all_tokens > 0:
            input_tensor = [
                torch.empty((all_tokens, K), device=hidden_states.device, dtype=qinput_tensor.dtype),
                torch.empty((all_tokens, K // 128), device=hidden_states.device, dtype=torch.float32),
            ]
            # when m_indices is filled ok.
            # m_indices show token use which expert, example, [0, 0, 0, 0, .... 1, 1, 1, 1,...., cur_expert_num - 1, ..]
            # the count of 0 is num_recv_tokens_per_expert_list[0], the count of 1 is num_recv_tokens_per_expert_list[1]
            # ...
            m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
            # output_index shape [recive_num_tokens, topk_num]
            # output_index use to show the token index in input_tensor
            output_index = torch.empty_like(recv_topk_idx)

            num_recv_tokens_per_expert = torch.tensor(
                num_recv_tokens_per_expert_list, dtype=torch.int32, pin_memory=True, device="cpu"
            ).cuda(non_blocking=True)

            expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)

            ep_scatter(
                recv_x[0],
                recv_x[1],
                recv_topk_idx,
                num_recv_tokens_per_expert,
                expert_start_loc,
                input_tensor[0],
                input_tensor[1],
                m_indices,
                output_index,
            )

            # groupgemm (contiguous layout)
            gemm_out_a = torch.empty((all_tokens, N), device=hidden_states.device, dtype=hidden_states.dtype)
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(input_tensor, (w1, w1_scale), gemm_out_a, m_indices)

            # silu_and_mul_fwd + qaunt
            # TODO fused kernel
            silu_out = torch.empty((all_tokens, N // 2), device=hidden_states.device, dtype=hidden_states.dtype)

            silu_and_mul_fwd(gemm_out_a.view(-1, N), silu_out)
            qsilu_out, qsilu_out_scale = tma_aligned_quantize(silu_out)

            # groupgemm (contiguous layout)
            gemm_out_b = torch.empty((all_tokens, K), device=hidden_states.device, dtype=hidden_states.dtype)

            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, m_indices
            )

            # gather and local reduce
            ep_gather(gemm_out_b, recv_topk_idx, recv_topk_weights, output_index, gather_out)
        # normal combine
        combined_x, _, event = buffer.combine(
            gather_out,
            handle,
            topk_weights=None,
            async_finish=False,
            previous_event=previous_event,
            allocate_on_comm_stream=False,
        )
    else:
        # low latency dispatch
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank()
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

        # NOTE: when decoding graph is open, we can not call logger. Thus it can only be used when --disable_cudagraph
        args = get_env_start_args()
        if args.enable_ep_fake_balance and args.disable_cudagraph:
            rank = dist.get_rank()
            all_tokens = sum(masked_m)
            if rank == 0:
                logger.info(
                    f"decode, [{rank}], all_tokens = {all_tokens}, "
                    f"expected_m = {expected_m}, num_recv_tokens_per_expert: {masked_m}"
                )

        # deepgemm
        gemm_out_b = masked_group_gemm(recv_x, masked_m, hidden_states.dtype, w1, w1_scale, w2, w2_scale, expected_m)
        # low latency combine
        combined_x, event_overlap, hook = buffer.low_latency_combine(
            gemm_out_b, topk_idx, topk_weights, handle, async_finish=False, return_recv_hook=False
        )
    return combined_x
