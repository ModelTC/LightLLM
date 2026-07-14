"""Fused MoE kernel."""
import torch
import triton
import triton.language as tl
from typing import Any, Optional, Tuple
from lightllm.distributed import dist_group_manager
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul_mix_quant_ep import (
    silu_and_mul_masked_post_quant_fwd,
)
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
    per_token_group_quant_fp8,
    tma_align_input_scale,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_scatter_gather import ep_scatter, ep_gather
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
)
from lightllm.common.triton_utils.autotuner import Autotuner
from lightllm.utils.device_utils import is_sm100_gpu

logger = init_logger(__name__)
SUPPORTED_EP_EXPERT_DTYPES = ("deepgemm-fp8w8a8-b128", "deepgemm-fp4fp8-b32")

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
    return is_sm100_gpu() and quant_method.method_name == "deepgemm-fp4fp8-b32"


def _per_token_cast_to_fp8_packed_ue8m0(hidden_states: torch.Tensor, gran_k: int):
    from deep_gemm.utils import per_token_cast_to_fp8

    hidden_states, scale = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=gran_k,
        use_packed_ue8m0=False,
    )
    assert scale.size(-1) % 4 == 0, "packed UE8M0 scale requires scale groups divisible by 4"
    scale = (scale.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)
    return hidden_states, scale


@triton.jit
def _ceil_to_ue8m0(x):
    bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
    exp = tl.maximum(tl.minimum(exp, 254), 1)
    return (exp << 23).to(tl.float32, bitcast=True), exp


@triton.jit
def _mega_moe_quant_topk_to_buffer_kernel(
    x_ptr,
    x_out_ptr,
    x_sf_out_ptr,
    topk_idx_ptr,
    topk_idx_out_ptr,
    topk_weights_ptr,
    topk_weights_out_ptr,
    stride_x_m: tl.constexpr,
    stride_x_k: tl.constexpr,
    stride_x_out_m: tl.constexpr,
    stride_x_out_k: tl.constexpr,
    stride_x_sf_out_m: tl.constexpr,
    stride_x_sf_out_k: tl.constexpr,
    stride_topk_idx_m: tl.constexpr,
    stride_topk_idx_k: tl.constexpr,
    stride_topk_idx_out_m: tl.constexpr,
    stride_topk_idx_out_k: tl.constexpr,
    stride_topk_weights_m: tl.constexpr,
    stride_topk_weights_k: tl.constexpr,
    stride_topk_weights_out_m: tl.constexpr,
    stride_topk_weights_out_k: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    pack_id = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    cols = pack_id * BLOCK + offsets

    x = tl.load(x_ptr + token_id * stride_x_m + cols * stride_x_k).to(tl.float32)
    abs_x = tl.abs(x)
    group_id = offsets // GROUP_SIZE

    amax0 = tl.max(tl.where(group_id == 0, abs_x, 0.0))
    amax1 = tl.max(tl.where(group_id == 1, abs_x, 0.0))
    amax2 = tl.max(tl.where(group_id == 2, abs_x, 0.0))
    amax3 = tl.max(tl.where(group_id == 3, abs_x, 0.0))

    scale0, exp0 = _ceil_to_ue8m0(tl.maximum(amax0, 1.0e-4) / FP8_MAX)
    scale1, exp1 = _ceil_to_ue8m0(tl.maximum(amax1, 1.0e-4) / FP8_MAX)
    scale2, exp2 = _ceil_to_ue8m0(tl.maximum(amax2, 1.0e-4) / FP8_MAX)
    scale3, exp3 = _ceil_to_ue8m0(tl.maximum(amax3, 1.0e-4) / FP8_MAX)

    scale = tl.where(
        group_id == 0,
        scale0,
        tl.where(group_id == 1, scale1, tl.where(group_id == 2, scale2, scale3)),
    )
    x_q = tl.clamp(x / scale, FP8_MIN, FP8_MAX).to(x_out_ptr.dtype.element_ty)
    tl.store(x_out_ptr + token_id * stride_x_out_m + cols * stride_x_out_k, x_q)

    packed_scale = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
    tl.store(x_sf_out_ptr + token_id * stride_x_sf_out_m + pack_id * stride_x_sf_out_k, packed_scale)

    if pack_id == 0:
        topk_offsets = tl.arange(0, TOPK_BLOCK)
        topk_mask = topk_offsets < TOPK
        topk_idx = tl.load(
            topk_idx_ptr + token_id * stride_topk_idx_m + topk_offsets * stride_topk_idx_k,
            mask=topk_mask,
        )
        topk_weights = tl.load(
            topk_weights_ptr + token_id * stride_topk_weights_m + topk_offsets * stride_topk_weights_k,
            mask=topk_mask,
        )
        tl.store(
            topk_idx_out_ptr + token_id * stride_topk_idx_out_m + topk_offsets * stride_topk_idx_out_k,
            topk_idx.to(topk_idx_out_ptr.dtype.element_ty),
            mask=topk_mask,
        )
        tl.store(
            topk_weights_out_ptr + token_id * stride_topk_weights_out_m + topk_offsets * stride_topk_weights_out_k,
            topk_weights.to(topk_weights_out_ptr.dtype.element_ty),
            mask=topk_mask,
        )


def _prepare_mega_moe_buffer(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buffer: Any,
    group_size: int,
):
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    assert hidden_size % (group_size * 4) == 0, "packed UE8M0 scale requires four FP8 groups per int32"
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert topk_ids.shape == topk_weights.shape and topk_ids.shape[0] == num_tokens
    assert topk_ids.shape[1] > 0
    assert buffer.x.shape[0] >= num_tokens and buffer.x.shape[1] == hidden_size
    assert buffer.x_sf.shape[0] >= num_tokens and buffer.x_sf.shape[1] == hidden_size // group_size // 4
    assert buffer.topk_idx.shape[0] >= num_tokens and buffer.topk_idx.shape[1] == topk_ids.shape[1]
    assert buffer.topk_weights.shape[0] >= num_tokens and buffer.topk_weights.shape[1] == topk_ids.shape[1]

    block = group_size * 4
    finfo = torch.finfo(buffer.x.dtype)
    _mega_moe_quant_topk_to_buffer_kernel[(num_tokens, hidden_size // block)](
        hidden_states,
        buffer.x,
        buffer.x_sf,
        topk_ids,
        buffer.topk_idx,
        topk_weights,
        buffer.topk_weights,
        hidden_states.stride(0),
        hidden_states.stride(1),
        buffer.x.stride(0),
        buffer.x.stride(1),
        buffer.x_sf.stride(0),
        buffer.x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        buffer.topk_idx.stride(0),
        buffer.topk_idx.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        buffer.topk_weights.stride(0),
        buffer.topk_weights.stride(1),
        FP8_MIN=finfo.min,
        FP8_MAX=finfo.max,
        TOPK=topk_ids.shape[1],
        TOPK_BLOCK=triton.next_power_of_2(topk_ids.shape[1]),
        GROUP_SIZE=group_size,
        BLOCK=block,
        num_warps=4,
        num_stages=4,
    )


def check_ep_expert_dtype(quant_method: Any):
    expert_dtype = getattr(quant_method, "method_name", None)
    if expert_dtype not in SUPPORTED_EP_EXPERT_DTYPES:
        raise ValueError(
            "EP MoE requires --expert_dtype to be one of ['fp8', 'fp4'], "
            f"but the resolved fused_moe quant method is `{expert_dtype}`. "
            "Please start with --expert_dtype fp8 or --expert_dtype fp4. "
            "Note that --expert_dtype fp4 is only supported on SM100 GPUs."
        )
    if expert_dtype == "deepgemm-fp4fp8-b32" and not is_sm100_gpu():
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

    _deepgemm_grouped_fp8_nt_masked(recv_x, (w1, w1_scale), gemm_out_a, masked_m, expected_m)

    silu_and_mul_masked_post_quant_fwd(gemm_out_a, qsilu_out, qsilu_out_scale, block_size, masked_m)
    _deepgemm_grouped_fp8_nt_masked((qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, masked_m, expected_m)
    return gemm_out_b


def transform_mega_moe_weights_in_place(w13: Any, w2: Any):
    """Convert to Mega MoE layout without retaining a second weight copy."""
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
        (w13.weight, w13.weight_scale),
        (w2.weight, w2.weight_scale),
    )
    w13.weight.copy_(transformed_l1[0])
    w13.weight_scale.copy_(transformed_l1[1])
    w2.weight_scale.copy_(transformed_l2[1])


def mega_moe_impl(
    hidden_states: torch.Tensor,
    w13: Any,
    w2: Any,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_method: Any,
):
    if not (HAS_DEEPGEMM and hasattr(deep_gemm, "fp8_fp4_mega_moe")):
        raise RuntimeError("deep_gemm does not provide fp8-fp4 Mega MoE kernel")

    buffer = getattr(dist_group_manager, "ep_mega_moe_buffer", None)
    if buffer is None:
        raise RuntimeError("SM100 Mega MoE requires dist_group_manager.ep_mega_moe_buffer to be initialized")

    num_tokens = hidden_states.shape[0]
    if num_tokens > buffer.num_max_tokens_per_rank:
        raise RuntimeError(
            f"Mega MoE got {num_tokens} tokens, exceeding num_max_tokens_per_rank={buffer.num_max_tokens_per_rank}"
        )

    l1_weights = (w13.weight, w13.weight_scale)
    l2_weights = (w2.weight, w2.weight_scale)
    _prepare_mega_moe_buffer(hidden_states, topk_ids, topk_weights, buffer, quant_method.block_size)

    output = torch.empty_like(hidden_states)
    deep_gemm.fp8_fp4_mega_moe(
        output,
        l1_weights,
        l2_weights,
        buffer,
    )
    return output


def quantize_fused_experts_input(
    hidden_states: torch.Tensor,
    w13: Any,
    quant_method: Any,
):
    check_ep_expert_dtype(quant_method)
    if use_sm100_mega_moe(quant_method):
        return _per_token_cast_to_fp8_packed_ue8m0(hidden_states, quant_method.block_size)

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
):
    check_ep_expert_dtype(quant_method)
    if use_sm100_mega_moe(quant_method):
        return mega_moe_impl(hidden_states, w13, w2, topk_weights, topk_idx, quant_method)

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
        qinput_tensor, input_scale = per_token_group_quant_fp8(hidden_states, block_size_k, dtype=w1.dtype)
        allocate_on_comm_stream = previous_event is not None
        # normal dispatch
        # recv_x [recive_num_tokens, hidden] recv_x_scale [recive_num_tokens, hidden // block_size]
        # recv_topk_idx [recive_num_tokens, topk_num]
        # recv_topk_weights [recive_num_tokens, topk_num]
        # num_recv_tokens_per_expert_list list [cur_node_expert_num] padding with expert_alignment=128
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
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
        )

        # scatter
        all_tokens = sum(handle.num_recv_tokens_per_expert_list)  # calcu padding all nums.
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
                handle.num_recv_tokens_per_expert_list, dtype=torch.int32, pin_memory=True, device="cpu"
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
            deepgemm_grouped_fp8_nt_contiguous(input_tensor, (w1, w1_scale), gemm_out_a, m_indices)

            # silu_and_mul_fwd + qaunt
            # TODO fused kernel
            silu_out = torch.empty((all_tokens, N // 2), device=hidden_states.device, dtype=hidden_states.dtype)

            silu_and_mul_fwd(gemm_out_a.view(-1, N), silu_out)
            qsilu_out, qsilu_out_scale = per_token_group_quant_fp8(
                silu_out, block_size_k, dtype=w1.dtype, column_major_scales=True, scale_tma_aligned=True
            )

            # groupgemm (contiguous layout)
            gemm_out_b = torch.empty((all_tokens, K), device=hidden_states.device, dtype=hidden_states.dtype)

            deepgemm_grouped_fp8_nt_contiguous((qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, m_indices)

            # gather and local reduce
            ep_gather(gemm_out_b, recv_topk_idx, recv_topk_weights, output_index, gather_out)
        else:
            ######################################## warning ##################################################
            # here is used to match autotune feature, make moe model run same triton kernel in different rank.
            # in some special case, one rank will recv 0 token, so add a token to make it run triton kernel.
            if Autotuner.is_autotune_warmup():
                _gemm_out_a = torch.zeros((1, N), device=hidden_states.device, dtype=hidden_states.dtype)
                _silu_out = torch.zeros((1, N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
                silu_and_mul_fwd(_gemm_out_a.view(-1, N), _silu_out)
                _gemm_out_a, _silu_out = None, None

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
        gemm_out_b = masked_group_gemm(recv_x, masked_m, hidden_states.dtype, w1, w1_scale, w2, w2_scale, expected_m)
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
