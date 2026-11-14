import torch
import triton
import triton.language as tl

from lightllm.distributed import dist_group_manager
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import ep_reduce_decode_output
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu
from .common import get_ep_num_sms
from .deepgemm import deep_gemm, get_mk_alignment_for_contiguous_layout


@triton.jit
def _expert_tile(
    Input,
    Weight,
    Output,
    expert,
    base,
    end,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    row = base + tl.arange(0, BM)
    acc = tl.full((BM, BN), 0, tl.float32)
    for first in range(tl.cdiv(K, BK)):
        k = first * BK + inner
        x = tl.load(
            Input + row[:, None].to(tl.int64) * K + k[None, :],
            (row[:, None] < end) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            Weight + (expert * N + col[None, :]).to(tl.int64) * K + k[:, None],
            (col[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(x, w, acc)
    tl.store(
        Output + row[:, None].to(tl.int64) * N + col[None, :],
        acc,
        (row[:, None] < end) & (col[None, :] < N),
    )


@triton.jit
def _grouped_gemm(
    Input,
    Weight,
    Output,
    ExpertEnds,
    K: tl.constexpr,
    N: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    expert = tl.program_id(0)
    end = tl.load(ExpertEnds + expert)
    previous = tl.load(ExpertEnds + expert - 1, expert > 0, other=0)
    start = tl.cdiv(previous, ALIGNMENT) * ALIGNMENT
    if end > start:
        if end - start <= BM:
            _expert_tile(Input, Weight, Output, expert, start, end, K, N, BM, BN, BK)
        else:
            for base in range(start, end, BM):
                _expert_tile(Input, Weight, Output, expert, base, end, K, N, BM, BN, BK)


def _expert_gemm(input, weight, output, expert_ends, expected_m, is_prefill):
    if not is_prefill and expected_m <= 8:
        _grouped_gemm[(weight.shape[0], triton.cdiv(weight.shape[1], 64))](
            input,
            weight,
            output,
            expert_ends,
            input.shape[1],
            weight.shape[1],
            get_mk_alignment_for_contiguous_layout(),
            16,
            64,
            256 if input.shape[1] >= 2048 and weight.shape[1] <= 2048 else 128,
            num_warps=4,
            num_stages=3,
        )
    else:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            input,
            weight,
            output,
            expert_ends,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m,
        )


@triton.jit
def _activate_valid_rows(
    Up,
    Activated,
    ExpertEnds,
    N: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    ADD_ONE: tl.constexpr,
    GELU: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SHARDS: tl.constexpr = 1,
):
    expert = tl.program_id(0)
    end = tl.load(ExpertEnds + expert)
    previous = tl.load(ExpertEnds + expert - 1, expert > 0, other=0)
    start = tl.cdiv(previous, ALIGNMENT) * ALIGNMENT
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    for base in range(start + tl.program_id(2) * BM, end, SHARDS * BM):
        row = base + tl.arange(0, BM)
        mask = (row[:, None] < end) & (col[None, :] < N)
        gate = tl.load(Up + row[:, None].to(tl.int64) * (2 * N) + col[None, :], mask, other=0.0).to(tl.float32)
        value = tl.load(Up + row[:, None].to(tl.int64) * (2 * N) + N + col[None, :], mask, other=0.0)
        if LIMIT is not None:
            gate = tl.minimum(gate, LIMIT)
            value = tl.minimum(tl.maximum(value, -LIMIT), LIMIT)
            gate = gate / (1.0 + tl.exp(-gate * ALPHA))
            if ADD_ONE:
                value += 1
        elif GELU:
            arg = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            gate = 0.5 * gate * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * arg)) - 1.0))
        else:
            gate = gate / (1.0 + tl.exp(-gate))
        gate = gate.to(Up.dtype.element_ty)
        tl.store(Activated + row[:, None].to(tl.int64) * N + col[None, :], value * gate, mask)


def fused_experts_bf16(
    hidden_states,
    w13,
    w2,
    topk_weights,
    topk_idx,
    num_experts,
    is_prefill,
    alpha=None,
    limit=None,
    clamp_up_add_one=True,
    shared_expert_func=None,
):
    from .prefill_decode_sync import _combine_with_optional_shared_expert

    if hidden_states.dtype != torch.bfloat16 or w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("Unquantized DeepEP experts require BF16 activations and weights")
    sync_prefill = is_prefill and not torch.cuda.is_current_stream_capturing()
    buffer = dist_group_manager.ep_buffer
    capacity = (
        get_deepep_num_max_dispatch_tokens_per_rank_prefill()
        if is_prefill
        else get_deepep_num_max_dispatch_tokens_per_rank_decode()
    )
    recv_x, _, recv_weights, handle, _ = buffer.dispatch(
        hidden_states,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=capacity,
        expert_alignment=get_mk_alignment_for_contiguous_layout(),
        num_sms=get_ep_num_sms(overlap_with_compute=False),
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
        do_cpu_sync=sync_prefill,
        do_handle_copy=False,
        do_expand=True,
        do_zero_padding=False,
    )
    rows = recv_x.shape[0]
    if rows:
        expected_m = max(1, triton.cdiv(hidden_states.shape[0] * buffer.num_ranks * topk_idx.shape[1], num_experts))
        up = torch.empty((rows, w13.shape[1]), device=recv_x.device, dtype=recv_x.dtype)
        _expert_gemm(
            recv_x,
            w13,
            up,
            handle.psum_num_recv_tokens_per_expert,
            expected_m,
            is_prefill,
        )
        activated = torch.empty((rows, w13.shape[1] // 2), device=recv_x.device, dtype=recv_x.dtype)
        if sync_prefill:
            silu_and_mul_fwd(up, activated, alpha=alpha, limit=limit, clamp_up_add_one=clamp_up_add_one)
        else:
            activation_shards = 8 if is_prefill and hidden_states.shape[0] >= 128 else 1
            _activate_valid_rows[(w13.shape[0], triton.cdiv(activated.shape[1], 128), activation_shards)](
                up,
                activated,
                handle.psum_num_recv_tokens_per_expert,
                activated.shape[1],
                get_mk_alignment_for_contiguous_layout(),
                alpha,
                limit,
                clamp_up_add_one,
                ffn_use_tanh_approximate_gelu(),
                16,
                128,
                activation_shards,
                num_warps=4,
            )
        del recv_x, up
        expert_output = torch.empty((rows, w2.shape[1]), device=activated.device, dtype=activated.dtype)
        _expert_gemm(
            activated,
            w2,
            expert_output,
            handle.psum_num_recv_tokens_per_expert,
            expected_m,
            is_prefill,
        )
        del activated
    else:
        expert_output = torch.empty((0, w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
        if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.GENERAL):
            up = torch.zeros((1, w13.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
            activated = torch.empty((1, w13.shape[1] // 2), device=hidden_states.device, dtype=hidden_states.dtype)
            silu_and_mul_fwd(up, activated, alpha=alpha, limit=limit, clamp_up_add_one=clamp_up_add_one)
    dense, metadata = ep_reduce_decode_output(
        expert_output=expert_output,
        route_weights=recv_weights,
        recv_src_metadata=handle.recv_src_metadata,
        num_valid_recv_tokens=handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
    )
    handle.recv_src_metadata = metadata
    return _combine_with_optional_shared_expert(
        buffer,
        dense,
        handle,
        None,
        False,
        shared_expert_func,
    )
