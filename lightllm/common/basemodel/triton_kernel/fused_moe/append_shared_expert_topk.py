import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def _append_fused_shared_experts_kernel(
    topk_weights_ptr,
    topk_ids_ptr,
    shared_expert_gate_ptr,
    out_topk_weights_ptr,
    out_topk_ids_ptr,
    token_num,
    topk_num: tl.constexpr,
    out_topk_num: tl.constexpr,
    shared_expert_start_id: tl.constexpr,
    num_fused_shared_experts: tl.constexpr,
    shared_expert_gate_stride_0: tl.constexpr,
    shared_expert_gate_stride_1: tl.constexpr,
    HAS_SHARED_EXPERT_GATE: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    SHARED_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_num = tl.num_programs(0)
    topk_offsets = tl.arange(0, TOPK_BLOCK)
    topk_mask = topk_offsets < topk_num
    shared_expert_offsets = tl.arange(0, SHARED_BLOCK)
    shared_expert_mask = shared_expert_offsets < num_fused_shared_experts

    for token_idx in tl.range(pid, token_num, grid_num, num_stages=4):
        topk_in_offsets = token_idx * topk_num + topk_offsets
        topk_out_offsets = token_idx * out_topk_num + topk_offsets
        topk_ids = tl.load(topk_ids_ptr + topk_in_offsets, mask=topk_mask, other=0)
        topk_weights = tl.load(topk_weights_ptr + topk_in_offsets, mask=topk_mask, other=0.0)
        tl.store(out_topk_ids_ptr + topk_out_offsets, topk_ids, mask=topk_mask)
        tl.store(out_topk_weights_ptr + topk_out_offsets, topk_weights, mask=topk_mask)

        shared_weights = tl.full((SHARED_BLOCK,), 1.0, tl.float32)
        if HAS_SHARED_EXPERT_GATE:
            gate_offsets = token_idx * shared_expert_gate_stride_0 + shared_expert_offsets * shared_expert_gate_stride_1
            gate_vals = tl.load(shared_expert_gate_ptr + gate_offsets, mask=shared_expert_mask, other=0.0).to(
                tl.float32
            )
            shared_weights = tl.sigmoid(gate_vals)

        shared_out_offsets = token_idx * out_topk_num + topk_num + shared_expert_offsets
        tl.store(
            out_topk_ids_ptr + shared_out_offsets,
            shared_expert_start_id + shared_expert_offsets,
            mask=shared_expert_mask,
        )
        tl.store(out_topk_weights_ptr + shared_out_offsets, shared_weights, mask=shared_expert_mask)


@torch.no_grad()
def append_fused_shared_experts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_expert_start_id: int,
    num_fused_shared_experts: int,
    shared_expert_gate: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_weights.dim() == 2 and topk_ids.dim() == 2
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert num_fused_shared_experts > 0

    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    token_num, topk_num = topk_ids.shape
    out_topk_num = topk_num + num_fused_shared_experts
    out_topk_weights = torch.empty((token_num, out_topk_num), dtype=topk_weights.dtype, device=topk_weights.device)
    out_topk_ids = torch.empty((token_num, out_topk_num), dtype=topk_ids.dtype, device=topk_ids.device)

    has_shared_expert_gate = shared_expert_gate is not None
    if has_shared_expert_gate:
        shared_expert_gate = shared_expert_gate.contiguous().view(token_num, -1)
        assert shared_expert_gate.shape[1] == num_fused_shared_experts, "shared_expert_gate shape mismatch"
        shared_expert_gate_stride_0 = shared_expert_gate.stride(0)
        shared_expert_gate_stride_1 = shared_expert_gate.stride(1)
    else:
        shared_expert_gate_stride_0 = 0
        shared_expert_gate_stride_1 = 0

    max_grid_num = 2048
    grid_num = min(token_num, max_grid_num)
    grid = (grid_num,)
    _append_fused_shared_experts_kernel[grid](
        topk_weights,
        topk_ids,
        shared_expert_gate,
        out_topk_weights,
        out_topk_ids,
        token_num,
        topk_num=topk_num,
        out_topk_num=out_topk_num,
        shared_expert_start_id=shared_expert_start_id,
        num_fused_shared_experts=num_fused_shared_experts,
        shared_expert_gate_stride_0=shared_expert_gate_stride_0,
        shared_expert_gate_stride_1=shared_expert_gate_stride_1,
        HAS_SHARED_EXPERT_GATE=has_shared_expert_gate,
        TOPK_BLOCK=triton.next_power_of_2(topk_num),
        SHARED_BLOCK=triton.next_power_of_2(num_fused_shared_experts),
        num_warps=1,
    )
    return out_topk_weights, out_topk_ids
