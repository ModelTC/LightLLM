import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
from lightllm.common.triton_utils.autotuner import autotune


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
    HAS_SHARED_EXPERT_GATE: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    SHARED_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_TOKEN + tl.arange(0, BLOCK_TOKEN)
    token_mask = token_offsets < token_num
    topk_offsets = tl.arange(0, TOPK_BLOCK)
    topk_mask = topk_offsets < topk_num
    shared_expert_offsets = tl.arange(0, SHARED_BLOCK)
    shared_expert_mask = shared_expert_offsets < num_fused_shared_experts

    topk_in_offsets = token_offsets[:, None] * topk_num + topk_offsets[None, :]
    topk_out_offsets = token_offsets[:, None] * out_topk_num + topk_offsets[None, :]
    topk_valid_mask = token_mask[:, None] & topk_mask[None, :]
    topk_ids = tl.load(topk_ids_ptr + topk_in_offsets, mask=topk_valid_mask, other=0)
    topk_weights = tl.load(topk_weights_ptr + topk_in_offsets, mask=topk_valid_mask, other=0.0)
    tl.store(out_topk_ids_ptr + topk_out_offsets, topk_ids, mask=topk_valid_mask)
    tl.store(out_topk_weights_ptr + topk_out_offsets, topk_weights, mask=topk_valid_mask)

    shared_out_offsets = token_offsets[:, None] * out_topk_num + topk_num + shared_expert_offsets[None, :]
    shared_valid_mask = token_mask[:, None] & shared_expert_mask[None, :]
    shared_ids = shared_expert_start_id + shared_expert_offsets
    tl.store(out_topk_ids_ptr + shared_out_offsets, shared_ids[None, :], mask=shared_valid_mask)

    shared_weights = tl.full((BLOCK_TOKEN, SHARED_BLOCK), 1.0, tl.float32)
    if HAS_SHARED_EXPERT_GATE:
        gate_offsets = token_offsets[:, None] * shared_expert_gate_stride_0 + shared_expert_offsets[None, :]
        gate_vals = tl.load(shared_expert_gate_ptr + gate_offsets, mask=shared_valid_mask, other=0.0).to(tl.float32)
        shared_weights = tl.sigmoid(gate_vals)
    tl.store(out_topk_weights_ptr + shared_out_offsets, shared_weights, mask=shared_valid_mask)


def _get_append_fused_shared_experts_static_key(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_fused_shared_experts: int,
    shared_expert_gate: Optional[torch.Tensor] = None,
) -> dict:
    return {
        "topk_num": topk_ids.shape[1],
        "num_fused_shared_experts": num_fused_shared_experts,
        "has_shared_expert_gate": shared_expert_gate is not None,
        "topk_weights_dtype": str(topk_weights.dtype),
        "topk_ids_dtype": str(topk_ids.dtype),
    }


def _get_append_fused_shared_experts_configs():
    block_token_choices = (4, 8, 16, 32, 64, 128, 256)
    num_warps_choices = (1, 2, 4, 8)
    return [
        {"BLOCK_TOKEN": block_token, "num_warps": num_warps}
        for block_token in block_token_choices
        for num_warps in num_warps_choices
    ]


@torch.no_grad()
@autotune(
    kernel_name="append_fused_shared_experts:v1",
    configs_gen_func=_get_append_fused_shared_experts_configs,
    static_key_func=_get_append_fused_shared_experts_static_key,
    run_key_func=lambda topk_ids: topk_ids.shape[0],
)
def append_fused_shared_experts(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_expert_start_id: int,
    num_fused_shared_experts: int,
    shared_expert_gate: Optional[torch.Tensor] = None,
    run_config: Optional[dict] = None,
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
        assert shared_expert_gate.stride(1) == 1, "shared_expert_gate last dim must be contiguous"
    else:
        shared_expert_gate_stride_0 = 0

    if run_config is None:
        if token_num <= 1:
            run_config = {"BLOCK_TOKEN": 4, "num_warps": 2}
        elif token_num <= 4:
            run_config = {"BLOCK_TOKEN": 8, "num_warps": 4}
        elif token_num <= 65536:
            run_config = {"BLOCK_TOKEN": 32, "num_warps": 8}
        elif token_num <= 131072:
            run_config = {"BLOCK_TOKEN": 64, "num_warps": 8}
        elif token_num <= 262144:
            run_config = {"BLOCK_TOKEN": 128, "num_warps": 8}
        else:
            run_config = {"BLOCK_TOKEN": 256, "num_warps": 8}

    block_token = run_config["BLOCK_TOKEN"]
    num_warps = run_config["num_warps"]
    grid_num = triton.cdiv(token_num, block_token)
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
        HAS_SHARED_EXPERT_GATE=has_shared_expert_gate,
        BLOCK_TOKEN=block_token,
        TOPK_BLOCK=triton.next_power_of_2(topk_num),
        SHARED_BLOCK=triton.next_power_of_2(num_fused_shared_experts),
        num_warps=num_warps,
    )
    return out_topk_weights, out_topk_ids
