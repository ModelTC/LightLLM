import torch
from typing import Any, Dict, Tuple

from lightllm.distributed import dist_group_manager
from lightllm.utils.device_utils import is_sm100_gpu

from .deepgemm import HAS_DEEPGEMM, deep_gemm

_MEGA_MOE_STATES: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}


def use_sm100_mega_moe(quant_method: Any) -> bool:
    return is_sm100_gpu() and quant_method.method_name == "fp4fp8-b32-deepgemm"


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

    output = torch.empty_like(hidden_states)
    deep_gemm.fp8_fp4_mega_moe(
        output,
        l1_weights,
        l2_weights,
        buffer,
        cumulative_local_expert_recv_stats=stats,
    )
    return output


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
