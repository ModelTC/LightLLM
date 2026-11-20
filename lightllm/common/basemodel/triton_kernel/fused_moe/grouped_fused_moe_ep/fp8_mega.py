import torch

from lightllm.distributed import dist_group_manager
from lightllm.common.basemodel.triton_kernel.fused_moe.sm90_fp8_mega_moe import sm90_mega_moe_forward


def use_sm90_mega_moe(quant_method):
    return dist_group_manager.ep_mega_moe_mma_type == "fp8xfp8" and quant_method.method_name == "fp8w8a8-b128-deepgemm"


def mega_moe_impl(hidden_states, w13, w2, topk_weights, topk_ids, limit=None):
    buffer = dist_group_manager.ep_mega_moe_buffer
    if buffer is None:
        raise RuntimeError("SM90 MegaMoE requires an initialized symmetric buffer")
    num_tokens = hidden_states.shape[0]
    if num_tokens > buffer.num_max_tokens_per_rank:
        raise RuntimeError(
            f"MegaMoE got {num_tokens} tokens, exceeding num_max_tokens_per_rank={buffer.num_max_tokens_per_rank}"
        )
    stats = getattr(w13, "_mega_moe_stats", None)
    if stats is None:
        stats = torch.zeros(w13.weight.shape[0], device=hidden_states.device, dtype=torch.int32)
        w13._mega_moe_stats = stats
    return sm90_mega_moe_forward(hidden_states, w13, w2, topk_weights, topk_ids, buffer, stats, limit, torch.empty)
