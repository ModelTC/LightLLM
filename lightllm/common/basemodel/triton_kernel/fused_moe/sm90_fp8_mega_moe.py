import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def get_sm90_mega_moe_module():
    required = ("fp8_mega_moe", "mega_moe_pre_dispatch_sm90", "get_symm_buffer_for_sm90_mega_moe")
    for name in ("deep_gemm", "sgl_deep_gemm"):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            continue
        if all(hasattr(module, attr) for attr in required):
            return module
    raise RuntimeError(
        "SM90 FP8 MegaMoE requires the SGLang DeepGEMM FP8 API "
        "(fp8_mega_moe, mega_moe_pre_dispatch_sm90, get_symm_buffer_for_sm90_mega_moe). "
        "See test/start_scripts/glm53/README_megamoe.md for the tested dependency revision."
    )


def prepare_sm90_mega_moe_weights(w13, w2):
    weights = getattr(w13, "_mega_moe_weights", None)
    if weights is not None:
        return weights
    get_sm90_mega_moe_module()
    for pack in (w13, w2):
        weight, scale = pack.weight, pack.weight_scale
        if weight.ndim != 3 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("SM90 MegaMoE requires FP8 E4M3 expert weights with shape [experts, N, K]")
        experts, n, k = weight.shape
        if n % 128 or k % 128:
            raise ValueError("SM90 MegaMoE expert dimensions must be multiples of 128")
        if scale.dtype != torch.float32 or scale.shape != (experts, n // 128, k // 128):
            raise ValueError("SM90 MegaMoE requires FP32 block scales with shape [experts, N/128, K/128]")
        if not weight.is_contiguous() or not scale.is_contiguous():
            raise ValueError("SM90 MegaMoE requires contiguous expert weights and block scales")
    experts, n, k = w13.weight.shape
    if w2.weight.shape != (experts, k, n // 2):
        raise ValueError("SM90 MegaMoE gate/up and down projection shapes do not match")
    gate, up = w13.weight.view(experts, 2, n // 2 // 8, 8, k).unbind(1)
    interleaved = torch.stack((gate, up), dim=2).reshape_as(w13.weight)
    w13.weight.copy_(interleaved)
    weights = ((w13.weight, w13.weight_scale), (w2.weight, w2.weight_scale))
    w13._mega_moe_weights = weights
    return weights


def sm90_mega_moe_forward(hidden_states, w13, w2, topk_weights, topk_ids, buffer, stats, limit, alloc_tensor_func):
    module = get_sm90_mega_moe_module()
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("SM90 FP8 MegaMoE requires BF16 hidden states")
    num_tokens = hidden_states.shape[0]
    module.mega_moe_pre_dispatch_sm90(
        hidden_states,
        topk_ids.to(torch.int32),
        topk_weights,
        buffer.x,
        buffer.x_sf,
        buffer.topk_idx,
        buffer.topk_weights,
        num_tokens=num_tokens,
        group_size=128,
        routed_scaling_factor=1.0,
    )
    weights = prepare_sm90_mega_moe_weights(w13, w2)
    output = alloc_tensor_func(
        (max(num_tokens, 1), hidden_states.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype
    )
    module.fp8_mega_moe(
        output,
        *weights,
        buffer,
        cumulative_local_expert_recv_stats=stats,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=limit,
    )
    return output[:num_tokens]
