"""EPLB initialization, transfer and model profiling helpers."""

import math


EPLB_FULL_STAGING_PARTS = 1
EPLB_STEADY_SAMPLE_STEPS = 4


def is_eplb_enabled(args) -> bool:
    """Resolve and validate EPLB at initialization; PD decode keeps its own layout."""
    if not args.enable_prefill_eplb or args.run_mode == "decode":
        return False
    if args.run_mode not in ("prefill", "normal"):
        raise ValueError("EPLB requires --run_mode prefill or single-node normal")
    if args.run_mode == "normal" and getattr(args, "nnodes", 1) != 1:
        raise ValueError("EPLB normal mode requires a single node")
    assert args.enable_ep_moe, "--enable_prefill_eplb requires --enable_ep_moe"
    assert not args.enable_prefill_cudagraph, "--enable_prefill_eplb does not support --enable_prefill_cudagraph"
    # Online updates use the initial expert-row map and do not invalidate staged migrations.
    assert not args.enable_rl, "--enable_prefill_eplb does not support --enable_rl"

    # Keep disabled/PD decode resolution and planner imports independent of CUDA.
    from lightllm.utils.device_utils import is_sm100_gpu

    # SM100 Mega-MoE caches transformed weights by data_ptr; EPLB updates them in place.
    assert not is_sm100_gpu(), "--enable_prefill_eplb does not support SM100"
    assert args.eplb_num_redundant_experts_per_rank >= 0, "redundant expert count must be nonnegative"
    return True


def find_ep_moe_weights(model):
    # Deferred import keeps planner/transfer helpers independent of model imports.
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight

    weights = {}
    for layer in model.trans_layers_weight:
        for weight in getattr(layer, "__dict__", {}).values():
            if isinstance(weight, FusedMoeWeight) and weight.enable_ep_moe:
                weights[id(weight)] = weight
    return sorted(weights.values(), key=lambda weight: weight.layer_num_)


def get_eplb_staging_shape(weights):
    """Stage one complete layer before modifying any live expert rows."""
    if not weights:
        return 0, 0
    parallel = weights[0].expert_parallel_state
    return 1, parallel.num_primary_experts_per_rank + parallel.eplb.num_redundant_experts_per_rank


def get_eplb_staging_tensor_shape(tensor, staged_rows):
    """Allocation shape shared by EPLB transfer and memory profiling."""
    row_numel = tensor[0].numel()
    return (staged_rows, (row_numel + EPLB_FULL_STAGING_PARTS - 1) // EPLB_FULL_STAGING_PARTS)


def get_eplb_staging_nbytes(weights) -> int:
    """Owned staging bytes, using the transfer's allocation shapes including scales."""
    if not weights:
        return 0
    depth, staged_rows = get_eplb_staging_shape(weights)
    return depth * sum(
        math.prod(get_eplb_staging_tensor_shape(tensor, staged_rows)) * tensor.element_size()
        for _, tensor in extract_eplb_expert_tensors(weights[0])
    )


def get_eplb_sampling_peak_nbytes(weights) -> int:
    """Peak stack/index-select temporaries; persistent route counters are already allocated."""
    counters = []
    seen = set()
    for weight in weights:
        state = getattr(weight, "expert_parallel_state", None)
        counter = getattr(getattr(state, "eplb", None), "route_counter", None)
        if counter is not None and id(counter) not in seen:
            seen.add(id(counter))
            counters.append(counter)
    if not counters:
        return 0
    first = counters[0]
    if any(tuple(counter.shape) != tuple(first.shape) or counter.dtype != first.dtype for counter in counters):
        raise ValueError("EPLB route counters must have identical shape and dtype")
    align = lambda value: (value + 511) // 512 * 512
    stack_bytes = align(len(counters) * first.numel() * first.element_size())
    index_temp_bytes = align(first.shape[0] * 8)
    return 2 * stack_bytes + index_temp_bytes


def extract_eplb_expert_tensors(weight):
    result = []
    for pack_name in ("w13", "w2"):
        pack = getattr(weight, pack_name)
        for value_name in ("weight", "weight_scale", "weight_zero_point"):
            tensor = getattr(pack, value_name, None)
            if tensor is not None:
                assert tensor.ndim >= 1 and tensor.is_contiguous(), f"{pack_name}.{value_name} must be contiguous"
                result.append((f"{pack_name}.{value_name}", tensor))
    return result
