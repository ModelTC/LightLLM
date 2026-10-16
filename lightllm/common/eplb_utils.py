"""Small, dependency-free EPLB helpers shared by transfer and model profiling."""

import math


EPLB_STEADY_SAMPLE_STEPS = 4
EPLB_MAX_STAGING_DEPTH = 8
EPLB_FULL_STAGING_PARTS = 2


def get_eplb_staging_shape(weights):
    """Shared allocation/profile policy: full uses one layer of row slices; replicas use up to eight layers."""
    if not weights:
        return 0, 0
    parallel = weights[0].expert_parallel_state
    state = parallel.eplb
    if getattr(state, "full_layout", False):
        return 1, parallel.num_primary_experts_per_rank + state.num_redundant_experts_per_rank
    return min(EPLB_MAX_STAGING_DEPTH, len(weights)), state.num_redundant_experts_per_rank


def get_eplb_staging_tensor_shape(tensor, staged_rows, *, full_layout):
    """Allocation shape shared by EPLB transfer and memory profiling."""
    if full_layout:
        row_numel = tensor[0].numel()
        return (staged_rows, (row_numel + EPLB_FULL_STAGING_PARTS - 1) // EPLB_FULL_STAGING_PARTS)
    return (staged_rows,) + tuple(tensor.shape[1:])


def get_eplb_staging_nbytes(weights) -> int:
    """Owned staging bytes, using the transfer's allocation shapes including scales."""
    if not weights:
        return 0
    depth, staged_rows = get_eplb_staging_shape(weights)
    full_layout = getattr(weights[0].expert_parallel_state.eplb, "full_layout", False)
    return depth * sum(
        math.prod(get_eplb_staging_tensor_shape(tensor, staged_rows, full_layout=full_layout)) * tensor.element_size()
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
        for value_name in ("weight", "weight_scale"):
            tensor = getattr(pack, value_name, None)
            if tensor is not None:
                assert tensor.ndim >= 1 and tensor.is_contiguous(), f"{pack_name}.{value_name} must be contiguous"
                result.append((f"{pack_name}.{value_name}", tensor))
    return result


def find_ep_moe_weights(model):
    # Deferred import keeps planner/transfer helpers independent of model imports.
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight

    weights = {}
    for layer in model.trans_layers_weight:
        for weight in getattr(layer, "__dict__", {}).values():
            if isinstance(weight, FusedMoeWeight) and weight.enable_ep_moe:
                weights[id(weight)] = weight
    return sorted(weights.values(), key=lambda weight: weight.layer_num_)
