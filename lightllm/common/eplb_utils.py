"""Small, dependency-free EPLB helpers shared by transfer and model profiling."""


EPLB_MAX_STAGING_DEPTH = 8


def get_eplb_staging_shape(weights):
    """Shared allocation/profile policy: one full layer or up to eight replica layers."""
    if not weights:
        return 0, 0
    parallel = weights[0].expert_parallel_state
    state = parallel.eplb
    if getattr(state, "full_layout", False):
        return 1, parallel.num_primary_experts_per_rank + state.num_redundant_experts_per_rank
    return min(EPLB_MAX_STAGING_DEPTH, len(weights)), state.num_redundant_experts_per_rank


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
