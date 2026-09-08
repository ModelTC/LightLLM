"""Small, dependency-free EPLB helpers shared by transfer and model profiling."""


EPLB_MAX_STAGING_DEPTH = 8


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
