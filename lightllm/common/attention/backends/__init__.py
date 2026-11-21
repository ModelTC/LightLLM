def require_positive(layer_config: object, *fields: str) -> None:
    """Reject a layer config whose named integer fields are not positive."""
    invalid = {name: getattr(layer_config, name) for name in fields if getattr(layer_config, name) <= 0}
    if invalid:
        raise ValueError(f"{type(layer_config).__name__} fields must be positive, got {invalid}")


def keep_for_cuda_graph(store: dict, key: tuple, value):
    """Return storage at a fixed address for key, holding a copy of value.

    The first call for a key keeps a clone; later calls copy value into it in place, so tensors a
    captured CUDA graph reads see each replayed step's values.
    """
    kept = store.get(key)
    if kept is None:
        kept = store[key] = value.clone()
    else:
        kept.copy_(value)
    return kept
