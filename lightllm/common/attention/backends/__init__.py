def require_positive(layer_config: object, *fields: str) -> None:
    """Reject a layer config whose named integer fields are not positive."""
    invalid = {name: getattr(layer_config, name) for name in fields if getattr(layer_config, name) <= 0}
    if invalid:
        raise ValueError(f"{type(layer_config).__name__} fields must be positive, got {invalid}")
