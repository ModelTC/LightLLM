"""Parse the target feature configuration of a draft checkpoint."""


def get_target_feature_sources(config: dict) -> tuple:
    """Return ordered target feature identifiers and an optional final-state marker.

    Integer sources are zero-based execution-layer outputs. The optional
    ``final_postnorm`` source follows all intermediate features. For backward
    compatibility, older checkpoints may declare the list in
    ``loop_teacher.feature_sources``.
    """
    config = config.get("text_config", config)
    layer_ids = config.get("target_layer_ids")
    if layer_ids is None:
        layer_ids = config.get("dflash_config", {}).get("target_layer_ids")
    if not layer_ids or any(type(layer) is not int or layer < 0 for layer in layer_ids):
        raise ValueError("target_layer_ids must contain non-negative execution-layer integers")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("target_layer_ids must not contain duplicates")
    ordered_layers = tuple(sorted(layer_ids))
    sources = config.get("target_feature_sources")
    legacy_sources = config.get("loop_teacher", {}).get("feature_sources")
    if sources is not None and legacy_sources is not None and list(sources) != list(legacy_sources):
        raise ValueError("target_feature_sources conflicts with loop_teacher.feature_sources")
    if sources is None:
        sources = legacy_sources
    if sources is None:
        return ordered_layers
    sources = tuple(sources)
    if any(type(source) is not int and source != "final_postnorm" for source in sources):
        raise ValueError("feature sources must be execution-layer integers or final_postnorm")
    if sources not in (ordered_layers, (*ordered_layers, "final_postnorm")):
        raise ValueError("feature sources must match sorted target_layer_ids, optionally followed by final_postnorm")
    return sources
