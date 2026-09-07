from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig


def get_kv_cache_layout(config):
    """Map Gemma's shared tail layers to their physical KV owners."""
    layer_types = config["layer_types"]
    cutoff = len(layer_types) - (config.get("num_kv_shared_layers") or 0)
    assert 0 < cutoff <= len(layer_types)
    layer_maps = {"sliding_attention": {}, "full_attention": {}}
    last_owner = {}
    owners = []
    for layer_index, layer_type in enumerate(layer_types):
        cache_map = layer_maps[layer_type]
        if layer_index < cutoff:
            last_owner[layer_type] = layer_index
            cache_map[layer_index] = len(set(cache_map.values()))
        else:
            cache_map[layer_index] = cache_map[last_owner[layer_type]]
        owner = last_owner[layer_type]
        owners.append(owner)
    return layer_maps, owners


def build_sliding_cache_config(config, tp_world_size, dtype):
    """Use the same physical owner layout in model and CPU-cache processes."""
    num_sliding_kv = config["num_key_value_heads"]
    num_full_kv = config.get("num_global_key_value_heads") or num_sliding_kv
    assert tp_world_size > 0
    assert num_sliding_kv % tp_world_size == 0, "sliding KV heads must be divisible by TP size"
    assert num_full_kv % tp_world_size == 0, "full KV heads must be divisible by TP size"
    layer_maps, _ = get_kv_cache_layout(config)
    return SlidingWindowCacheConfig(
        sliding_layer_to_cache_index=layer_maps["sliding_attention"],
        full_layer_to_cache_index=layer_maps["full_attention"],
        sliding_window=config["sliding_window"],
        sliding_head_num=num_sliding_kv // tp_world_size,
        sliding_head_dim=config["head_dim"],
        full_head_num=num_full_kv // tp_world_size,
        full_head_dim=config["global_head_dim"],
        dtype=dtype,
    )
