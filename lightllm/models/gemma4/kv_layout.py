def get_kv_cache_layout(config):
    """Map Gemma's shared tail layers to physical owners and their last readers."""
    layer_types = config["layer_types"]
    cutoff = len(layer_types) - (config.get("num_kv_shared_layers") or 0)
    assert 0 < cutoff <= len(layer_types)
    layer_maps = {"sliding_attention": {}, "full_attention": {}}
    last_owner = {}
    last_reader = {}
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
        last_reader[owner] = layer_index
    return layer_maps, owners, last_reader
