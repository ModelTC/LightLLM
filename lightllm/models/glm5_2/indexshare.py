def owns_indexer_layer(layer_id: int, config: dict) -> bool:
    num_hidden_layers = config.get("num_hidden_layers")
    if num_hidden_layers is not None and layer_id >= num_hidden_layers:
        return True

    pattern = config.get("index_topk_pattern")
    if pattern is not None and 0 <= layer_id < len(pattern):
        return pattern[layer_id] != "S"

    freq = config.get("index_topk_freq", 1)
    offset = config.get("index_skip_topk_offset", 2)
    return max(layer_id - offset + 1, 0) % freq == 0
