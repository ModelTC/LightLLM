"""SenseNova Loop execution layout."""


def get_loop_layer_indices(settings: dict, num_layers: int) -> tuple[int, ...]:
    """Map execution depth to checkpoint layers; caches use execution depth."""
    start, end, count = settings["start"], settings["end"], settings["eval_count"]
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


def get_loop_layer_types(settings: dict, text_config: dict) -> tuple[str, ...]:
    num_layers = text_config["num_hidden_layers"]
    indices = get_loop_layer_indices(settings, num_layers)
    layer_types = text_config["layer_types"]
    return tuple(layer_types[i] for i in indices)
