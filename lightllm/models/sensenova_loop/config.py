"""SenseNova Loop execution layout."""


def get_loop_layer_indices(settings: dict, num_layers: int) -> tuple[int, ...]:
    """Map execution depth to checkpoint layers; caches use execution depth."""
    start, end = settings["start"], settings["end"]
    # Published checkpoints carry both the legacy scalar ``eval_count`` and
    # the newer ``counts`` list.  Accept either spelling and reject malformed
    # ranges before they produce an obscure index error during weight setup.
    count = settings.get("eval_count")
    if count is None:
        counts = settings.get("counts")
        if not isinstance(counts, (list, tuple)) or len(counts) != 1:
            raise ValueError("SenseNova Loop config must provide one loop count in eval_count or counts")
        count = counts[0]
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(count, int):
        raise TypeError("SenseNova Loop start, end and count must be integers")
    if not (0 <= start <= end <= num_layers) or count < 1:
        raise ValueError(f"Invalid SenseNova Loop range/count: start={start}, end={end}, count={count}")
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


def get_loop_layer_types(settings: dict, text_config: dict) -> tuple[str, ...]:
    num_layers = text_config["num_hidden_layers"]
    indices = get_loop_layer_indices(settings, num_layers)
    layer_types = text_config["layer_types"]
    return tuple(layer_types[i] for i in indices)
