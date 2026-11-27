"""DeepSeek V4.1 cache layouts and compressor state configuration."""

from lightllm.common.attention_state.bounded_att import WindowCompressorStateConfig
from lightllm.common.attention_state.full_att.compressed_att.compressed_att_state import CompressedAttentionConfig


def _validated_layer_ratios(model_config, model_logical_layer_ids, tp_rank, tp_world_size):
    assert model_logical_layer_ids and len(set(model_logical_layer_ids)) == len(model_logical_layer_ids)
    assert 0 <= tp_rank < tp_world_size
    head_dim = model_config["head_dim"]
    rope_dim = model_config["qk_rope_head_dim"]
    assert head_dim > rope_dim > 0 and (head_dim - rope_dim) % 64 == 0
    assert model_config["index_head_dim"] > 0 and model_config["sliding_window"] > 0

    ratios = model_config["compress_ratios"]
    layer_ratios = tuple((layer_id, ratios[layer_id]) for layer_id in model_logical_layer_ids)
    assert all(ratio in (0, 1, 2) for _, ratio in layer_ratios)
    return layer_ratios


def create_compressed_configs(
    model_config: dict[str, object],
    model_logical_layer_ids: tuple[int, ...],
    tp_rank: int = 0,
    tp_world_size: int = 1,
) -> tuple[CompressedAttentionConfig, ...]:
    layer_ratios = _validated_layer_ratios(model_config, model_logical_layer_ids, tp_rank, tp_world_size)
    kv_source_layers = set(model_config["kv_source_layer_ids"])
    groups_by_ratio = {1: [], 2: []}
    # Each consumer shares the preceding source layer's KV and indexer pages for its ratio.
    for layer_id, ratio in layer_ratios:
        if ratio == 0:
            continue
        groups = groups_by_ratio[ratio]
        if layer_id in kv_source_layers:
            groups.append([])
        groups[-1].append(layer_id)

    return tuple(
        CompressedAttentionConfig(
            logical_layer_groups=tuple(tuple(group) for group in groups),
            compress_ratio=ratio,
            kv_dim=model_config["head_dim"],
            rope_dim=model_config["qk_rope_head_dim"],
            block_token_size=64 * ratio,
            indexer_dim=model_config["index_head_dim"],
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        for ratio, groups in groups_by_ratio.items()
        if groups
    )


def create_compressor_configs(
    model_config: dict[str, object],
    model_logical_layer_ids: tuple[int, ...],
    tp_rank: int = 0,
    tp_world_size: int = 1,
) -> tuple[WindowCompressorStateConfig, ...]:
    # Only C2 sources retain a pending pair. C1 writes its latent directly.
    layer_ratios = _validated_layer_ratios(model_config, model_logical_layer_ids, tp_rank, tp_world_size)
    kv_source_layers = set(model_config["kv_source_layer_ids"])
    c2_source_layers = tuple(layer for layer, ratio in layer_ratios if ratio == 2 and layer in kv_source_layers)
    if not c2_source_layers:
        return ()
    config = WindowCompressorStateConfig(
        logical_layer_ids=c2_source_layers,
        window_size=2,
        compress_stride=2,
        candidate_dim=model_config["head_dim"],
        target="kv",
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
    )
    return (config,)
