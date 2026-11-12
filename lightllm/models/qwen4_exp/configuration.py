from lightllm.common.build_utils import repair_config


def normalize_model_config(checkpoint, finetune_config=None):
    config = dict(checkpoint.get("text_config", checkpoint))
    for aliases in (
        ["num_attention_heads", "n_head"],
        ["hidden_size", "n_embd", "n_embed"],
        ["num_hidden_layers", "n_layer"],
    ):
        repair_config(config, same_names=aliases)
    rope = config.get("rope_parameters")
    if isinstance(rope, dict):
        for name in ("rope_theta", "partial_rotary_factor"):
            if name in rope:
                config.setdefault(name, rope[name])
        config.setdefault("rope_scaling", dict(rope))
    for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if name in checkpoint:
            config[name] = checkpoint[name]
    config.setdefault("norm_topk_prob", True)
    if finetune_config is not None:
        config["vocab_size"] = finetune_config.vocab_size
    return config, checkpoint.get("vision_config")
