"""Shared model configuration normalization."""

from lightllm.common.build_utils import repair_config


def normalize_model_config(model_config: dict[str, object]) -> dict[str, object]:
    """复制文本配置并补齐通用别名；模型专有覆盖由模型类在调用前完成。"""
    config = model_config.get("thinker_config", model_config)
    config = config.get("llm_config", config)
    config = config.get("text_config", config).copy()
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])
    return config
