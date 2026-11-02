"""模型启动与 CPU cache 共用的配置展开及字段别名处理。"""

from lightllm.common.build_utils import repair_config


def normalize_model_config(model_config: dict[str, object]) -> dict[str, object]:
    """复制文本配置并补齐通用别名；模型专有覆盖由模型类在调用前完成。"""
    config = model_config.get("text_config", model_config).copy()
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])
    return config
