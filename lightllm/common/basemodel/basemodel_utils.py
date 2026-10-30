"""模型启动与 CPU cache 共用的配置展开及字段别名处理。"""

from __future__ import annotations

from typing import TYPE_CHECKING
from lightllm.common.build_utils import repair_config

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


def get_resource_config(
    model_config: dict[str, object], *, model_class: type[TpPartBaseModel] | None = None,
) -> dict[str, object]:
    """优先采用模型专有配置转换；默认展开 text_config 并补齐常见字段别名。"""
    build_config = getattr(model_class, "_resource_config", None)
    if build_config is not None:
        return build_config(model_config)
    config = model_config.get("text_config", model_config).copy()
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])
    return config
