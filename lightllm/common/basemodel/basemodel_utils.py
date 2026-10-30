"""Shared helpers for model resource definitions used by startup and CPU caches."""

from __future__ import annotations

from typing import TYPE_CHECKING
from lightllm.common.build_utils import repair_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from lightllm.common.att_resources.full_att import FullAttentionState
    from easydict import EasyDict
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.server.core.objs.start_args_type import StartArgs


def get_resource_config(model_config: dict[str, object], *, model_class: type[TpPartBaseModel] | None = None):
    """Normalize resource config, applying a model-specific override when present."""
    build_config = getattr(model_class, "_resource_config", None)
    if build_config is not None:
        return build_config(model_config)
    config = model_config.get("text_config", model_config).copy()
    repair_config(config, same_names=["num_attention_heads", "n_head"])
    repair_config(config, same_names=["hidden_size", "n_embd", "n_embed"])
    repair_config(config, same_names=["num_hidden_layers", "n_layer"])
    return config


def get_full_attention_options(
    config: dict[str, object],
    args: StartArgs | EasyDict,
    *,
    model_class: type[TpPartBaseModel] | None = None,
    model_logical_layer_ids: Sequence[int] | None = None,
):
    """Return default KV geometry or the model's specialized storage options."""
    build_options = getattr(model_class, "_full_attention_options", None)
    if build_options is not None:
        options = build_options(config, args)
    else:
        heads = config.get("num_key_value_heads", config["num_attention_heads"])
        dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        logical_layer_count = config["num_hidden_layers"]
        options = dict(
            heads=heads, dim=dim,
            logical_layer_groups=tuple((logical_layer_id,) for logical_layer_id in range(logical_layer_count)),
        )
    if model_logical_layer_ids is not None:
        # 模型选项使用本模型内的层号；转换为实际逻辑层号时保留共享分组及物理层顺序。
        options["logical_layer_groups"] = tuple(
            tuple(
                model_logical_layer_ids[local_logical_layer_id]
                for local_logical_layer_id in logical_layer_group
                if local_logical_layer_id < len(model_logical_layer_ids)
            )
            for logical_layer_group in options["logical_layer_groups"]
            if any(local_logical_layer_id < len(model_logical_layer_ids) for local_logical_layer_id in logical_layer_group)
        )
    return options


def create_full_attention_definition(
    model_class: type[TpPartBaseModel],
    model_config: dict[str, object],
    args: StartArgs | EasyDict,
    *,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    model_logical_layer_ids: Sequence[int] | None = None,
) -> FullAttentionState | None:
    """按模型配置创建算法 State；静态 FP8 标定由全部 State 就绪后的统一阶段绑定。"""
    from lightllm.common.att_resources.full_att import FullAttentionConfig, create_gqa_state, create_mla_state
    from lightllm.utils.torch_dtype_utils import get_torch_dtype

    if model_class.is_mtp_draft_model and args.mtp_mode in ("vanilla_no_att", "eagle_no_att"):
        return None
    config = get_resource_config(model_config, model_class=model_class)
    options = get_full_attention_options(config, args, model_class=model_class, model_logical_layer_ids=model_logical_layer_ids)
    if not options["logical_layer_groups"]:
        return None
    mla = options.get("mla", False)
    packed = options.get("packed", False)
    dtype = options.get("dtype")
    layout_config = FullAttentionConfig(
        logical_layer_groups=options["logical_layer_groups"],
        global_head_num=options["heads"],
        head_dim=options["dim"],
        dtype=get_torch_dtype(args.data_type) if dtype is None else dtype,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        replicated=options.get("replicated", False) or mla or packed,
    )
    if mla or packed:
        return create_mla_state(
            config=layout_config, kv_type=args.llm_kv_type, packed=packed, page_size=args.page_size, device="meta",
        )

    return create_gqa_state(
        config=layout_config, kv_type=args.llm_kv_type, page_size=args.page_size, device="meta",
    )
