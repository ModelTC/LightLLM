"""Shared helpers for model resource definitions used by startup and CPU caches."""

from __future__ import annotations

from typing import TYPE_CHECKING
from lightllm.common.build_utils import repair_config

if TYPE_CHECKING:
    from collections.abc import Sequence
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
    model_layer_ids: Sequence[int] | None = None,
):
    """Return default KV geometry or the model's specialized storage options."""
    build_options = getattr(model_class, "_full_attention_options", None)
    if build_options is not None:
        options = build_options(config, args)
    else:
        heads = config.get("num_key_value_heads", config["num_attention_heads"])
        dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        layers = config["num_hidden_layers"]
        options = dict(heads=heads, dim=dim, layer_ids=tuple(range(layers)))
    if model_layer_ids is not None:
        options["layer_ids"] = tuple(model_layer_ids[i] for i in options["layer_ids"] if i < len(model_layer_ids))
    return options


def create_full_attention_definition(
    model_class: type[TpPartBaseModel],
    model_config: dict[str, object],
    args: StartArgs | EasyDict,
    *,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    model_layer_ids: Sequence[int] | None = None,
    calibration_layer_start: int | None = None,
):
    """Build a meta State using the model's configuration and storage hooks."""
    from lightllm.utils.att_resource_config import build_full_attention_state

    config = get_resource_config(model_config, model_class=model_class)
    options = get_full_attention_options(config, args, model_class=model_class, model_layer_ids=model_layer_ids)
    if not options["layer_ids"]:
        return None
    state = build_full_attention_state(
        args,
        **options,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        calibration_layer_start=calibration_layer_start,
    )
    return state


def create_bounded_attention_definition(
    model_class: type[TpPartBaseModel],
    model_config: dict[str, object],
    args: StartArgs | EasyDict,
    *,
    model_layer_ids: Sequence[int],
    tp_rank: int = 0,
    tp_world_size: int = 1,
):
    """Use the model's optional bounded-attention definition, including inherited implementations."""
    create_definition = getattr(model_class, "create_bounded_attention_definition", None)
    if create_definition is None:
        return None
    return create_definition(
        model_config, args, model_layer_ids=model_layer_ids, tp_rank=tp_rank, tp_world_size=tp_world_size
    )


def get_model_calibration_layer_start(model: TpPartBaseModel) -> int:
    """Count preceding models' full-attention layers without initializing their States."""
    if not model.is_mtp_draft_model:
        return 0
    layer_start = 0
    for previous in (model.main_model, *model.mtp_previous_draft_models):
        if previous.is_mtp_draft_model and previous.args.mtp_mode in ("vanilla_no_att", "eagle_no_att"):
            continue
        config = get_resource_config(previous.config, model_class=type(previous))
        options = get_full_attention_options(
            config,
            previous.args,
            model_class=type(previous),
            model_layer_ids=tuple(layer.layer_num_ for layer in previous.layers_infer),
        )
        layer_start += len(options["layer_ids"])
    return layer_start


def init_model_full_attention(model: TpPartBaseModel) -> None:
    """按当前模型的配置和层号创建 full-attention meta State，直接挂到共享管理器。"""
    if model.is_mtp_draft_model and model.args.mtp_mode in ("vanilla_no_att", "eagle_no_att"):
        return
    definition = create_full_attention_definition(
        type(model),
        model.config,
        model.args,
        tp_rank=model.tp_rank_,
        tp_world_size=model.tp_world_size_,
        model_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
        calibration_layer_start=get_model_calibration_layer_start(model),
    )
    if definition is None:
        return
    model.att_resource_manager.set_att_state(model.att_resource_role, definition)
    # 层和 indexer 只记录规范成员名，推理时直接访问管理器中已分配的 State。
    name = definition.resource_id
    for layer in model.layers_infer:
        layer_id = layer.kv_share_target_layer_ if getattr(layer, "is_kv_shared_", False) else layer.layer_num_
        if layer_id in definition.config.layer_ids:
            layer.full_att_state_name = name
            if hasattr(layer, "indexer"):
                layer.indexer.full_att_state_name = name


def init_model_bounded_attention(model: TpPartBaseModel) -> None:
    """按当前模型的配置创建请求运行态，并直接挂到共享管理器；大小页稍后分配。"""
    definition = create_bounded_attention_definition(
        type(model), model.config, model.args,
        model_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
        tp_rank=model.tp_rank_, tp_world_size=model.tp_world_size_,
    )
    if definition is None:
        return
    from lightllm.common.att_resources.bounded_att import create_gdn_state, create_kda_state

    create_state = {"gdn": create_gdn_state, "kda": create_kda_state}[definition.get_state_type()]
    state = create_state(
        config=definition.config, request_capacity=model.max_req_num + 1, mtp_step=model.args.mtp_step, device="cuda"
    )
    model.att_resource_manager.set_att_state(model.att_resource_role, state)
