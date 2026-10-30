"""Shared helpers for model resource definitions used by startup and CPU caches."""

from __future__ import annotations

from typing import TYPE_CHECKING
import hashlib
import json
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
    calibration_storage_layer_start: int = 0,
) -> FullAttentionState | None:
    """按模型配置直接创建算法 State；静态 FP8 使用当前模型的物理层标定区间。"""
    from lightllm.common.att_resources.full_att import FullAttentionConfig, create_gqa_state, create_mla_state
    from lightllm.utils.att_resource_config import load_static_fp8_calibration
    from lightllm.utils.torch_dtype_utils import get_torch_dtype

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

    scales = q_scales = None
    calibration_id = ""
    if args.llm_kv_type in ("fp8kv_sph", "fp8kv_spt"):
        calibration, scales, q_scales = load_static_fp8_calibration(
            args=args,
            per_head=args.llm_kv_type == "fp8kv_sph",
            storage_layer_count=len(layout_config.logical_layer_groups),
            heads=layout_config.head_num,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            storage_layer_start=calibration_storage_layer_start,
        )
        calibration_id = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    return create_gqa_state(
        config=layout_config, kv_type=args.llm_kv_type, page_size=args.page_size, device="meta",
        scales=scales, q_scales=q_scales, calibration_id=calibration_id,
    )


def create_bounded_attention_definition(
    model_class: type[TpPartBaseModel],
    model_config: dict[str, object],
    args: StartArgs | EasyDict,
    *,
    model_logical_layer_ids: Sequence[int],
    tp_rank: int = 0,
    tp_world_size: int = 1,
):
    """Use the model's optional bounded-attention definition, including inherited implementations."""
    create_definition = getattr(model_class, "create_bounded_attention_definition", None)
    if create_definition is None:
        return None
    return create_definition(
        model_config, args, model_logical_layer_ids=model_logical_layer_ids, tp_rank=tp_rank, tp_world_size=tp_world_size
    )


def get_model_calibration_storage_layer_start(model: TpPartBaseModel) -> int:
    """按前序模型的物理层数累计标定表行偏移，不创建 State，也不使用逻辑层号作为行号。"""
    if not model.is_mtp_draft_model:
        return 0
    storage_layer_start = 0
    for previous in (model.main_model, *model.mtp_previous_draft_models):
        if previous.is_mtp_draft_model and previous.args.mtp_mode in ("vanilla_no_att", "eagle_no_att"):
            continue
        config = get_resource_config(previous.config, model_class=type(previous))
        options = get_full_attention_options(
            config,
            previous.args,
            model_class=type(previous),
            model_logical_layer_ids=tuple(layer.layer_num_ for layer in previous.layers_infer),
        )
        storage_layer_start += len(options["logical_layer_groups"])
    return storage_layer_start


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
        model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
        calibration_storage_layer_start=get_model_calibration_storage_layer_start(model),
    )
    if definition is None:
        return
    model.att_resource_manager.set_att_state(model.att_resource_role, definition)


def init_model_bounded_attention(model: TpPartBaseModel) -> None:
    """按当前模型的配置创建请求运行态，并直接挂到共享管理器；大小页稍后分配。"""
    definition = create_bounded_attention_definition(
        type(model), model.config, model.args,
        model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
        tp_rank=model.tp_rank_, tp_world_size=model.tp_world_size_,
    )
    if definition is None:
        return
    from lightllm.common.att_resources.bounded_att import GDNState, KDAState, create_gdn_state, create_kda_state

    create_state = {GDNState: create_gdn_state, KDAState: create_kda_state}[type(definition)]
    state = create_state(
        config=definition.config, request_capacity=model.max_req_num + 1, mtp_step=model.args.mtp_step, device="cuda"
    )
    model.att_resource_manager.set_att_state(model.att_resource_role, state)
