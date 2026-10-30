"""Select explicit model fields in tests parametrized over Qwen and GLM."""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import torch
from lightllm.common.att_resources.types import TensorField
from lightllm.common.att_resources.full_att import (
    FullAttentionState, FullAttentionConfig, StaticFP8FullAttentionState,
    GQAFullAttentionState, GroupQuantizedFullAttentionState, MLAFullAttentionState, PackedMLAFullAttentionState,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from lightllm.common.att_resources.att_resource_manager import AttResourceManager
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


def rebuild_full_state(
    template: FullAttentionState,
    *,
    config: FullAttentionConfig | None = None,
    token_capacity: int | None = None,
    page_size: int | None = None,
    device: str | torch.device | None = None,
    scales: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
) -> FullAttentionState:
    """测试中按模板重建独立存储，显式传递运行时参数与静态标定值。"""
    config = template.config if config is None else config
    token_capacity = template.capacity if token_capacity is None else token_capacity
    page_size = template.page_size if page_size is None else page_size
    device = template.device if device is None else device
    if isinstance(template, StaticFP8FullAttentionState):
        prepared_full_state = type(template)(
            config, page_size=page_size, device=device,
            scales=template.scales if scales is None else scales,
            q_scales=template.q_scales if q_scales is None else q_scales,
        )
        prepared_full_state.initialize_storage(token_capacity)
        return prepared_full_state
    prepared_full_state = type(template)(
        config, page_size=page_size, device=device,
    )
    prepared_full_state.initialize_storage(token_capacity)
    return prepared_full_state


def owned_states(attributes: Mapping[str, object], state_class: type | tuple[type, ...]):
    """从显式成员选择指定类型的 State，按资源身份排序。"""
    return tuple(
        sorted(
            (state for state in attributes.values() if isinstance(state, state_class)),
            key=lambda state: state.resource_id,
        )
    )


def main_linear_state(resources: AttResourceManager):
    if hasattr(resources, "main_kda_state"):
        return resources.main_kda_state
    return resources.main_gdn_state


def prepare_model_states(model: TpPartBaseModel, *, logical_layer_ids: Sequence[int] | None = None):
    config = model.config.get("text_config", model.config)
    if logical_layer_ids is None:
        logical_layer_ids = range(config.get("num_hidden_layers", config.get("n_layer")))
    model.layers_infer = [SimpleNamespace(layer_num_=i) for i in logical_layer_ids]
    if not hasattr(model, "torch_memory_saver"):
        model.torch_memory_saver = SimpleNamespace(region=lambda **_: nullcontext())
    model.prepare_att_resources()


def create_full_attention_roles(
    template: FullAttentionState, main_layer_count: int,
) -> tuple[FullAttentionState, FullAttentionState | None]:
    """仅用于测试：按配置层号创建各自拥有独立存储的 main 和单个 draft。"""
    layers = template.config.logical_layer_groups
    boundary = sum(all(layer < main_layer_count for layer in group) for group in layers)
    assert boundary and all(layer < main_layer_count for group in layers[:boundary] for layer in group)
    assert all(layer >= main_layer_count for group in layers[boundary:] for layer in group), (
        "draft layers must follow main layers and cannot share main storage"
    )
    states = []
    for start, end in ((0, boundary), (boundary, len(layers))):
        if start == end:
            states.append(None)
            continue
        config = replace(template.config, logical_layer_groups=layers[start:end])
        if isinstance(template, StaticFP8FullAttentionState):
            state = rebuild_full_state(
                template, config=config, scales=template.scales[start:end].clone(),
                q_scales=None if template.q_scales is None else template.q_scales[start:end].clone(),
            )
        else:
            state = rebuild_full_state(template, config=config)
        states.append(state)
    return states[0], states[1]


def read_kv_storage(state: FullAttentionState) -> torch.Tensor | None:
    """测试只读比较用：按 K、V 顺序拼接；修改数据须直接操作 State 的真实 buffer。"""
    if isinstance(state, (MLAFullAttentionState, PackedMLAFullAttentionState)):
        return state.storage
    if state.k_storage is None:
        return None
    return torch.cat((state.k_storage, state.v_storage), dim=2)


def full_state_test_buffers(state: FullAttentionState) -> dict[str, torch.Tensor | None]:
    """仅供多方案参数化测试枚举实际存储，检查容量、独立分配及字节一致性。"""
    if isinstance(state, GroupQuantizedFullAttentionState):
        return {
            "k": state.k_storage, "v": state.v_storage,
            "k_scale": state.k_scales, "v_scale": state.v_scales,
        }
    if isinstance(state, (GQAFullAttentionState, StaticFP8FullAttentionState)):
        return {"k": state.k_storage, "v": state.v_storage}
    assert isinstance(state, (MLAFullAttentionState, PackedMLAFullAttentionState))
    return {"kv": state.storage}


def full_state_test_fields(
    state: FullAttentionState, buffers: dict[str, torch.Tensor] | None = None,
) -> dict[str, TensorField]:
    """测试中独立描述实际数据，用于校验不同存储方案的传输内容。"""
    buffers = full_state_test_buffers(state) if buffers is None else buffers
    axes = ("layer", "token", "head", "dim")
    if isinstance(state, PackedMLAFullAttentionState):
        storage = buffers["kv"]
        raw = storage.view(torch.uint8)
        if state.config.fp8_nope_dim:
            fields = {
                "kv_nope": TensorField(tensor=raw[..., :512], axes=axes),
                "kv_scale": TensorField(tensor=raw[..., 512:528].view(torch.float32), axes=("layer", "token", "head", "group")),
                "kv_rope": TensorField(tensor=raw[..., 528:656].view(torch.bfloat16), axes=axes),
            }
        else:
            fields = {"kv": TensorField(tensor=storage[..., :state.config.kv_dim], axes=axes)}
        start = state.indexer_offset
        end = start + state.config.indexer_dim
        fields["indexer_k"] = TensorField(tensor=raw[..., start:end], axes=axes)
        fields["indexer_scale"] = TensorField(tensor=raw[..., end:end + 4].view(torch.float32), axes=("layer", "token", "head", "scale"))
        return fields
    return {
        name: TensorField(tensor=tensor, axes=("layer", "token", "head", "group") if name.endswith("_scale") else axes)
        for name, tensor in buffers.items()
    }
