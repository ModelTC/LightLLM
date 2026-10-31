"""Select explicit model fields in tests parametrized over Qwen and GLM."""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import torch
from lightllm.common.attention_state.types import TensorField
from lightllm.common.attention_state.full_att import (
    FullAttentionState, GQAAttentionConfig, MLAAttentionConfig,
    GroupQuantizedAttentionConfig, StaticFP8AttentionConfig, PackedMLAAttentionConfig,
    CompressedAttentionConfig, FP8E8M0PackedMLAConfig, StaticFP8FullAttentionState,
    GQAFullAttentionState, GroupQuantizedFullAttentionState, MLAFullAttentionState, PackedMLAFullAttentionState,
)


if TYPE_CHECKING:
    from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
    from lightllm.common.attention_state.allocators import FullAttKVAllocator
    from collections.abc import Mapping, Sequence
    from lightllm.common.attention_state.manager import AttentionStateManager
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


def bind_test_fp8_calibration(
    state: StaticFP8FullAttentionState,
    scales: torch.Tensor,
    q_scales: torch.Tensor | None = None,
) -> None:
    """将测试的本地期望值展开为标定文件格式，再通过生产绑定入口初始化。

    每个 TP 分片放置相同测试值；这样重建任意 rank 都能保留原测试的本地 scale。
    真实的非重复全局 head 切片由 test_model_calibration 单独覆盖。
    """
    config = state.config
    calibration = {"num_layers": len(config.logical_layer_groups), "quant_type": config.scale_mode}
    if config.scale_mode == "per_head":
        heads = config.head_num
        k, v = scales.split(heads, dim=-1)
        scales = torch.cat((k.repeat(1, config.tp_world_size), v.repeat(1, config.tp_world_size)), dim=-1)
        assert q_scales is not None
        q_scales = q_scales.repeat(1, config.tp_world_size)
        calibration["num_head"] = heads * config.tp_world_size
        calibration["q_calibration"] = {
            "num_head": q_scales.shape[1], "scales_shape": list(q_scales.shape), "scales": q_scales.tolist(),
        }
    calibration["scales_shape"] = list(scales.shape)
    calibration["scales"] = scales.tolist()
    state.bind_calibration(calibration=calibration, storage_layer_start=0)


def rebuild_full_state(
    template: FullAttentionState,
    config: (
        GQAAttentionConfig | MLAAttentionConfig | GroupQuantizedAttentionConfig
        | StaticFP8AttentionConfig | PackedMLAAttentionConfig
        | CompressedAttentionConfig | FP8E8M0PackedMLAConfig | None
    ) = None,
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
        )
        bind_test_fp8_calibration(
            state=prepared_full_state,
            scales=template.scales if scales is None else scales,
            q_scales=template.q_scales if q_scales is None else q_scales,
        )
        prepared_full_state.allocate_storage(token_capacity)
        return prepared_full_state
    prepared_full_state = type(template)(
        config, page_size=page_size, device=device,
    )
    prepared_full_state.allocate_storage(token_capacity)
    return prepared_full_state


def owned_states(attributes: Mapping[str, object], state_class: type | tuple[type, ...]):
    """从显式成员选择指定类型的 State，按资源身份排序。"""
    return tuple(
        sorted(
            (state for state in attributes.values() if isinstance(state, state_class)),
            key=lambda state: state.state_name,
        )
    )


def main_linear_state(resources: AttentionStateManager):
    (state,) = resources.get_bounded_attention_states("main")
    return state


def prepare_model_states(model: TpPartBaseModel, logical_layer_ids: Sequence[int] | None = None):
    config = model.config.get("text_config", model.config)
    if logical_layer_ids is None:
        logical_layer_ids = range(config.get("num_hidden_layers", config.get("n_layer")))
    model.layers_infer = [SimpleNamespace(layer_num_=i) for i in logical_layer_ids]
    if not hasattr(model, "torch_memory_saver"):
        model.torch_memory_saver = SimpleNamespace(region=lambda **_: nullcontext())
    model.prepare_attention_states()


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
                template=template,
                config=config,
                scales=template.scales[start:end].clone(),
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
        if state.config.dtype == torch.float8_e4m3fn:
            fields = {
                "kv_nope": TensorField(tensor=raw[..., :512], axes=axes),
                "kv_scale": TensorField(tensor=raw[..., 512:528].view(torch.float32), axes=("layer", "token", "head", "group")),
                "kv_rope": TensorField(tensor=raw[..., 528:656].view(torch.bfloat16), axes=axes),
            }
        else:
            fields = {"kv": TensorField(tensor=storage[..., :state.config.latent_dim], axes=axes)}
        start = state.config.indexer_offset
        end = start + state.config.indexer_dim
        fields["indexer_k"] = TensorField(tensor=raw[..., start:end], axes=axes)
        fields["indexer_scale"] = TensorField(tensor=raw[..., end:end + 4].view(torch.float32), axes=("layer", "token", "head", "scale"))
        return fields
    return {
        name: TensorField(tensor=tensor, axes=("layer", "token", "head", "group") if name.endswith("_scale") else axes)
        for name, tensor in buffers.items()
    }


def build_hybrid_resources(
    config: Qwen3NextResourceConfig,
    request_capacity: int,
    mtp_step: int,
    full_state: FullAttentionState,
    full_allocator: FullAttKVAllocator,
    big_page_capacity: int,
    small_page_capacity: int,
    draft_full_state: FullAttentionState | None = None,
    tp_rank: int = 0,
) -> AttentionStateManager:
    """测试按既定几何参数组装通用管理器，不依赖旧模型管理器的构造/绑定流程。"""
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.allocators import StatePageAllocator
    from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, create_gdn_state, create_kda_state
    from lightllm.common.attention_state.att_tail import KPoolStateConfig, create_kpool_state
    from lightllm.models.glm5_next.config import Glm5NextResourceConfig

    manager = AttentionStateManager(request_capacity=request_capacity)
    glm = isinstance(config, Glm5NextResourceConfig)
    create_state = create_kda_state if glm else create_gdn_state
    bounded_state = create_state(
        config=ConvSSMStateConfig(
            logical_layer_ids=tuple(
                layer for layer in range(config.all_layer_num) if (layer + 1) % config.full_attention_interval
            ),
            global_k_heads=config.global_linear_k_heads,
            global_v_heads=config.global_linear_v_heads,
            key_dim=config.head_linear_k_dim,
            value_dim=config.head_linear_v_dim,
            conv_width=config.conv_kernel_size - 1,
            conv_dtype=config.conv_state_dtype,
            ssm_dtype=config.ssm_state_dtype,
            tp_rank=tp_rank,
            tp_world_size=config.tp_world_size,
        ),
        request_capacity=request_capacity,
        mtp_step=mtp_step,
        device="cuda",
    )
    bounded_state.initialize_page_pools(
        big_page_capacity=big_page_capacity, small_page_capacity=small_page_capacity, pin_memory=True,
    )
    manager.register_state("main", bounded_state)
    for role, state in (("main", full_state), ("draft0", draft_full_state)):
        if state is None:
            continue
        manager.register_state(role, state)
        if glm:
            tail = create_kpool_state(
                config=KPoolStateConfig(
                    logical_layer_ids=tuple(layer for group in state.config.logical_layer_groups for layer in group),
                    pool_size=config.index_kpool,
                    head_dim=config.index_head_dim,
                    dtype=state.config.dtype,
                    tp_rank=tp_rank,
                    tp_world_size=config.tp_world_size,
                ),
                request_capacity=request_capacity,
                mtp_step=mtp_step,
                device="cuda",
            )
            manager.register_state(role, tail)
    manager.full_att_allocator = full_allocator
    manager.bounded_att_big_page_allocator = StatePageAllocator(big_page_capacity, reserved_slots=2)
    manager.bounded_att_small_page_allocator = StatePageAllocator(small_page_capacity) if small_page_capacity else None
    manager.bounded_att_cpu_load_big_page_slot = big_page_capacity - 2
    manager.bounded_att_cpu_offload_big_page_slot = big_page_capacity - 1
    manager.cpu_tail_restore = glm
    manager.finish_state_registration()
    return manager
