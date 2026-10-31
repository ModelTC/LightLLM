"""State 配置独立保留全局布局，改变 TP 后仍可恢复同一份缓存。"""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import rebuild_full_state

from typing import TYPE_CHECKING
import pickle
from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.common.attention_state.full_att import (
    FullAttentionConfig,
    GQAFullAttentionState,
    MLAFullAttentionState,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from easydict import EasyDict
    from lightllm.common.attention_state.bounded_att.base_state import BoundedAttentionState
    from lightllm.common.attention_state.full_att.base_state import FullAttentionState


def cpu_layout(state: FullAttentionState | BoundedAttentionState):
    manager = AttentionStateManager(request_capacity=1)
    manager.register_state("main", state)
    manager.finish_state_registration()
    return manager.get_cpu_page_layouts()[state.state_name]


@pytest.mark.parametrize(
    "state_cls,heads,replicated,source_ranges,target_ranges",
    [
        (GQAFullAttentionState, 4, False, [(0, 2), (2, 4)], [(i, i + 1) for i in range(4) for _ in range(2)]),
        (GQAFullAttentionState, 2, False, [(0, 1)] * 4 + [(1, 2)] * 4, [(0, 2)]),
        (GQAFullAttentionState, 2, True, [(0, 2)] * 4, [(0, 2)] * 2),
        (MLAFullAttentionState, 1, True, [(0, 1)] * 4, [(0, 1)]),
    ],
)
def test_serialized_full_config_restores_cpu_pages_with_different_tp(
    state_cls: type[FullAttentionState],
    heads: int,
    replicated: bool,
    source_ranges: Sequence[tuple[int, int]],
    target_ranges: Sequence[tuple[int, int]],
    cpu_args: EasyDict,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cpu_args.cpu_cache_token_page_size = 4
    config = FullAttentionConfig(
        logical_layer_groups=((0,),),
        global_head_num=heads,
        head_dim=4,
        dtype=torch.float32,
        replicated=replicated,
    )
    reference = state_cls(config, page_size=4, device="cuda")
    reference.allocate_storage(4)
    for storage_buffer in full_state_test_buffers(reference).values():
        storage_buffer.copy_(torch.arange(storage_buffer.numel()).reshape(storage_buffer.shape))
    fields = {name: field.tensor for name, field in full_state_test_fields(reference, full_state_test_buffers(reference)).items()}
    reference_layout = cpu_layout(reference)
    buffer = torch.zeros((1, reference_layout.byte_end), dtype=torch.uint8, pin_memory=True)

    # 接收方只需要配置和缓存页；config 中不保存本地 head 数或 manager 映射。
    restored_config = pickle.loads(pickle.dumps(config))
    assert "head_num" not in vars(restored_config)
    for rank, (start, end) in enumerate(source_ranges):
        source = state_cls(replace(config, tp_rank=rank, tp_world_size=len(source_ranges)), page_size=4, device="cuda")
        source.allocate_storage(4)
        layout = cpu_layout(source)
        assert layout.byte_end == reference_layout.byte_end
        for name, field in full_state_test_fields(source, full_state_test_buffers(source)).items():
            field.tensor.copy_(fields[name][:, :, start:end])
        source.write_to_cpu(
            token_indices=torch.arange(4, dtype=torch.long, device="cuda"),
            cpu_page_buffers=buffer,
            cpu_page_indices=[0],
            layout=layout,
        )

    for rank, (start, end) in enumerate(target_ranges):
        target = state_cls(replace(restored_config, tp_rank=rank, tp_world_size=len(target_ranges)), page_size=4, device="cuda")
        target.allocate_storage(4)
        layout = cpu_layout(target)
        assert layout.byte_end == reference_layout.byte_end
        target.read_from_cpu(
            cpu_page_buffers=buffer,
            cpu_page_indices=[0],
            token_indices=torch.arange(4, dtype=torch.long, device="cuda"),
            layout=layout,
        )
        for name, field in full_state_test_fields(target, full_state_test_buffers(target)).items():
            torch.testing.assert_close(field.tensor[:, :4], fields[name][:, :4, start:end])


def test_bounded_config_keeps_distinct_global_heads_when_local_heads_are_replicated(cpu_args: EasyDict):
    cpu_args.cpu_cache_token_page_size = 4
    config = ConvSSMStateConfig(
        logical_layer_ids=(0,),
        global_k_heads=2,
        global_v_heads=4,
        key_dim=2,
        value_dim=3,
        conv_width=2,
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        tp_rank=7,
        tp_world_size=8,
    )
    source = GDNState(config, request_capacity=1, device="meta")
    assert config.num_k_heads == config.num_v_heads == 1
    restored_config = replace(pickle.loads(pickle.dumps(config)), tp_rank=0, tp_world_size=1)
    target = GDNState(restored_config, request_capacity=1, device="meta")
    assert restored_config.num_k_heads == 2 and restored_config.num_v_heads == 4
    assert "num_k_heads" not in vars(config) and "num_v_heads" not in vars(config)
    assert target.conv_state.shape == (1, 1, 20, 2)
    assert target.ssm_state.shape == (1, 1, 4, 2, 3)
    source_layout = cpu_layout(source)
    target_layout = cpu_layout(target)
    assert source_layout.byte_end == target_layout.byte_end
    assert [(r.field, r.byte_start, r.byte_end, r.dtype, r.axes, r.shape) for r in source_layout.regions] == [
        (r.field, r.byte_start, r.byte_end, r.dtype, r.axes, r.shape) for r in target_layout.regions
    ]


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("page_tokens", [4, 8])
def test_drafts_keep_independent_global_layouts_with_equal_local_shapes(
    bounded: bool,
    page_tokens: int,
    pd_args: EasyDict,
):
    pd_args.pd_kv_page_size = page_tokens
    if bounded:
        config = ConvSSMStateConfig(
            logical_layer_ids=(0,),
            global_k_heads=2,
            global_v_heads=4,
            key_dim=2,
            value_dim=2,
            conv_width=2,
            conv_dtype=torch.float32,
            ssm_dtype=torch.float32,
            tp_world_size=8,
        )
        source = GDNState(config, request_capacity=1, device="meta")
        other_config = replace(config, logical_layer_ids=(1,), global_k_heads=4)
    else:
        config = FullAttentionConfig(
            logical_layer_groups=((0,),),
            global_head_num=2,
            head_dim=4,
            dtype=torch.float32,
            tp_world_size=8,
        )
        source = GQAFullAttentionState(config, page_size=4, device="meta")
        other_config = replace(config, logical_layer_groups=((1,),), global_head_num=4)
    if bounded:
        other = type(source)(
            config=other_config,
            request_capacity=source.request_capacity,
            mtp_step=source.mtp_step,
            device="meta",
        )
    else:
        other = rebuild_full_state(source, config=other_config)
    manager = AttentionStateManager(request_capacity=1)
    manager.register_state("draft0", source)
    manager.register_state("draft1", other)
    manager.finish_state_registration()
    query = manager.get_bounded_attention_states if bounded else manager.get_full_attention_states
    assert query("draft0") == (source,)
    assert query("draft1") == (other,)
    assert query() == (source, other)

    page_bytes = manager.get_pd_page_bytes()
    assert not hasattr(manager, "pd_page_bytes")
    assert not hasattr(manager, "pd_transfer_layouts")
    if not bounded:
        # 两个 draft 共 6 个全局 KV head：K/V、head_dim=4、FP32，无额外页头。
        assert page_bytes == page_tokens * 192
    assert manager.get_pd_page_bytes() == page_bytes
    getter = (
        manager.get_pd_runtime_page_layouts if bounded else manager.get_pd_token_page_layouts
    )
    layouts = getter()
    assert getter() is layouts
    assert set(layouts) == {source.state_name, other.state_name}
    assert all(layout.tokens_per_page == (None if bounded else page_tokens) for layout in layouts.values())
    unused_getter = (
        manager.get_pd_token_page_layouts if bounded else manager.get_pd_runtime_page_layouts
    )
    assert unused_getter() == {}
    assert manager.get_pd_page_bytes() == page_bytes
