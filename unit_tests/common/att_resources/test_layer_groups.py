"""逻辑层共享物理 KV：容量、视图、标定索引和外部页传输保持一致。"""

from dataclasses import is_dataclass, replace
from typing import TYPE_CHECKING

import pytest
import torch
from torch.utils._pytree import tree_leaves

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.full_att import FullAttentionConfig
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.basemodel_utils import create_full_attention_definition
from lightllm.models.gemma4.model import Gemma4TpPartModel
from unit_tests.common.att_resources.model_helpers import (
    full_state_test_buffers,
    full_state_test_fields,
    rebuild_full_state,
)
from unit_tests.common.att_resources.test_capacity import definition, allocated_bytes

if TYPE_CHECKING:
    from easydict import EasyDict


SCHEMES = ["gqa", "int4", "int8", "static_head", "static_tensor", "mla", "packed", "packed_fp8", "packed_glm"]


@pytest.mark.parametrize("kind", SCHEMES)
def test_shared_layers_use_two_physical_stores(kind: str, pd_args: "EasyDict") -> None:
    template = definition(kind)
    config = replace(template.config, logical_layer_groups=((0, 4), (8, 11)))
    state = rebuild_full_state(template, config=config, token_capacity=8, device="cpu")
    assert all(state.has_logical_layer(logical_layer_id) for logical_layer_id in (0, 4, 8, 11))
    assert not state.has_logical_layer(1)
    assert all(buffer.shape[0] == 2 for buffer in full_state_test_buffers(state).values())
    assert state.get_token_bytes() == template.get_token_bytes()
    assert allocated_bytes(state, token_only=True) == state.get_token_bytes() * (8 + state.page_size)

    for getter in (state.get_cpu_transfer_regions, state.get_pd_transfer_regions):
        assert all(region.shape[region.axes.index("layer")] == 2 for region in getter())

    # 同组返回相同地址，不同组返回独立数据；包括量化 scale 和 packed indexer 的计算视图。
    views = []
    for layer_id in (0, 4, 8, 11):
        view = state.get_layer_view(logical_layer_id=layer_id)
        if is_dataclass(view):
            view = tuple(vars(view).values())
        views.append([(tensor.data_ptr(), tensor.shape, tensor.stride()) for tensor in tree_leaves(view)])
    assert views[0] == views[1]
    assert views[2] == views[3]
    assert views[0] != views[2]

    if kind in ("gqa", "static_head", "static_tensor"):
        k, v = state.get_layer_view(logical_layer_id=11)
        assert BaseAttBackend._find_storage_layer_index(None, k, v, state) == 1


@pytest.mark.parametrize("groups", [((),), ((0,), (0,)), ((0, 0),), (0, 4)])
def test_invalid_layer_groups_are_rejected(groups: tuple) -> None:
    with pytest.raises(AssertionError):
        FullAttentionConfig(logical_layer_groups=groups, global_head_num=1, head_dim=128, dtype=torch.bfloat16)


def test_model_local_shared_groups_preserve_global_logical_layer_ids(pd_args: "EasyDict") -> None:
    pd_args.data_type = "bfloat16"
    config = dict(
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, hidden_size=512,
        head_dim=128, global_head_dim=128, num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 3,
    )
    state = create_full_attention_definition(
        model_class=Gemma4TpPartModel, model_config=config, args=pd_args,
        model_logical_layer_ids=(10, 11, 12, 13, 14, 15),
    )
    # 共享消费者 14/15 分别使用最后一个同类型生产层 12/13，保持四份物理层存储。
    assert state.config.logical_layer_groups == ((10,), (11,), (12, 14), (13, 15))
    assert state.get_token_bytes() == 4 * 2 * 2 * 128 * state.config.dtype.itemsize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", SCHEMES)
@pytest.mark.parametrize("transfer", ["cpu", "pd"])
def test_shared_layers_transfer_each_physical_layer_once(
    kind: str, transfer: str, pd_args: "EasyDict",
) -> None:
    pd_args.cpu_cache_token_page_size = pd_args.pd_kv_page_size = 4
    template = definition(kind)
    source = rebuild_full_state(
        template, config=replace(template.config, logical_layer_groups=((0, 4), (8,))), token_capacity=8, device="cuda",
    )
    target = rebuild_full_state(
        template, config=replace(template.config, logical_layer_groups=((20, 24), (28,))), token_capacity=8, device="cuda",
    )
    source_buffers = full_state_test_buffers(source)
    target_buffers = full_state_test_buffers(target)
    for buffer in source_buffers.values():
        buffer.view(torch.uint8).random_(0, 256)
    for buffer in target_buffers.values():
        buffer.zero_()
    managers = []
    for state in (source, target):
        manager = AttResourceManager(request_capacity=1)
        manager.set_att_state(model_name="main", state=state)
        manager.finish_state_creation()
        managers.append(manager)
    src_ids = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    dst_ids = torch.tensor([1, 4, 7], dtype=torch.int32, device="cuda")
    if transfer == "cpu":
        pages = torch.zeros((2, managers[0].get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=True)
        managers[0].write_req_to_cpu(
            token_indices=src_ids, cpu_page_buffers=pages, cpu_page_indices=[1],
            layouts=managers[0].get_cpu_transfer_layouts(),
        )
        managers[1].read_req_from_cpu(
            cpu_page_buffers=pages, cpu_page_indices=[1], token_indices=dst_ids,
            layouts=managers[1].get_cpu_transfer_layouts(),
        )
    else:
        pages = torch.zeros((2, managers[0].get_pd_page_bytes()), dtype=torch.uint8, device="cuda")
        managers[0].write_to_pd_buffer(
            pd_page_buffers=pages, pd_page_index=1, kind="kv", token_indices=src_ids,
            layouts=managers[0].get_pd_full_att_transfer_layouts(),
        )
        managers[1].read_from_pd_buffer(
            pd_page_buffers=pages, pd_page_index=1, kind="kv", token_indices=dst_ids,
            layouts=managers[1].get_pd_full_att_transfer_layouts(),
        )
    torch.cuda.synchronize()
    source_fields = full_state_test_fields(source, source_buffers)
    target_fields = full_state_test_fields(target, target_buffers)
    for name, field in source_fields.items():
        expected = field.tensor[:, src_ids].contiguous().view(torch.uint8)
        actual = target_fields[name].tensor[:, dst_ids].contiguous().view(torch.uint8)
        assert torch.equal(actual, expected)
