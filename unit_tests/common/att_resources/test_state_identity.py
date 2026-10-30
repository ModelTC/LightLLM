from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from unit_tests.common.att_resources.model_helpers import rebuild_full_state

from typing import TYPE_CHECKING
from dataclasses import replace

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import FullAttKVAllocator
from lightllm.common.att_resources.full_att import StaticFP8FullAttentionState
from unit_tests.common.att_resources.model_helpers import create_full_attention_roles
from unit_tests.common.att_resources.test_capacity import definition

if TYPE_CHECKING:
    from easydict import EasyDict


KINDS = ["gqa", "int4", "int8", "static_head", "static_tensor", "mla", "packed", "packed_fp8", "packed_glm"]


@pytest.mark.parametrize("kind", KINDS)
def test_full_state_prepares_layout_before_capacity_allocation(kind: str, pd_args: EasyDict):
    from inspect import signature

    state = definition(kind)
    parameters = signature(type(state)).parameters
    assert not {"token_capacity", "buffer", "scale_buffer"} & parameters.keys()
    assert state.capacity == 0 and read_kv_storage(state) is None
    token_bytes = state.get_token_bytes()
    cpu_regions = state.get_cpu_transfer_regions()
    pd_regions = state.get_pd_transfer_regions()
    assert read_kv_storage(state) is None
    original_config = state.config
    state.device = torch.device("cpu")
    state.initialize_storage(8)
    assert state.config is original_config
    assert state.capacity == 8 and read_kv_storage(state).shape[1] == 8 + state.page_size
    assert not read_kv_storage(state).is_meta
    assert state.get_token_bytes() == token_bytes
    assert state.get_cpu_transfer_regions() == cpu_regions
    assert state.get_pd_transfer_regions() == pd_regions
    actual_bytes = sum(buffer.numel() * buffer.element_size() for buffer in full_state_test_buffers(state).values())
    assert actual_bytes == token_bytes * (8 + state.page_size)


def test_numbered_drafts_keep_resource_id_order_and_unique_members():
    manager = AttResourceManager(request_capacity=1)
    for role in ("draft35", "draft10", "draft1", "main", "draft0", "draft2"):
        manager.set_att_state(role, definition("gqa"))
    with pytest.raises(AssertionError, match="already bound"):
        manager.set_att_state("draft1", definition("gqa"))
    manager.finish_state_creation()
    assert [state.resource_id.split("_", 1)[0] for state in manager.get_full_attention_states()] == [
        "draft0", "draft10", "draft1", "draft2", "draft35", "main",
    ]
    assert manager.get_full_attention_states("draft1") == (manager.draft1_gqa_bf16_state,)
    assert manager.get_full_attention_states("draft10") == (manager.draft10_gqa_bf16_state,)


def test_custom_state_type_controls_registered_resource_name():
    from unit_tests.common.att_resources.test_full_state import full_state
    from lightllm.common.att_resources.full_att import GQAFullAttentionState

    class CustomGQAState(GQAFullAttentionState):
        def get_state_type(self) -> str:
            return "gqa_custom_fp32"

    state = full_state(CustomGQAState)
    manager = AttResourceManager(request_capacity=1)
    manager.set_att_state("draft0", state)
    manager.finish_state_creation()
    assert manager.draft0_gqa_custom_fp32_state is state
    assert state.resource_id == "draft0_gqa_custom_fp32_state"
    assert manager.get_full_attention_states("draft0") == (state,)
    assert not hasattr(state, "state_type")


@pytest.mark.parametrize("kind", ["full_meta", "full", "bounded", "tail"])
@pytest.mark.parametrize("model_name", ["main", "draft1"])
def test_set_att_state_preserves_object_and_storage(kind: str, model_name: str):
    from unit_tests.common.att_resources.test_bounded_state import make_state
    from unit_tests.common.att_resources.test_tail_state import make_tail

    if kind == "full_meta":
        state = definition("gqa")
        storage = tuple(full_state_test_buffers(state).values())
    elif kind == "full":
        state = allocated_state("gqa")
        storage = tuple(full_state_test_buffers(state).values())
    elif kind == "bounded":
        state = make_state()
        storage = state.conv_state
        pool = state.big_pages_pool
    else:
        state = make_tail()
        storage = state.tail_buffer
    config = state.config
    manager = AttResourceManager(request_capacity=3)
    manager.set_att_state(model_name, state)
    name = f"{model_name}_{state.get_state_type()}_state"
    assert getattr(manager, name) is state
    assert state.resource_id == name
    assert state.config is config and not hasattr(config, "resource_id")
    if kind.startswith("full"):
        assert all(actual is original for actual, original in zip(full_state_test_buffers(state).values(), storage))
    elif kind == "bounded":
        assert state.conv_state is storage and state.big_pages_pool is pool
    else:
        assert state.tail_buffer is storage
    other_role = "draft0" if model_name == "main" else "main"
    with pytest.raises(AssertionError, match="multiple model roles"):
        manager.set_att_state(other_role, state)
    assert state.resource_id == name and getattr(manager, name) is state
    manager.finish_state_creation()
    with pytest.raises(AssertionError, match="creation is already finished"):
        manager.set_att_state(other_role, definition("gqa"))


def test_model_resource_role_numbers_the_36th_draft_and_rejects_the_37th():
    from lightllm.models.llama.model import LlamaTpPartModel

    draft = LlamaTpPartModel.__new__(LlamaTpPartModel)
    draft.is_mtp_draft_model = True
    draft.mtp_previous_draft_models = [object()] * 35
    assert draft.att_resource_role == "draft35"
    draft.mtp_previous_draft_models.append(object())
    with pytest.raises(AssertionError, match="at most 36"):
        _ = draft.att_resource_role


@pytest.mark.parametrize("role", ["draft", "draft36", "draft-1", "draft01"])
def test_set_att_state_rejects_invalid_draft_numbers(role: str):
    manager = AttResourceManager(request_capacity=1)
    with pytest.raises(AssertionError, match="unknown model role"):
        manager.set_att_state(role, definition("gqa"))


def allocated_state(kind: str, device: str = "cpu"):
    template = definition(kind)
    return rebuild_full_state(
        template=template,
        config=template.config,
        token_capacity=8,
        device=device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", KINDS)
def test_independent_role_states_copy_and_transfer_distinct_layers(kind: str, cpu_args: EasyDict):
    cpu_args.cpu_cache_token_page_size = 4
    physical = allocated_state(kind, device="cuda")
    main, draft = create_full_attention_roles(physical, 1)
    assert main.config.logical_layer_groups == ((0,),) and draft.config.logical_layer_groups == ((2,),)
    for role, state, layer in (("main", main, 0), ("draft0", draft, 1)):
        physical_buffers = full_state_test_buffers(physical)
        for name, buffer in full_state_test_buffers(state).items():
            original = physical_buffers[name]
            assert buffer.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
            buffer.fill_(layer + 3)
        if isinstance(state, StaticFP8FullAttentionState):
            for buffer, original in ((state.scales, physical.scales), (state.q_scales, physical.q_scales)):
                if buffer is not None:
                    assert buffer.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()

    source = AttResourceManager(request_capacity=2)
    source.set_att_state("main", main)
    source.set_att_state("draft0", draft)
    source.full_att_allocator = FullAttKVAllocator(8, 4, pin_memory=False)
    source.finish_state_creation()
    assert getattr(source, main.resource_id) is main
    assert getattr(source, draft.resource_id) is draft
    source.copy_full_att_tokens(torch.tensor([0, 1], dtype=torch.long, device="cuda"), torch.tensor([4, 5], dtype=torch.long, device="cuda"))

    target_physical = allocated_state(kind, device="cuda")
    target_main, target_draft = create_full_attention_roles(target_physical, 1)
    target = AttResourceManager(request_capacity=2)
    target.set_att_state("main", target_main)
    target.set_att_state("draft0", target_draft)
    target.full_att_allocator = FullAttKVAllocator(8, 4, pin_memory=False)
    target.finish_state_creation()
    page = torch.zeros((1, source.get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=True)
    source.write_req_to_cpu(
        token_indices=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
        cpu_page_buffers=page,
        cpu_page_indices=[0],
        layouts=source.get_cpu_transfer_layouts(),
    )
    target.read_req_from_cpu(
        cpu_page_buffers=page,
        cpu_page_indices=[0],
        token_indices=torch.tensor([6, 7], dtype=torch.long, device="cuda"),
        layouts=source.get_cpu_transfer_layouts(),
    )
    for src, dst in ((main, target_main), (draft, target_draft)):
        fields = {name: field.tensor for name, field in full_state_test_fields(dst, full_state_test_buffers(dst)).items()}
        for name, field in full_state_test_fields(src, full_state_test_buffers(src)).items():
            if "token" in field.axes:
                assert torch.equal(field.tensor[:, :2], fields[name][:, 6:8])
    assert set(source.get_cpu_transfer_layouts()) == {main.resource_id, draft.resource_id}


@pytest.mark.parametrize("overlap", [False, True])
def test_explicit_members_keep_role_and_scheme_distinct_even_with_overlapping_layers(overlap: bool):
    main = allocated_state("gqa")
    draft = allocated_state("gqa")
    manager = AttResourceManager(request_capacity=2)
    manager.set_att_state("main", main)
    manager.set_att_state("draft0", draft)
    manager.full_att_allocator = FullAttKVAllocator(main.capacity, main.page_size, pin_memory=False)
    template = allocated_state("int8")
    quant = rebuild_full_state(template, config=replace(template.config, logical_layer_groups=((4,), (6,))))
    manager.set_att_state("main", quant)
    if overlap:
        overlapping = allocated_state("static_head")
        manager.set_att_state("main", overlapping)
    manager.finish_state_creation()
    assert manager.main_gqa_bf16_state is main
    assert manager.draft0_gqa_bf16_state is draft
    assert manager.main_gqa_int8_group8_bf16_scale_state is quant
    assert manager.get_full_attention_states("draft0") == (draft,)
    expected = [main, quant]
    if overlap:
        assert manager.main_gqa_fp8_static_per_head_state is overlapping
        expected.append(overlapping)
    assert manager.get_full_attention_states("main") == tuple(sorted(expected, key=lambda state: state.resource_id))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_quantization_in_one_role_remains_distinct_in_copy_and_cpu_pages(cpu_args: EasyDict):
    cpu_args.cpu_cache_token_page_size = 4
    full = allocated_state("gqa", device="cuda")
    quant = allocated_state("int8", device="cuda")
    assert full.get_state_type() == "gqa_bf16"
    assert quant.get_state_type() == "gqa_int8_group8_bf16_scale"
    for state in (full, quant):
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            field.tensor.fill_(5)
            field.tensor[:, 4:6].zero_()
    manager = AttResourceManager(request_capacity=2)
    for state in (full, quant):
        manager.set_att_state("main", state)
    manager.finish_state_creation()
    manager.copy_full_att_tokens(torch.tensor([0, 1], dtype=torch.long, device="cuda"), torch.tensor([4, 5], dtype=torch.long, device="cuda"))
    for state in (full, quant):
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            assert (field.tensor[:, 4:6] == 5).all()
    layouts = manager.get_cpu_transfer_layouts()
    page = torch.zeros((1, max(layout.byte_end for layout in layouts.values())), dtype=torch.uint8, pin_memory=True)
    manager.write_req_to_cpu(
        token_indices=torch.tensor([0, 1], dtype=torch.long, device="cuda"),
        cpu_page_buffers=page,
        cpu_page_indices=[0],
        layouts=layouts,
    )
    for state in (full, quant):
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            field.tensor[:, 6:8].zero_()
    manager.read_req_from_cpu(
        cpu_page_buffers=page,
        cpu_page_indices=[0],
        token_indices=torch.tensor([6, 7], dtype=torch.long, device="cuda"),
        layouts=layouts,
    )
    for state in (full, quant):
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            assert (field.tensor[:, 6:8] == 5).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", KINDS)
def test_token_copy_captures_large_index_sets_and_preserves_complete_buffers(kind: str):
    template = definition(kind)
    state = rebuild_full_state(template, token_capacity=512, device="cuda")
    # 覆盖 int32/int64 和非连续索引；128 个元素足以触发旧 isin 实现的同步路径。
    source = torch.arange(256, dtype=torch.int32, device="cuda")[::2]
    target = torch.arange(
        start=256,
        end=512,
        dtype=torch.int64,
        device="cuda",
    )[::2]
    buffers = tuple(full_state_test_buffers(state).values())
    for buffer in buffers:
        buffer.view(torch.uint8).random_(0, 256)
    state.copy_full_att_tokens(source, target)  # 先完成 Triton 编译。
    torch.cuda.synchronize()

    expected = []
    for buffer in buffers:
        buffer.view(torch.uint8).random_(0, 256)
        snapshot = buffer.view(torch.uint8).clone()
        snapshot[:, target] = snapshot[:, source]
        expected.append(snapshot)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.copy_full_att_tokens(source, target)
    for buffer in buffers:
        buffer.index_fill_(1, target, 0)
    graph.replay()
    torch.cuda.synchronize()
    for buffer, snapshot in zip(buffers, expected):
        # 逐字节核对整块存储，包含量化 scale、packed padding 和未参与搬运的槽位。
        assert torch.equal(buffer.view(torch.uint8), snapshot)
