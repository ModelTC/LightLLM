from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import bind_test_fp8_calibration

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage

from unit_tests.common.att_resources.model_helpers import rebuild_full_state

from typing import TYPE_CHECKING
from dataclasses import replace
from functools import partial
from math import prod

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import StatePageAllocator, FullAttKVAllocator
from lightllm.common.att_resources.bounded_att import ConvSSMStateConfig, GDNState, KDAState
from lightllm.common.att_resources.att_tail import KPoolAttentionTailState, KPoolStateConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.common.att_resources.full_att import (
    FullAttentionState,
    GQAFullAttentionState,
    FullAttentionConfig,
    GroupQuantizedAttentionConfig,
    GroupQuantizedFullAttentionState,
    MLAFullAttentionState,
    PackedMLAAttentionConfig,
    PackedMLAFullAttentionState,
    StaticFP8AttentionConfig,
    StaticFP8FullAttentionState,
)

if TYPE_CHECKING:
    from easydict import EasyDict


def definition(kind: str):
    config = FullAttentionConfig(((0,), (2,)), 2, 16, torch.bfloat16)
    if kind in ("int4", "int8"):
        return GroupQuantizedFullAttentionState(
            GroupQuantizedAttentionConfig(**vars(config), bits=4 if kind == "int4" else 8),
             page_size=4, device="meta",
        )
    if kind.startswith("static"):
        heads = 2 if kind == "static_head" else 1
        state = StaticFP8FullAttentionState(
            StaticFP8AttentionConfig(
                **vars(config), scale_mode="per_head" if heads == 2 else "per_tensor"
            ),
            page_size=4, device="meta",
        )
        bind_test_fp8_calibration(
            state=state, scales=torch.ones((2, heads * 2), dtype=torch.float32),
            q_scales=torch.ones((2, 2), dtype=torch.float32) if heads == 2 else None,
        )
        return state
    if kind.startswith("packed"):
        fp8 = kind == "packed_fp8"
        kv_dim = 512 if kind == "packed_glm" else 576
        config = replace(
            config, global_head_num=1, head_dim=800 if fp8 else kv_dim + 72, dtype=torch.uint8 if fp8 else config.dtype
        )
        return PackedMLAFullAttentionState(
            PackedMLAAttentionConfig(**vars(config), kv_dim=kv_dim, fp8_nope_dim=512 if fp8 else 0), page_size=4, device="meta"
        )
    if kind == "mla":
        return MLAFullAttentionState(replace(config, global_head_num=1, head_dim=576), page_size=4, device="meta")
    return GQAFullAttentionState(config, page_size=4, device="meta")


def allocated_bytes(state: FullAttentionState, *, token_only: bool = False):
    buffers = list(full_state_test_buffers(state).values())
    if not token_only and isinstance(state, StaticFP8FullAttentionState):
        buffers.append(state.scales)
        if state.q_scales is not None:
            buffers.append(state.q_scales)
    stores = {}
    for buffer in buffers:
        storage = buffer.untyped_storage()
        stores[(buffer.device, storage.data_ptr())] = storage.nbytes()
    return sum(stores.values())


@pytest.mark.parametrize(
    "kind", ["gqa", "int4", "int8", "static_head", "static_tensor", "mla", "packed", "packed_fp8", "packed_glm"]
)
def test_budget_matches_token_stores_including_hold_and_padding(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    pd_args: EasyDict,
):
    template = definition(kind)
    state = rebuild_full_state(template, config=template.config, token_capacity=20, device="cpu")
    actual = allocated_bytes(state, token_only=True)
    planning = AttResourceManager(request_capacity=1)
    planning.set_att_state("main", rebuild_full_state(template, config=template.config))
    planning.finish_state_creation()
    token_bytes = planning.get_full_att_token_bytes()
    pd_page_bytes = planning.get_pd_page_bytes()
    assert not hasattr(planning, "pd_page_bytes")
    assert not hasattr(planning, "pd_transfer_layouts")
    assert token_bytes * (20 + 4) == actual
    if kind.startswith("static"):
        assert allocated_bytes(state) > actual
    assert state.get_token_bytes() == token_bytes
    minimum = token_bytes * 4
    assert minimum > 0

    manager = AttResourceManager(request_capacity=1)
    manager.set_att_state("main", state)
    manager.full_att_allocator = FullAttKVAllocator(20, 4, pin_memory=False)
    manager.finish_state_creation()
    (state,) = manager.get_full_attention_states("main")
    assert manager.get_pd_page_bytes() == pd_page_bytes
    args = StartArgs(page_size=4, run_mode="prefill", pd_kv_page_size=8, pd_kv_page_num=3)
    reserve = manager.get_pd_page_bytes() * args.pd_kv_page_num
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr("lightllm.common.att_resources.att_resource_manager.get_current_device_id", lambda: 0)
    monkeypatch.setattr(
        "lightllm.common.att_resources.att_resource_manager.FullAttKVAllocator", partial(FullAttKVAllocator, pin_memory=False)
    )
    for budget, expected in ((actual, 20), (actual - 1, 16), (minimum, 0), (minimum - 1, None)):
        resources = AttResourceManager(request_capacity=1)
        resources.set_att_state("main", rebuild_full_state(template, config=template.config))
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (budget + reserve, budget + reserve))
        resources.finish_state_creation()
        if expected is None:
            with pytest.raises(MemoryError, match="HOLD"):
                resources.initialize_resources(args, token_capacity=None, mem_fraction=1.0, device="cpu")
        elif expected == 0:
            with pytest.raises(AssertionError, match="at least one complete page"):
                resources.initialize_resources(args, token_capacity=None, mem_fraction=1.0, device="cpu")
        else:
            resources.initialize_resources(args, token_capacity=None, mem_fraction=1.0, device="cpu")
            assert resources.full_att_allocator.size == expected
            assert resources.get_pd_page_bytes() == manager.get_pd_page_bytes()


def test_shared_allocator_accounts_for_distinct_state_stores():
    main, draft = definition("int4"), definition("static_head")
    manager = AttResourceManager(request_capacity=1)
    manager.set_att_state("main", main)
    manager.set_att_state("draft0", draft)
    manager.finish_state_creation()
    expected = sum(
        allocated_bytes(
            rebuild_full_state(state, config=state.config, token_capacity=12, page_size=4, device="cpu"),
            token_only=True,
        )
        for state in (main, draft)
    )
    assert manager.get_full_att_token_bytes() * (12 + 4) == expected


@pytest.mark.parametrize("state_cls", [KDAState, GDNState])
def test_pd_state_page_covers_one_snapshot_without_transfer_metadata(
    monkeypatch: pytest.MonkeyPatch,
    state_cls: type[KDAState] | type[GDNState],
    pd_args: EasyDict,
):
    config = ConvSSMStateConfig(
        logical_layer_ids=(0,), global_k_heads=2, global_v_heads=2 if state_cls is KDAState else 4, key_dim=3,
        value_dim=3 if state_cls is KDAState else 5, conv_width=1, conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )
    tail = KPoolAttentionTailState(
        KPoolStateConfig(logical_layer_ids=(1,), pool_size=4, head_dim=3, dtype=torch.bfloat16),
        request_capacity=3, mtp_step=3, device="cpu",
    )
    reference = state_cls(config, request_capacity=3, mtp_step=3, device="cpu")
    conv = reference.conv_state[:, 0, :, : config.conv_width]
    ssm = reference.ssm_state[:, 0]
    bounded_bytes = conv.numel() * conv.element_size() + ssm.numel() * ssm.element_size()
    tail_bytes = len(tail.config.logical_layer_ids) * tail.config.pool_size * 2 * tail.config.head_dim * tail.config.dtype.itemsize
    manager = AttResourceManager(request_capacity=3)
    manager.set_att_state("main", reference)
    manager.set_att_state("main", tail)
    manager.finish_state_creation()
    # PD 页只包含运行态数据，使用第二页验证 8 字节页跨度下的读写。
    assert manager.get_pd_page_bytes() == (bounded_bytes + 7) // 8 * 8 + (tail_bytes + 7) // 8 * 8
    reference.conv_state.fill_(2)
    reference.ssm_state.fill_(3)
    tail.tail_buffer.fill_(4)
    pages = torch.zeros((2, manager.get_pd_page_bytes()), dtype=torch.uint8)
    layouts = manager.get_pd_bounded_att_or_tail_transfer_layouts()
    assert {region.field for layout in layouts.values() for region in layout.regions} == {"conv", "ssm", "tail"}
    manager.write_to_pd_buffer(pages, pd_page_index=1, kind="att_state", layouts=layouts, req_idx=0, state_token_len=8)
    reference.conv_state.zero_()
    reference.ssm_state.zero_()
    tail.tail_buffer.zero_()
    manager.read_from_pd_buffer(pages, pd_page_index=1, kind="att_state", layouts=layouts, req_idx=0, state_token_len=8)
    assert (reference.conv_state[:, 0, :, : config.conv_width] == 2).all()
    assert (reference.ssm_state[:, 0] == 3).all()
    expected_tail = torch.zeros_like(tail.tail_buffer[:, 0])
    for position in range(4, 8):
        expected_tail[:, position % tail.tail_buffer.shape[2]].fill_(4)
    assert torch.equal(tail.tail_buffer[:, 0], expected_tail)
    assert not pages[0].any()
    for state in (reference, tail):
        monkeypatch.setattr(
            state, "get_pd_transfer_regions", lambda *args, **kwargs: pytest.fail("cached page size must not rebuild regions")
        )
    assert manager.get_pd_page_bytes() == pages.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pd_layout_reserves_aligned_storage_for_mixed_token_formats(
    monkeypatch: pytest.MonkeyPatch,
    pd_args: EasyDict,
):
    pd_args.pd_kv_page_size = 3
    manager = AttResourceManager(request_capacity=1)
    for state_cls, role, dim, dtype in (
        (MLAFullAttentionState, "draft0", 1, torch.bfloat16),
        (GQAFullAttentionState, "main", 3, torch.bfloat16),
        (GQAFullAttentionState, "main", 1, torch.float32),
    ):
        config = FullAttentionConfig(((0,),), 1, dim, dtype)
        manager.set_att_state(role, state_cls(config, page_size=1, device="meta"))
    manager.finish_state_creation()
    # 无页头；三个 State 的 3-token 区域分别由 6/36/24 字节对齐到 8/40/24。
    assert manager.get_pd_page_bytes() == 72
    monkeypatch.setattr(
        "lightllm.common.att_resources.att_resource_manager.FullAttKVAllocator", partial(FullAttKVAllocator, pin_memory=False)
    )
    manager.initialize_storage(4, 1, big_page_tokens=4, device="cuda")
    assert manager.get_pd_page_bytes() == 72
    states = manager.get_full_attention_states()
    for i, state in enumerate(states, 1):
        for storage_buffer in full_state_test_buffers(state).values():
            storage_buffer.fill_(i)
    pages = torch.zeros((2, manager.get_pd_page_bytes()), dtype=torch.uint8, device="cuda")
    page = pages[1]
    layouts = manager.get_pd_full_att_transfer_layouts()
    assert [min(region.byte_start for region in layouts[state.resource_id].regions) for state in states] == [0, 8, 48]
    manager.write_to_pd_buffer(page.unsqueeze(0), pd_page_index=0, kind="kv", layouts=layouts, token_indices=torch.tensor([0, 1, 2], dtype=torch.long, device="cuda"))
    assert (page[:6].view(torch.bfloat16) == 1).all()
    for state in states:
        for storage_buffer in full_state_test_buffers(state).values():
            storage_buffer.zero_()
    manager.read_from_pd_buffer(page.unsqueeze(0), pd_page_index=0, kind="kv", layouts=layouts, token_indices=torch.tensor([0, 1, 2], dtype=torch.long, device="cuda"))
    for i, state in enumerate(states, 1):
        assert (read_kv_storage(state)[:, :3] == i).all()
    assert not pages[0].any()


@pytest.mark.parametrize("layout_first", [False, True])
@pytest.mark.parametrize("transfer", ["cpu", "pd"])
def test_capacity_uses_quantized_field_padding_and_preserves_page_boundaries(
    monkeypatch: pytest.MonkeyPatch, pd_args: EasyDict, layout_first: bool, transfer: str
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    class AlignedFieldsState(GroupQuantizedFullAttentionState):
        def _get_transfer_regions(self, page_tokens: int):
            regions, offset = [], 0
            for region in super()._get_transfer_regions(page_tokens):
                offset = (offset + 7) // 8 * 8
                regions.append(replace(region, byte_start=offset, byte_end=offset + region.byte_end - region.byte_start))
                offset += prod(region.shape) * region.dtype.itemsize
            return tuple(regions)

    pd_args.pd_kv_page_size = pd_args.cpu_cache_token_page_size = 3
    config = GroupQuantizedAttentionConfig(
        logical_layer_groups=((0,),), global_head_num=1, head_dim=128,
        dtype=torch.float32, bits=8, group_size=128,
    )
    manager = AttResourceManager(request_capacity=1)
    prepared_full_state = AlignedFieldsState(config, page_size=1, device="cuda")
    prepared_full_state.initialize_storage(8)
    manager.set_att_state("main", prepared_full_state)
    prepared_full_state = AlignedFieldsState(config, page_size=1, device="cuda")
    prepared_full_state.initialize_storage(8)
    manager.set_att_state("draft0", prepared_full_state)
    manager.full_att_allocator = FullAttKVAllocator(8, 1, pin_memory=False)
    manager.finish_state_creation()
    # 每个 State 的裸数据为 792 字节；scale 间 padding 使实际末尾为 796，页段需 800 字节。
    for state in manager.get_full_attention_states():
        assert state.get_token_bytes() * 3 == 792
        monkeypatch.setattr(state, "get_token_bytes", lambda: pytest.fail("capacity must use field layouts"))
    get_layouts = manager.get_cpu_transfer_layouts if transfer == "cpu" else manager.get_pd_full_att_transfer_layouts
    get_page_bytes = manager.get_cpu_page_bytes if transfer == "cpu" else manager.get_pd_page_bytes
    if layout_first:
        with monkeypatch.context() as patch:
            patch.setattr(manager, f"get_{transfer}_page_bytes", lambda: pytest.fail("layout must not query page bytes"))
            layouts = get_layouts()
    assert get_page_bytes() == 1600
    layouts = get_layouts()
    assert get_layouts() is layouts
    pages = torch.full(
        (3, 1600), 91, dtype=torch.uint8, device="cpu" if transfer == "cpu" else "cuda", pin_memory=transfer == "cpu"
    )
    for index, state in enumerate(manager.get_full_attention_states()):
        layout = layouts[state.resource_id]
        assert (layout.byte_start, layout.byte_end) == (index * 800, (index + 1) * 800)
        regions = layout.regions
        assert [region.byte_start for region in regions] == [index * 800 + n for n in (0, 384, 768, 784)]
        for field_index, field in enumerate(full_state_test_fields(state, full_state_test_buffers(state)).values()):
            field.tensor.zero_()
            field.tensor[:, :3].fill_(index * 4 + field_index + 1)
    if transfer == "cpu":
        manager.write_req_to_cpu(torch.tensor([0, 1, 2], dtype=torch.long, device="cuda"), pages, [1], layouts=layouts)
        manager.read_req_from_cpu(pages, [1], torch.tensor([4, 5, 6], dtype=torch.long, device="cuda"), layouts=layouts)
    else:
        manager.write_to_pd_buffer(pages, pd_page_index=1, kind="kv", layouts=layouts, token_indices=torch.tensor([0, 1, 2], dtype=torch.long, device="cuda"))
        manager.read_from_pd_buffer(pages, pd_page_index=1, kind="kv", layouts=layouts, token_indices=torch.tensor([4, 5, 6], dtype=torch.long, device="cuda"))
    for state in manager.get_full_attention_states():
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            torch.testing.assert_close(field.tensor[:, 4:7], field.tensor[:, :3])
    assert (pages[[0, 2]] == 91).all()
    for offset in (0, 800):
        assert (pages[1, offset + 780:offset + 784] == 91).all()
        assert (pages[1, offset + 796:offset + 800] == 91).all()


@pytest.mark.parametrize("full_page_tokens", [None, 8, 10, 16])
def test_pd_runtime_capacity_includes_field_padding(
    monkeypatch: pytest.MonkeyPatch,
    pd_args: EasyDict,
    full_page_tokens: int,
):
    class PaddedGDNState(GDNState):
        def get_pd_transfer_regions(self):
            return tuple(
                replace(region, byte_start=region.byte_start + 64, byte_end=region.byte_end + 64) if region.field == "ssm" else region
                for region in super().get_pd_transfer_regions()
            )

    manager = AttResourceManager(request_capacity=1)
    if full_page_tokens is not None:
        pd_args.pd_kv_page_size = full_page_tokens
        config = FullAttentionConfig(((0,),), 1, 1, torch.float32)
        manager.set_att_state("main", GQAFullAttentionState(config, page_size=1, device="meta"))
    state = PaddedGDNState(
        ConvSSMStateConfig(
            logical_layer_ids=(0,), global_k_heads=1, global_v_heads=1, key_dim=1, value_dim=1, conv_width=1,
            conv_dtype=torch.bfloat16, ssm_dtype=torch.float32,
        ),
        request_capacity=1, device="cpu",
    )
    manager.set_att_state("main", state)
    manager.finish_state_creation()
    # Conv 为 6 字节，SSM 从偏移 72 开始，占 4 字节；完整运行态需要 80 字节。
    # 即使 KV 页不足，布局仍可独立查询；最终块大小查询选择两类占用的最大值。
    with monkeypatch.context() as patch:
        patch.setattr(manager, "get_pd_full_att_transfer_layouts", lambda: pytest.fail("runtime layout must be independent"))
        patch.setattr(manager, "get_pd_page_bytes", lambda: pytest.fail("layout must not query page bytes"))
        layouts = manager.get_pd_bounded_att_or_tail_transfer_layouts()
    layout = layouts[state.resource_id]
    assert layout.byte_end == 80 and layout.tokens_per_page is None
    assert layout.regions[-1].byte_start == 72
    warnings = []
    monkeypatch.setattr("lightllm.common.att_resources.att_resource_manager.logger.warning", warnings.append)
    expected = max(80, (full_page_tokens or 0) * 8)
    assert manager.get_pd_page_bytes() == expected
    assert manager.get_pd_bounded_att_or_tail_transfer_layouts() is layouts
    # 实际传输页可大于运行态布局，选中第二页后仍从该页起点搬运。
    pages = torch.full((3, expected), 91, dtype=torch.uint8)
    state.conv_state.fill_(3)
    state.ssm_state.fill_(5)
    manager.write_to_pd_buffer(pages, pd_page_index=1, kind="att_state", layouts=layouts, req_idx=0, state_token_len=8)
    state.conv_state.zero_()
    state.ssm_state.zero_()
    manager.read_from_pd_buffer(pages, pd_page_index=1, kind="att_state", layouts=layouts, req_idx=0, state_token_len=8)
    assert (state.conv_state == 3).all() and (state.ssm_state == 5).all()
    assert (pages[[0, 2]] == 91).all() and (pages[1, 80:] == 91).all()
    assert len(warnings) == int(full_page_tokens == 8)
    if warnings:
        assert "64 bytes" in warnings[0] and "using 80 bytes per transfer page" in warnings[0]
    # 相同 manager 的容量查询命中缓存，不重复打印 warning。
    assert manager.get_pd_page_bytes() == expected
    assert len(warnings) == int(full_page_tokens == 8)


@pytest.mark.parametrize("transfer", ["cpu", "pd"])
def test_page_bytes_refresh_with_storage_dependent_layout(
    monkeypatch: pytest.MonkeyPatch,
    pd_args: EasyDict,
    transfer: str,
):
    class PaddedState(GQAFullAttentionState):
        def _get_transfer_regions(self, page_tokens: int):
            return tuple(
                replace(
                    region,
                    byte_start=region.byte_start + self.page_size * 8,
                    byte_end=region.byte_end + self.page_size * 8,
                )
                for region in super()._get_transfer_regions(page_tokens)
            )

    pd_args.pd_kv_page_size = pd_args.cpu_cache_token_page_size = 3
    manager = AttResourceManager(request_capacity=1)
    config = FullAttentionConfig(((0,),), 1, 3, torch.bfloat16)
    manager.set_att_state("main", PaddedState(config, page_size=1, device="meta"))
    manager.finish_state_creation()
    get_page_bytes = manager.get_cpu_page_bytes if transfer == "cpu" else manager.get_pd_page_bytes
    assert get_page_bytes() == 48
    monkeypatch.setattr(
        "lightllm.common.att_resources.att_resource_manager.FullAttKVAllocator", partial(FullAttKVAllocator, pin_memory=False)
    )
    manager.initialize_storage(8, 4, big_page_tokens=8, device="cpu")
    assert get_page_bytes() == (80 if transfer == "cpu" else 72)
    layouts = manager.get_cpu_transfer_layouts() if transfer == "cpu" else manager.get_pd_full_att_transfer_layouts()
    assert layouts["main_gqa_bf16_state"].byte_end == 72


@pytest.mark.parametrize("with_tail", [False, True])
@pytest.mark.parametrize("extra_tokens", [-2, 0, 2])
def test_hybrid_pd_capacity_covers_all_runtime_states_before_storage_allocation(
    with_tail: bool,
    extra_tokens: int,
    pd_args: EasyDict,
):
    manager = AttResourceManager(request_capacity=1)
    config = FullAttentionConfig(((0,),), 1, 1, torch.float32)
    full_state = GQAFullAttentionState(config, page_size=1, device="meta")
    manager.set_att_state("main", full_state)
    for cls, role in ((KDAState, "main"), (GDNState, "draft0")):
        config = ConvSSMStateConfig(
            logical_layer_ids=(0,), global_k_heads=1, global_v_heads=1, key_dim=1, value_dim=1, conv_width=1,
            conv_dtype=torch.float32, ssm_dtype=torch.float32,
        )
        state = cls(config, request_capacity=1, device="meta")
        manager.set_att_state(role, state)
    if with_tail:
        state = KPoolAttentionTailState(
            KPoolStateConfig(logical_layer_ids=(0,), pool_size=4, head_dim=1, dtype=torch.float32),
            request_capacity=1, mtp_step=0, device="meta",
        )
        manager.set_att_state("main", state)
    # 两份运行态各占 16 字节数据，tail 另占 32 字节，无页头及长度、版本字段。
    runtime_page_bytes = 32 + 32 * with_tail
    page_tokens = runtime_page_bytes // 8 + extra_tokens
    pd_args.pd_kv_page_size = page_tokens
    manager.finish_state_creation()
    assert manager.get_pd_page_bytes() == max(runtime_page_bytes, runtime_page_bytes + extra_tokens * 8)
    assert manager.full_att_allocator is None
    assert read_kv_storage(manager.main_gqa_fp32_state) is None
    assert not hasattr(manager, "pd_transfer_layouts")


@pytest.mark.parametrize(
    "kind", ["gqa", "int4", "int8", "static_head", "static_tensor", "mla", "packed", "packed_fp8", "packed_glm"]
)
@pytest.mark.parametrize("tp_world_size", [1, 4])
@pytest.mark.parametrize("page_tokens", [1, 3])
def test_cpu_page_bytes_derive_from_cached_transfer_layouts(
    monkeypatch: pytest.MonkeyPatch, cpu_args: EasyDict, kind: str, tp_world_size: int, page_tokens: int
):
    cpu_args.cpu_cache_token_page_size = page_tokens
    template = definition(kind)
    config = replace(template.config, tp_world_size=tp_world_size, tp_rank=tp_world_size - 1)
    scales = torch.ones((len(config.logical_layer_groups), 2 * config.head_num)) if kind == "static_head" else None
    q_scales = torch.ones((len(config.logical_layer_groups), config.head_num)) if kind == "static_head" else None
    manager = AttResourceManager(request_capacity=3)
    for role in ("main", "draft0"):
        manager.set_att_state(role, rebuild_full_state(template, config=config, scales=scales, q_scales=q_scales))
    for state_cls, v_heads, value_dim in ((KDAState, 1, 1), (GDNState, 2, 3)):
        # 奇数长度的 BF16 Conv 与 FP32 SSM 之间需要补齐，MTP 候选区不应计入页大小。
        bounded_config = ConvSSMStateConfig(
            logical_layer_ids=(0,), global_k_heads=1, global_v_heads=v_heads, key_dim=1, value_dim=value_dim, conv_width=1,
            conv_dtype=torch.bfloat16, ssm_dtype=torch.float32, tp_rank=tp_world_size - 1,
            tp_world_size=tp_world_size,
        )
        bounded = state_cls(bounded_config, request_capacity=3, mtp_step=2, device="meta")
        manager.set_att_state("main", bounded)
    manager.finish_state_creation()
    # CPU 容量和布局都只依赖 State 的 CPU 字段描述，不读取 GPU 存储开销或 PD 布局。
    for state in (*manager.get_full_attention_states(), *manager.get_bounded_attention_states()):
        if isinstance(state, FullAttentionState):
            monkeypatch.setattr(state, "get_token_bytes", lambda: pytest.fail("CPU capacity must use field layouts"))
        monkeypatch.setattr(state, "get_pd_transfer_regions", lambda: pytest.fail("CPU layout must be independent"))
    layouts = manager.get_cpu_transfer_layouts()
    assert min(region.byte_start for layout in layouts.values() for region in layout.regions) == 0
    assert not {"state_token_len", "format_version"} & {
        region.field for layout in layouts.values() for region in layout.regions
    }
    expected_page_bytes = (max(layout.byte_end for layout in layouts.values()) + 15) // 16 * 16
    assert all(layout.tokens_per_page == page_tokens for layout in layouts.values())
    for state in (*manager.get_full_attention_states(), *manager.get_bounded_attention_states()):
        monkeypatch.setattr(state, "get_cpu_transfer_regions", lambda: pytest.fail("cached layout must be reused"))
    assert manager.get_cpu_page_bytes() == expected_page_bytes
    assert manager.get_cpu_transfer_layouts() is layouts


def test_cpu_bounded_capacity_includes_state_field_padding(cpu_args: EasyDict):
    class PaddedGDNState(GDNState):
        def get_cpu_transfer_regions(self):
            return tuple(
                replace(region, byte_start=region.byte_start + 64, byte_end=region.byte_end + 64)
                if region.field == "ssm" else region
                for region in super().get_cpu_transfer_regions()
            )

    manager = AttResourceManager(request_capacity=1)
    state = PaddedGDNState(
        ConvSSMStateConfig(
            logical_layer_ids=(0,), global_k_heads=1, global_v_heads=1, key_dim=1, value_dim=1, conv_width=1,
            conv_dtype=torch.bfloat16, ssm_dtype=torch.float32,
        ),
        request_capacity=1, device="meta",
    )
    manager.set_att_state("main", state)
    manager.finish_state_creation()
    # Conv 占 6 字节，SSM 位于 [72, 76)，State 与物理页终点均对齐到 80。
    assert manager.get_cpu_page_bytes() == 80
    layout = manager.get_cpu_transfer_layouts()[state.resource_id]
    assert (layout.byte_start, layout.byte_end) == (0, 80)
    assert layout.regions[-1].byte_end == 76


@pytest.mark.parametrize("kind", ["packed", "packed_fp8", "packed_glm"])
def test_cpu_packed_page_excludes_unused_gpu_row_padding(kind: str, cpu_args: EasyDict):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cpu_args.cpu_cache_token_page_size = 3
    states = []
    for encoding in ("gqa", kind):
        template = definition(encoding)
        state = rebuild_full_state(template, config=template.config, token_capacity=8, device="cuda")
        for storage_buffer in full_state_test_buffers(state).values():
            storage_buffer.view(torch.uint8).fill_(211)
        states.append(state)
    manager = AttResourceManager(request_capacity=1)
    manager.set_att_state("main", states[0])
    manager.set_att_state("draft0", states[1])
    manager.full_att_allocator = FullAttKVAllocator(8, 4, pin_memory=False)
    manager.finish_state_creation()
    layouts = manager.get_cpu_transfer_layouts()
    packed, gqa = manager.get_full_attention_states()
    packed_layout, gqa_layout = layouts[packed.resource_id], layouts[gqa.resource_id]
    data_end = max(r.byte_end for r in packed_layout.regions)
    gqa_start = gqa_layout.byte_start
    # 后一个 State 紧跟实际字段终点，仅保留对齐空间，不为 GPU packed 行 padding 预留位置。
    assert gqa_start == (data_end + 7) // 8 * 8
    assert gqa_start < read_kv_storage(packed)[:, :3].numel() * read_kv_storage(packed).element_size()
    for value, state in enumerate((packed, gqa), 3):
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            field.tensor[:, :3].fill_(value)
            field.tensor[:, 4:7].zero_()
    pages = torch.full((3, manager.get_cpu_page_bytes()), 165, dtype=torch.uint8, pin_memory=True)
    manager.write_req_to_cpu(torch.tensor([0, 1, 2], dtype=torch.long, device="cuda"), pages, [1], layouts=layouts)
    torch.cuda.synchronize()
    assert (pages[1, data_end:gqa_start] == 165).all()
    assert (pages[[0, 2]] == 165).all()
    manager.read_req_from_cpu(pages, [1], torch.tensor([4, 5, 6], dtype=torch.long, device="cuda"), layouts=layouts)
    for value, state in enumerate((packed, gqa), 3):
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            assert (field.tensor[:, 4:7] == value).all()


@pytest.mark.parametrize("page_tokens", [1, 9])
def test_cpu_bounded_snapshots_reserve_independent_aligned_regions(cpu_args: EasyDict, page_tokens: int):
    cpu_args.cpu_cache_token_page_size = page_tokens
    manager = AttResourceManager(request_capacity=1)
    manager.bounded_att_big_page_allocator = StatePageAllocator(2)
    manager.bounded_att_small_page_allocator = StatePageAllocator(1)
    for state_cls, role, value_dim in ((KDAState, "draft0", 1), (GDNState, "main", 3)):
        config = ConvSSMStateConfig(
            logical_layer_ids=(0,), global_k_heads=1, global_v_heads=1, key_dim=1, value_dim=value_dim, conv_width=1,
            conv_dtype=torch.bfloat16, ssm_dtype=torch.float32,
        )
        state = state_cls(config, request_capacity=1, mtp_step=2, device="cpu")
        state.initialize_page_pools(big_page_capacity=2, small_page_capacity=1, pin_memory=False)
        manager.set_att_state(role, state)
    manager.finish_state_creation()
    states = manager.get_bounded_attention_states()
    layouts = manager.get_cpu_transfer_layouts()
    # 两份快照分别含 10/22 字节数据，预留 16/24 字节；页 token 数不放大快照。
    assert [min(r.byte_start for r in layouts[state.resource_id].regions) for state in states] == [0, 16]
    assert manager.get_cpu_page_bytes() == 48
    assert [(layout.byte_start, layout.byte_end) for layout in layouts.values()] == [(0, 16), (16, 40)]
    for i, state in enumerate(states):
        state.conv_state.fill_(i + 3)
        state.ssm_state.fill_(i + 7)
    source_slots = manager.bounded_att_big_page_allocator.alloc(1)
    manager.save_bounded_att_pages([0], page_kind="big", slot_ids=source_slots)
    pages = torch.full((2, manager.get_cpu_page_bytes()), 87, dtype=torch.uint8)
    manager.write_req_to_cpu(
        torch.tensor([], dtype=torch.long), pages, [1], layouts=layouts, big_page_slot_ids=source_slots
    )
    assert (pages[0] == 87).all()
    assert (pages[1, 12:16] == 87).all()
    manager.reset_bounded_att_state([0])
    target_slots = manager.bounded_att_big_page_allocator.alloc(1)
    manager.read_req_from_cpu(
        pages, [1], torch.tensor([], dtype=torch.long), layouts=layouts, big_page_slot_ids=target_slots, req_idx=0
    )
    for i, state in enumerate(states):
        assert (state.conv_state[:, 0, :, :1] == i + 3).all()
        assert (state.ssm_state[:, 0] == i + 7).all()


@pytest.mark.parametrize("transfer", ["cpu", "pd_full_att", "pd_bounded_att_or_tail"])
def test_transfer_layouts_are_lazy_cached_and_refreshed_after_storage_allocation(
    monkeypatch: pytest.MonkeyPatch,
    transfer: str,
    pd_args: EasyDict,
):
    pd_args.pd_kv_page_size = 3
    pd_args.cpu_cache_token_page_size = 3
    manager = AttResourceManager(request_capacity=1)
    config = FullAttentionConfig(((0,),), 1, 3, torch.bfloat16)
    manager.set_att_state("main", GQAFullAttentionState(config, page_size=1, device="meta"))
    manager.set_att_state("main", GDNState(
        ConvSSMStateConfig(
            logical_layer_ids=(0,), global_k_heads=1, global_v_heads=1, key_dim=1, value_dim=1, conv_width=1,
            conv_dtype=torch.bfloat16, ssm_dtype=torch.bfloat16,
        ),
        request_capacity=1, device="cpu",
    ))
    manager.finish_state_creation()
    getter = getattr(manager, f"get_{transfer}_transfer_layouts")
    before = getter()
    assert getter() is before
    state = manager.main_gdn_conv_bf16_ssm_bf16_state if transfer == "pd_bounded_att_or_tail" else manager.main_gqa_bf16_state
    before_layout = before[state.resource_id]
    monkeypatch.setattr(
        "lightllm.common.att_resources.att_resource_manager.FullAttKVAllocator", partial(FullAttKVAllocator, pin_memory=False)
    )
    manager.initialize_storage(8, 4, big_page_tokens=8, device="cpu")
    after = getter()
    after_layout = after[state.resource_id]
    assert after is not before and getter() is after
    assert after_layout is not before_layout
    full_state = manager.main_gqa_bf16_state
    assert full_state.page_size == 4 and not read_kv_storage(full_state).is_meta
    page_bytes = manager.get_cpu_page_bytes() if transfer == "cpu" else manager.get_pd_page_bytes()
    if transfer == "pd_bounded_att_or_tail":
        assert after_layout.byte_end == 8 and page_bytes == 40
    elif transfer == "cpu":
        layout_end = max(layout.byte_end for layout in after.values())
        assert (layout_end + 15) // 16 * 16 == page_bytes
    else:
        assert after_layout.byte_end == page_bytes
    assert not {
        "cpu_page_tokens", "cpu_page_bytes", "cpu_transfer_layouts", "pd_page_bytes", "pd_transfer_layouts"
    } & manager.__dict__.keys()


@pytest.mark.parametrize("transfer", ["cpu", "pd"])
def test_manager_uses_state_regions_and_adds_each_resource_offset(transfer: str, pd_args: EasyDict):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    class ReversedCpuRegionsState(GQAFullAttentionState):
        def get_cpu_transfer_regions(self):
            regions = super().get_cpu_transfer_regions()
            # 本实现的 CPU 页先保存 V 再保存 K，PD 仍沿用 K/V 顺序。
            return tuple(
                replace(region, byte_start=other.byte_start, byte_end=other.byte_end) for region, other in zip(regions, reversed(regions))
            )

    pd_args.cpu_cache_token_page_size = pd_args.pd_kv_page_size = 4
    config = FullAttentionConfig(
        logical_layer_groups=((0,),), global_head_num=1, head_dim=2, dtype=torch.float32
    )
    manager = AttResourceManager(request_capacity=1)
    prepared_full_state = ReversedCpuRegionsState(config, page_size=4, device="cuda")
    prepared_full_state.initialize_storage(8)
    manager.set_att_state("main", prepared_full_state)
    prepared_full_state = ReversedCpuRegionsState(config, page_size=4, device="cuda")
    prepared_full_state.initialize_storage(8)
    manager.set_att_state("draft0", prepared_full_state)
    manager.full_att_allocator = FullAttKVAllocator(8, 4, pin_memory=False)
    manager.finish_state_creation()
    layouts = manager.get_cpu_transfer_layouts() if transfer == "cpu" else manager.get_pd_full_att_transfer_layouts()
    for index, state in enumerate(manager.get_full_attention_states()):
        for storage_buffer in full_state_test_buffers(state).values():
            storage_buffer.zero_()
        state.k_storage[:, :4].fill_(index + 3)
        state.v_storage[:, :4].fill_(index + 7)
        # 1 层、4 tokens，K/V 各 1 head、每 head 2 个 FP32 元素。
        state_bytes = 4 * 2 * 2 * torch.float32.itemsize
        k_offset = state_bytes // 2 if transfer == "cpu" else 0
        k_region, v_region = layouts[state.resource_id].regions
        assert k_region.field == "k" and v_region.field == "v"
        assert k_region.byte_start == index * state_bytes + k_offset
        assert v_region.byte_start == index * state_bytes + state_bytes // 2 - k_offset

    page_bytes = manager.get_cpu_page_bytes() if transfer == "cpu" else manager.get_pd_page_bytes()
    pages = torch.zeros(
        (1, page_bytes), dtype=torch.uint8, device="cpu" if transfer == "cpu" else "cuda", pin_memory=transfer == "cpu"
    )
    if transfer == "cpu":
        manager.write_req_to_cpu(torch.arange(4, dtype=torch.long, device="cuda"), pages, [0], layouts=layouts)
        manager.read_req_from_cpu(pages, [0], torch.arange(4, 8, dtype=torch.long, device="cuda"), layouts=layouts)
    else:
        manager.write_to_pd_buffer(pages, pd_page_index=0, kind="kv", layouts=layouts, token_indices=torch.arange(4, dtype=torch.long, device="cuda"))
        manager.read_from_pd_buffer(pages, pd_page_index=0, kind="kv", layouts=layouts, token_indices=torch.arange(4, 8, dtype=torch.long, device="cuda"))
    for state in manager.get_full_attention_states():
        torch.testing.assert_close(read_kv_storage(state)[:, 4:8], read_kv_storage(state)[:, :4])
        assert not read_kv_storage(state)[:, 8:].any()
