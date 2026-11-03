from lightllm.common.attention_state.storage import allocate_runtime_storage, initialize_state_page_pools
import pytest
import torch

from lightllm.common.attention_state.memory_plan import MemoryDemand, plan_memory, merge_memory_demands
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState, KDAState
from lightllm.common.attention_state.bounded_att import WindowCompressorState, WindowCompressorStateConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replanning_reuses_resident_calibration_without_reserving_it_twice(cpu_args):
    from lightllm.common.attention_state import AttentionStateManager
    from unit_tests.common.att_resources.test_capacity import definition

    state = definition("static_head")
    manager = AttentionStateManager(1)
    manager.register_state("main", state)
    manager.finish_state_registration()
    cpu_args.page_size = 4
    device = torch.device("cuda", torch.cuda.current_device())
    constants = state.get_fixed_storage_bytes(device)
    storage_bytes = state.get_token_bytes() * (32 + cpu_args.page_size)
    limits = {"cpu": 1 << 20, "cuda": storage_bytes + constants}
    manager.initialize_storage(cpu_args, None, 1.0, device=device, memory_limits=limits)
    assert manager.memory_plan.capacity == 32 and constants > 0
    pointers = state.scales.data_ptr(), state.q_scales.data_ptr()
    manager.release_unpublished_storage()
    assert state.get_fixed_storage_bytes(device) == 0
    manager.initialize_storage(cpu_args, None, 1.0, device=device, memory_limits={**limits, "cuda": storage_bytes})
    assert manager.memory_plan.capacity == 32
    assert (state.scales.data_ptr(), state.q_scales.data_ptr()) == pointers


def test_checkpoint_boundary_can_limit_capacity_before_device_memory():
    demands = (MemoryDemand("main_kv", "cuda", 64, 16), MemoryDemand("checkpoints", "cpu", 500, 100, 16))
    plan = plan_memory(demands, {"cuda": 3264, "cpu": 1200}, 4)
    assert plan.capacity == 112
    assert plan.usage("cpu") == 1200
    assert plan.usage("cuda") < 3264
    with pytest.raises(MemoryError, match="cpu budget"):
        plan_memory(demands, {"cuda": 3264, "cpu": 1200}, 4, 116)


def test_capacity_accounts_for_main_draft_hold_staging_and_reserved_workspace():
    demands = (
        MemoryDemand("main", "cuda", 16, 4),
        MemoryDemand("draft", "cuda", 32, 8),
        MemoryDemand("staging", "cuda", 80),
        MemoryDemand("workspace_lane0", "cuda", 64),
        MemoryDemand("workspace_lane1", "cuda", 64),
    )
    plan = plan_memory(demands, {"cuda": 400}, 4)
    assert plan.capacity == 12
    assert plan.usage("cuda") == 400
    with pytest.raises(MemoryError, match="HOLD"):
        plan_memory(demands, {"cuda": 256}, 4)
    with pytest.raises(ValueError, match="counted once"):
        plan_memory((*demands, demands[0]), {"cuda": 400}, 4)


def test_provider_workspace_is_budgeted_once_per_concurrent_lane(monkeypatch):
    from lightllm.common.basemodel.attention.base_att import BaseAttBackend

    monkeypatch.setattr("lightllm.common.basemodel.attention.base_att.get_current_device_id", lambda: 0)
    monkeypatch.setattr(BaseAttBackend, "_workspace_buffers", {})
    backend = object.__new__(BaseAttBackend)
    first = backend.workspace_memory_demands("shared", 64, 2)
    second = backend.workspace_memory_demands("shared", 64, 2)
    demands = merge_memory_demands((MemoryDemand("kv", "cuda", bytes_per_unit=8),), first, second)
    plan = plan_memory(demands, {"cuda": 256}, 4)
    assert plan.capacity == 16
    assert len(demands) == 3

    BaseAttBackend._workspace_buffers[(0, "shared", 64, torch.int8, 0)] = object()
    remaining = backend.workspace_memory_demands("shared", 64, 2)
    assert remaining == first[1:]
    with pytest.raises(ValueError, match="inconsistent requirements"):
        merge_memory_demands(first, (MemoryDemand(first[0].name, "cuda", 65),))


@pytest.mark.parametrize("family", ["gdn", "kda", "compressor", "no_overlap"])
def test_checkpoint_requirement_matches_native_pool_allocations(family):
    if family in ("gdn", "kda"):
        config = ConvSSMStateConfig(
            logical_layer_ids=(0, 2),
            global_k_heads=2,
            global_v_heads=2,
            key_dim=8,
            value_dim=8,
            conv_width=3,
            conv_dtype=torch.bfloat16,
            ssm_dtype=torch.float32,
        )
        state = allocate_runtime_storage(
            (GDNState if family == "gdn" else KDAState)(config, request_capacity=2, mtp_step=3), "cpu"
        )
    else:
        config = WindowCompressorStateConfig(
            logical_layer_ids=(0, 2),
            window_size=8 if family == "compressor" else 4,
            compress_stride=4,
            candidate_dim=8,
            target="indexer",
            dtype=torch.bfloat16,
        )
        state = allocate_runtime_storage(WindowCompressorState(config, request_capacity=2, mtp_step=3), "cpu")
    initialize_state_page_pools(state, 3, 2, device="cpu", pin_memory=False)
    actual = sum(
        t.numel() * t.element_size()
        for pool in (state.big_pages_pool, state.small_pages_pool)
        if pool is not None
        for t in vars(pool).values()
        if isinstance(t, torch.Tensor)
    )
    assert state.get_checkpoint_bytes() * 5 == actual


@pytest.mark.parametrize("family", ["gdn", "kda", "window", "compressor", "no_overlap"])
def test_bounded_runtime_is_allocated_only_after_budget_acceptance(cpu_args, family):
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.bounded_att import (
        FP8E8M0SlidingWindowConfig,
        FP8E8M0SlidingWindowState,
    )

    if family in ("gdn", "kda"):
        cls = GDNState if family == "gdn" else KDAState
        config = ConvSSMStateConfig((0, 2), 2, 2, 8, 8, 3, torch.bfloat16, torch.float32)
    elif family == "window":
        cls = FP8E8M0SlidingWindowState
        config = FP8E8M0SlidingWindowConfig((0, 2), 16, 128, 64)
    else:
        cls = WindowCompressorState
        config = WindowCompressorStateConfig((0, 2), 8 if family == "compressor" else 4, 4, 8)
    state = allocate_runtime_storage(cls(config, request_capacity=3, mtp_step=2), "meta")
    manager = AttentionStateManager(3)
    manager.register_state("main", state)
    manager.finish_state_registration()
    cpu_args.page_size = 4
    cpu_args.linear_att_hash_page_size = 8
    cpu_args.linear_att_page_block_num = 1
    cpu_args.linear_att_cache_size = 2
    runtime_bytes = state.get_unallocated_runtime_bytes()
    demands = manager.get_storage_memory_demands(cpu_args, "cpu")
    assert next(d.fixed_bytes for d in demands if d.name.endswith("_runtime")) == runtime_bytes
    with pytest.raises(MemoryError, match="cpu budget"):
        manager.initialize_storage(cpu_args, 8, 1.0, device="cpu", memory_limits={"cpu": runtime_bytes - 1})
    assert manager.full_att_allocator is None and state.big_pages_pool is None
    assert state.get_unallocated_runtime_bytes() == runtime_bytes

    preview = manager.plan_storage(cpu_args, 8, 1.0, device="cpu", memory_limits={"cpu": 1 << 20})
    assert preview.capacity == 8 and manager.full_att_allocator is None and state.big_pages_pool is None
    assert state.get_unallocated_runtime_bytes() == runtime_bytes
    manager.initialize_storage(cpu_args, 8, 1.0, device="cpu", memory_limits={"cpu": 1 << 20})
    assert manager.memory_plan == preview
    expected = allocate_runtime_storage(cls(config, request_capacity=3, mtp_step=2), "cpu")
    assert not manager.get_full_attention_states()
    for name in state.runtime_storage_fields:
        actual, reference = getattr(state, name), getattr(expected, name)
        if actual is not None:
            assert actual.device.type == "cpu" and torch.equal(actual, reference)
    assert state.get_unallocated_runtime_bytes() == 0
    assert not any(d.name.endswith("_runtime") for d in manager.get_storage_memory_demands(cpu_args, "cpu"))
    pointers = [
        getattr(state, name).data_ptr() for name in state.runtime_storage_fields if getattr(state, name) is not None
    ]
    assert allocate_runtime_storage(state, "cpu") is state
    assert pointers == [
        getattr(state, name).data_ptr() for name in state.runtime_storage_fields if getattr(state, name) is not None
    ]
