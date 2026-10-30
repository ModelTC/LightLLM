"""State 对象创建完成后即可查询，容量分配保持对象身份和查询缓存不变。"""

from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import read_kv_storage


import gc
import pickle
import weakref
from functools import partial

import pytest
from easydict import EasyDict

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import FullAttKVAllocator
from lightllm.common.att_resources.full_att import FullAttentionState
from lightllm.server.core.objs.start_args_type import StartArgs
from unit_tests.common.att_resources.manager_fixtures import MainDraftTestResourceManager
from unit_tests.common.att_resources.test_capacity import definition


def test_created_state_queries_keep_the_same_objects_through_storage_initialization(monkeypatch: pytest.MonkeyPatch):
    args = StartArgs(page_size=4)
    definitions = []
    manager = AttResourceManager(request_capacity=1)
    for role, kind in (("main", "gqa"), ("draft0", "int8")):
        manager.set_att_state(role, definition(kind))
        definitions.append(next(
            state for state in vars(manager).values()
            if isinstance(state, FullAttentionState) and state.resource_id.startswith(role + "_")
        ))
        for query in (
            manager.get_full_attention_states, manager.get_bounded_attention_states, manager.get_attention_tail_states,
        ):
            with pytest.raises(AssertionError, match="State creation"):
                query()
        assert manager._get_states.cache_info().currsize == 0

    cached = []
    initialize_storage = AttResourceManager.initialize_storage

    def check_created_before_allocation(manager: AttResourceManager, *args: object, **kwargs: object):
        states = manager.get_full_attention_states()
        assert all(read_kv_storage(state) is None for state in states)
        assert manager.full_att_allocator is None
        cached.append(states)
        initialize_storage(manager, *args, **kwargs)

    monkeypatch.setattr(AttResourceManager, "initialize_storage", check_created_before_allocation)
    monkeypatch.setattr(
        "lightllm.common.att_resources.att_resource_manager.FullAttKVAllocator", partial(FullAttKVAllocator, pin_memory=False)
    )
    manager.finish_state_creation()
    manager.initialize_resources(args, token_capacity=8, mem_fraction=1.0, device="cpu")
    assert manager.get_full_attention_states() is cached[0]
    for role, meta in zip(("main", "draft0"), definitions):
        (state,) = manager.get_full_attention_states(role)
        assert state is meta and read_kv_storage(state).device.type == "cpu"
        assert manager.get_full_attention_states(role) == (state,)


def test_cached_role_queries_reuse_instance_references_and_keep_stable_order():
    manager = MainDraftTestResourceManager()
    queries = (manager.get_full_attention_states, manager.get_bounded_attention_states, manager.get_attention_tail_states)
    results = {(query.__name__, role): query(role) for query in queries for role in (None, "main", "draft0")}
    assert manager._get_states.cache_info().currsize == 9
    assert manager.get_full_attention_states() == (
        manager.draft0_mla_fp32_state, manager.main_gqa_fp32_state,
    )
    assert manager.get_bounded_attention_states("main") == (manager.main_kda_conv_fp32_ssm_fp32_state,)
    assert manager.get_bounded_attention_states("draft0") == (manager.draft0_gdn_conv_fp32_ssm_fp32_state,)
    assert manager.get_attention_tail_states("draft0") == ()

    hits = manager._get_states.cache_info().hits
    for query in queries:
        for role in (None, "main", "draft0"):
            assert query(role) is results[(query.__name__, role)]
        with pytest.raises(AssertionError, match="unknown model role"):
            query("target")
    assert manager._get_states.cache_info().hits == hits + 9
    manager.main_kda_conv_fp32_ssm_fp32_state.conv_state.fill_(7)
    assert (manager.get_bounded_attention_states("main")[0].conv_state == 7).all()


def test_finished_state_creation_allows_unallocated_storage():
    manager = AttResourceManager(request_capacity=1)
    manager.set_att_state("main", definition("gqa"))
    manager.full_att_allocator = FullAttKVAllocator(0, 4, pin_memory=False)
    manager.finish_state_creation()
    assert read_kv_storage(manager.get_full_attention_states()[0]) is None


def test_ipc_copy_preserves_owner_storage_and_shares_runtime_tensors():
    source = MainDraftTestResourceManager()
    source_states = source.get_bounded_attention_states()
    owner_pools = [(state.big_pages_pool, state.small_pages_pool) for state in source_states]

    exported = source.create_pd_ipc_copy()

    assert type(exported) is type(source) and exported is not source
    for name in ("full_att_allocator", "bounded_att_big_page_allocator", "bounded_att_small_page_allocator"):
        assert getattr(source, name) is not None
        assert getattr(exported, name) is None
    for state, (big_pool, small_pool) in zip(source_states, owner_pools):
        copied = getattr(exported, state.resource_id)
        assert copied is not state
        assert copied.conv_state is state.conv_state
        assert copied.ssm_state is state.ssm_state
        assert copied.mtp_state_index is state.mtp_state_index
        assert copied.big_pages_pool is None and copied.small_pages_pool is None
        assert copied.request_capacity == state.request_capacity
        assert copied.mtp_step == state.mtp_step
        assert state.big_pages_pool is big_pool and state.small_pages_pool is small_pool
        assert big_pool.size == source.bounded_att_big_page_allocator.size
        assert small_pool.size == source.bounded_att_small_page_allocator.size
    assert exported.get_bounded_attention_states() == tuple(
        getattr(exported, state.resource_id) for state in source_states
    )
    for state in (*source.get_full_attention_states(), *source.get_attention_tail_states()):
        assert getattr(exported, state.resource_id) is state



@pytest.mark.parametrize("share_pools", [False, True])
def test_pd_and_dp_export_have_distinct_pool_ownership(cpu_args: EasyDict, share_pools: bool) -> None:
    cpu_args.enable_dp_prompt_cache_fetch = share_pools
    source = MainDraftTestResourceManager()
    pd_copy = source.create_pd_ipc_copy()
    for state in source.get_bounded_attention_states():
        exported = getattr(pd_copy, state.resource_id)
        assert exported.big_pages_pool is None and exported.small_pages_pool is None
        assert state.big_pages_pool is not None and state.small_pages_pool is not None
    if not share_pools:
        with pytest.raises(AssertionError, match="shared CPU page pools"):
            source.create_dp_ipc_copy()
        return
    dp_copy = source.create_dp_ipc_copy()
    for name in ("full_att_allocator", "bounded_att_big_page_allocator", "bounded_att_small_page_allocator"):
        assert getattr(dp_copy, name) is None
        assert getattr(source, name) is not None
    for state in source.get_bounded_attention_states():
        exported = getattr(dp_copy, state.resource_id)
        assert exported.big_pages_pool is state.big_pages_pool
        assert exported.small_pages_pool is state.small_pages_pool
        assert exported.conv_state is state.conv_state


def test_ipc_rebuilds_queries_from_imported_members_without_allocators():
    source = MainDraftTestResourceManager()
    source.get_full_attention_states()
    source.get_bounded_attention_states()
    source.get_attention_tail_states()
    imported = pickle.loads(pickle.dumps(source.create_pd_ipc_copy()))
    assert imported._states_created
    assert imported.full_att_allocator is None
    assert imported.bounded_att_big_page_allocator is None
    assert imported.bounded_att_small_page_allocator is None
    misses = source._get_states.cache_info().misses
    for query in (
        imported.get_full_attention_states, imported.get_bounded_attention_states, imported.get_attention_tail_states,
    ):
        for state in query():
            name = state.resource_id
            assert state is getattr(imported, name) and state is not getattr(source, name)
    assert imported._get_states.cache_info().misses == misses + 3


def test_clearing_query_cache_releases_manager_and_states():
    manager = MainDraftTestResourceManager()
    manager.get_full_attention_states()
    manager.get_bounded_attention_states()
    manager.get_attention_tail_states()
    manager_ref = weakref.ref(manager)
    state_ref = weakref.ref(manager.main_kda_conv_fp32_ssm_fp32_state)
    del manager
    gc.collect()
    assert manager_ref() is not None and state_ref() is not None
    AttResourceManager._get_states.cache_clear()
    gc.collect()
    assert manager_ref() is None and state_ref() is None
