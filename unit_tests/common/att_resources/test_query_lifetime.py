import gc
import pickle
import weakref
from copy import copy

import pytest
import torch

from unit_tests.common.att_resources.manager_fixtures import MainDraftTestResourceManager


def test_queries_and_layouts_do_not_own_resource_lifetime(cpu_args):
    cpu_args.pd_kv_page_size = 8
    enabled = gc.isenabled()
    gc.disable()
    try:
        manager = MainDraftTestResourceManager()
        manager.get_full_attention_states()
        manager.get_bounded_attention_states()
        state = manager.main_gqa_fp32_state
        manager.get_state("main", "gqa", state.config.logical_layer_groups[0][0])
        manager.get_cpu_cache_operator().get_page_layouts()
        manager.get_pd_operator().get_page_bytes()
        refs = [weakref.ref(value) for value in (manager, state, state.k_storage, manager._pd_operator)]
        del state, manager
        assert all(ref() is None for ref in refs), "queries must not retain the manager or its native tensors"
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("copy_resources", [copy, lambda value: pickle.loads(pickle.dumps(value))])
def test_copies_rebuild_queries_and_operators_without_source_references(copy_resources):
    source = MainDraftTestResourceManager()
    original_states = source.get_full_attention_states()
    imported = copy_resources(source.get_pd_operator().create_ipc_copy())
    assert not imported.__dict__.get("_query_cache")
    assert imported.get_full_attention_states() is not original_states
    for operator in (imported.get_cpu_cache_operator(), imported.get_pd_operator(), imported.get_dp_shared_operator()):
        assert operator.owner is imported
    source_ref = weakref.ref(source)
    del source
    assert source_ref() is None
    assert all(state is getattr(imported, state.state_name) for state in imported.get_full_attention_states())


def test_invalidating_one_layout_does_not_drop_another_resources_cache():
    from lightllm.common.execution.storage.query_cache import CachedQueries, cached_query

    calls = []

    class Description(CachedQueries):
        @cached_query
        def describe(self, size):
            calls.append(size)
            return {"size": size}

    one, two = Description(), Description()
    first, second = one.describe(4), two.describe(4)
    assert first == second and first is not second
    assert one.describe(4) is first and two.describe(4) is second
    assert calls == [4, 4]
    one.clear_query_cache()
    assert one.describe(4) == first and two.describe(4) is second
    assert calls == [4, 4, 4]


def test_export_excludes_unpicklable_local_query_results():
    source = MainDraftTestResourceManager()
    source._query_cache = {"local operation": lambda: None}
    copied = copy(source)
    rebuilt = pickle.loads(pickle.dumps(source))
    assert "_query_cache" not in vars(copied)
    assert "_query_cache" not in vars(rebuilt)
    assert copied.get_pd_operator().owner is copied
    assert rebuilt.get_pd_operator().owner is rebuilt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bound_transfer_keeps_native_storage_without_manager(pd_args):
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.execution.transfer import Transfer
    from unit_tests.common.att_resources.model_helpers import rebuild_full_state
    from unit_tests.common.att_resources.test_capacity import definition

    state = rebuild_full_state(definition("gqa"), token_capacity=8, device="cuda")
    state.k_storage.fill_(3)
    state.v_storage.fill_(5)
    manager = AttentionStateManager(request_capacity=1)
    manager.register_state("main", state)
    manager.finish_state_registration()
    operator = manager.get_pd_operator()
    layouts = operator.get_pd_full_att_page_layouts()
    page = torch.zeros((1, operator.get_page_bytes()), dtype=torch.uint8, device="cuda")
    plan = operator.plan_write_page(
        page, 0, "kv", layouts, token_indices=torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    )
    manager_ref, state_ref, storage_ref = weakref.ref(manager), weakref.ref(state), weakref.ref(state.k_storage)
    del manager, state
    assert manager_ref() is None
    assert state_ref() is not None and storage_ref() is not None
    completed = Transfer().submit(plan)
    del plan
    completed.destination_ready.synchronize()
    target = rebuild_full_state(definition("gqa"), token_capacity=8, device="cuda")
    target.read_from_pd_page(
        page, torch.tensor([2, 4], dtype=torch.int64, device="cuda"), 0, next(iter(layouts.values()))
    )
    torch.cuda.synchronize()
    assert (target.k_storage[:, [2, 4]] == 3).all()
    assert (target.v_storage[:, [2, 4]] == 5).all()
    del completed
    assert state_ref() is None and storage_ref() is None
