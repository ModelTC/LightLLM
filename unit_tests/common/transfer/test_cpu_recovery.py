from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.manager import AttentionStateManager
from lightllm.common.attention_state.allocators import StatePageAllocator
from lightllm.common.transfer import CopyOperation, CopyPlan, Transfer, TransferFailure
from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as implementation
from unit_tests.common.transfer.helpers import Runtime


@pytest.mark.parametrize("binding_failure", [False, True])
def test_failed_cpu_load_retires_new_resources_without_publishing_prefix(monkeypatch, binding_failure):
    runtime = Runtime()
    resources = AttentionStateManager(1)
    released = []
    resources.full_att_allocator = SimpleNamespace(free=lambda tokens: released.extend(tokens.tolist()))
    resources.bounded_att_big_page_allocator = StatePageAllocator(1)
    req = SimpleNamespace(req_idx=0, cur_kv_len=0, hold_kv_len=0, hybrid_len_to_big_page_id={})
    table = torch.full((1, 4), -1, dtype=torch.int32)
    copied = []

    def fail():
        raise ValueError("bad state format")

    def plan(**kwargs):
        if binding_failure:
            fail()
        return CopyPlan(
            "unpack",
            (CopyOperation("kv", None, lambda: copied.append("kv")), CopyOperation("state", None, fail)),
        )

    resources._cpu_cache_operator = SimpleNamespace(plan_read_req=plan, get_page_layouts=lambda: {})

    def alloc(req, count):
        req.hold_kv_len += count
        table[0, :] = torch.arange(count)
        return table[0].clone()

    module = implementation.MultiLevelKvCacheModule.__new__(implementation.MultiLevelKvCacheModule)
    module.args = SimpleNamespace(cpu_cache_token_page_size=4)
    module.backend = SimpleNamespace(
        is_hybrid_att_model=True,
        _alloc_req_kv_mem=alloc,
        model=SimpleNamespace(att_state_manager=resources, req_manager=SimpleNamespace(req_to_token_indexs=table)),
    )
    module.cpu_cache_client = SimpleNamespace(cpu_kv_cache_tensor=None)
    module.need_sync_compute_stream = lambda: False
    monkeypatch.setattr(implementation, "Transfer", lambda **kwargs: Transfer(runtime))
    monkeypatch.setattr(implementation, "CudaRuntime", lambda: runtime)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)

    with pytest.raises(ValueError if binding_failure else TransferFailure):
        module.read_req_from_cpu(req, [0], [4], 0)
    assert copied == ([] if binding_failure else ["kv"])
    assert req.hold_kv_len == req.cur_kv_len == 0
    assert not req.hybrid_len_to_big_page_id
    assert resources.bounded_att_big_page_allocator.free_count == 0
    assert resources.reclaimer.collect() == 0
    assert not released
    runtime.events[-1].ready = True
    assert resources.reclaimer.collect() == 1
    assert released == [0, 1, 2, 3]
    assert resources.bounded_att_big_page_allocator.free_count == 1


@pytest.mark.parametrize("tp_size", [1, 2])
def test_failed_cpu_offload_waits_before_dereferencing_pages(monkeypatch, tp_size):
    from contextlib import nullcontext
    from lightllm.server.multi_level_kv_cache import CacheTier
    from lightllm.server.router.model_infer.infer_batch import InferReq

    runtime = Runtime()
    resources = AttentionStateManager(1)
    dereferenced = []

    def fail():
        raise ValueError("copy failed after partial submission")

    resources._cpu_cache_operator = SimpleNamespace(
        get_page_layouts=lambda: {},
        plan_write_req=lambda **kwargs: CopyPlan("pack", (CopyOperation("kv", None, fail),)),
    )
    module = implementation.MultiLevelKvCacheModule.__new__(implementation.MultiLevelKvCacheModule)
    module.args = SimpleNamespace(cpu_cache_token_page_size=4)
    module.gloo_group = None
    module.backend = SimpleNamespace(
        is_hybrid_att_model=False,
        is_master_in_dp=True,
        dp_world_size=tp_size,
        model=SimpleNamespace(
            att_state_manager=resources,
            req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(4).view(1, 4)),
        ),
    )
    module.cpu_cache_client = SimpleNamespace(
        lock=SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
        allocate_pages=lambda *args, **kwargs: ([3], [False]),
        deref_pages=lambda page_list: dereferenced.extend(page_list),
        cpu_kv_cache_tensor=None,
    )
    req = SimpleNamespace(
        cache_tiers=(CacheTier.CPU,),
        shm_req=SimpleNamespace(
            token_hash_list=SimpleNamespace(get_all=lambda: [123]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [4]),
        ),
        req_idx=0,
        cur_kv_len=4,
        cpu_cache_task_status=InferReq._CpuCacheTaskStatus.NOT_STARTED,
    )
    monkeypatch.setattr(implementation, "Transfer", lambda **kwargs: Transfer(runtime))
    monkeypatch.setattr(implementation.g_infer_context, "is_hybrid_att_model", False)
    monkeypatch.setattr(implementation.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    with pytest.raises(TransferFailure):
        module.write_req_to_cpu(req, object())
    assert req.cpu_cache_task_status.is_not_started()
    assert resources.reclaimer.collect() == 0
    assert dereferenced == []
    runtime.events[-1].ready = True
    resources.reclaimer.collect()
    # A local fence proves nothing about another rank that may still write this CPU page.
    assert dereferenced == ([3] if tp_size == 1 else [])
