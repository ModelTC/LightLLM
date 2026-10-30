"""DP 前缀迁移：独立共享大页列表、目标页分配和源/目标索引配对。"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.att_resources.allocators import StatePageAllocator
from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans


@pytest.mark.parametrize("hybrid,max_req_len", [(False, 32), (True, 32), (True, 4)])
def test_dp_page_ids_use_separate_shared_array(monkeypatch: pytest.MonkeyPatch, hybrid: bool, max_req_len: int):
    arrays = {}

    def shared_array(name: str, shape: tuple[int, ...], dtype: type) -> SimpleNamespace:
        array = SimpleNamespace(arr=np.full(shape, -1, dtype=dtype), create_shm=lambda: None)
        arrays[name] = array
        return array

    monkeypatch.setattr(dp_shared_kv_trans, "ShmArray", shared_array)
    monkeypatch.setattr(dp_shared_kv_trans, "get_dp_rank_in_node", lambda: 0)
    monkeypatch.setattr(dp_shared_kv_trans, "get_env_start_args", lambda: SimpleNamespace(diverse_mode=False))
    backend = SimpleNamespace(
        is_hybrid_att_model=hybrid,
        args=SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2, max_req_total_len=max_req_len),
    )
    module = dp_shared_kv_trans.DPKVSharedMoudle(max_req_num=3, dp_size_in_node=2, backend=backend)
    assert arrays["dp_shared_req_infos"].arr.shape == (3, 2, 2)
    if hybrid and max_req_len >= 8:
        assert arrays["dp_shared_req_big_page_ids"].arr.shape == (3, 2, 4)
    else:
        assert module.shared_req_big_page_ids is None
        assert list(arrays) == ["dp_shared_req_infos"]


def test_fill_orders_radix_and_owned_pages_and_clears_reused_rows(monkeypatch: pytest.MonkeyPatch):
    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = SimpleNamespace(is_master_in_dp=True, node_nccl_group=None)
    module.dp_rank_in_node = 0
    module.big_page_tokens = 8
    module.shared_req_infos = SimpleNamespace(arr=np.full((2, 2, 2), 99, dtype=np.int64))
    module.shared_req_big_page_ids = SimpleNamespace(arr=np.full((2, 2, 4), 99, dtype=np.int64))
    node = object()
    context = SimpleNamespace(radix_cache=SimpleNamespace(get_big_page_ids_by_node=lambda _: [40]))
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", context)
    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", lambda group: None)
    req = SimpleNamespace(
        cur_kv_len=24,
        req_idx=7,
        shared_kv_node=node,
        hybrid_len_to_big_page_id={24: 3, 16: 12, 32: 88},
    )
    second = SimpleNamespace(
        cur_kv_len=8,
        req_idx=5,
        shared_kv_node=None,
        hybrid_len_to_big_page_id={8: 6},
    )
    module.fill_reqs_info([req, second])
    assert module.shared_req_infos.arr[:, 0].tolist() == [[24, 7], [8, 5]]
    assert module.shared_req_big_page_ids.arr[:, 0].tolist() == [[40, 12, 3, -1], [6, -1, -1, -1]]
    assert (module.shared_req_big_page_ids.arr[:, 1] == 99).all()

    req.cur_kv_len = 8
    module.fill_reqs_info([req])
    assert module.shared_req_big_page_ids.arr[0, 0].tolist() == [40, -1, -1, -1]
    module.backend.is_master_in_dp = False
    req.cur_kv_len = 0
    module.fill_reqs_info([req])
    assert module.shared_req_infos.arr[0, 0, 0] == 8


@pytest.mark.parametrize("capacity,src_kv_len", [(2, 26), (1, 26), (0, 14)])
def test_build_reserves_only_new_big_pages_before_kv(
    monkeypatch: pytest.MonkeyPatch, capacity: int, src_kv_len: int,
):
    module, req, allocator, allocations = _build_fixture(monkeypatch, capacity, src_kv_len)
    if capacity == 1:
        with pytest.raises(AssertionError, match="DP fetch big page allocation must succeed"):
            module.build_shared_kv_trans_tasks([req], req_dp_ranks=[0])
        assert not allocations
        assert req.hold_kv_len == 8 and allocator.free_count == 1
        return
    tasks = module.build_shared_kv_trans_tasks([req], req_dp_ranks=[0])
    task, = tasks
    assert task.dst_req is req and task.src_req_idx == 1
    assert task.src_dp_rank == 1
    assert task.src_token_indexes.tolist() == list(range(40, 32 + src_kv_len))
    assert len(task.dst_token_indexes) == src_kv_len - 8
    assert allocations == [((src_kv_len + 3) // 4) * 4 - 8]
    assert req.cur_kv_len == 8 and req.hybrid_len_to_big_page_id == {8: 77}
    if src_kv_len == 26:
        assert task.src_big_page_ids == [20, 21]
        assert task.dst_big_page_ids == [0, 1]
        assert allocator.free_count == 0
    else:
        assert task.src_big_page_ids == task.dst_big_page_ids == []


def test_missing_source_page_fails_before_allocating(monkeypatch: pytest.MonkeyPatch):
    module, req, allocator, allocations = _build_fixture(monkeypatch, 2, 26)
    module.shared_req_big_page_ids.arr[0, 1, 2] = -1
    with pytest.raises(AssertionError, match="missing prefix big pages"):
        module.build_shared_kv_trans_tasks([req], req_dp_ranks=[0])
    assert not allocations and allocator.free_count == 2


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("local_dp_rank", [0, 1])
@pytest.mark.parametrize("local_len,remote_len", [(24, 16), (24, 24), (31, 24), (0, 0)])
def test_local_longest_prefix_skips_copy_but_keeps_barriers(
    monkeypatch: pytest.MonkeyPatch,
    hybrid: bool,
    local_dp_rank: int,
    local_len: int,
    remote_len: int,
):
    # 并列最长且本地为 DP 1 时，argmax 会优先选 DP 0；仍应在选择来源前跳过复制。
    allocate_pages = Mock(side_effect=AssertionError("unexpected big page allocation"))
    allocate_tokens = Mock(side_effect=AssertionError("unexpected KV allocation"))
    copy_resources = Mock(side_effect=AssertionError("unexpected DP copy"))
    allocation_need = Mock(side_effect=AssertionError("unexpected allocation calculation"))
    available_tokens = Mock(side_effect=AssertionError("unexpected capacity query"))
    barriers = []
    manager = SimpleNamespace(
        bounded_att_big_page_allocator=SimpleNamespace(alloc=allocate_pages),
        copy_from_other_dp=copy_resources,
    )
    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = SimpleNamespace(
        is_hybrid_att_model=hybrid,
        is_master_in_dp=True,
        node_nccl_group="node",
        model=SimpleNamespace(att_resource_manager=manager),
        dp_att_resource_managers={},
        dp_req_to_token_indices={},
        _alloc_req_kv_mem=allocate_tokens,
    )
    module.dp_rank_in_node = local_dp_rank
    module.big_page_tokens = 8 if hybrid else 0
    module.shared_req_infos = SimpleNamespace(arr=np.zeros((1, 2, 2), dtype=np.int64))
    module.shared_req_infos.arr[0, 1 - local_dp_rank] = [remote_len, 9]
    module.shared_req_big_page_ids = (
        SimpleNamespace(arr=np.full((1, 2, 4), -1, dtype=np.int64)) if hybrid else None
    )
    owned_pages = {length: 70 + length // 8 for length in range(8, local_len + 1, 8)} if hybrid else {}
    req = SimpleNamespace(
        req_idx=7,
        cur_kv_len=local_len,
        hold_kv_len=local_len,
        shared_kv_node=None,
        hybrid_len_to_big_page_id=owned_pages.copy(),
        get_cur_total_len=lambda: 32,
        _kv_cache_alloc_need=allocation_need,
        shm_req=SimpleNamespace(shm_cur_kv_len=local_len),
    )
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", SimpleNamespace(get_can_alloc_token_num=available_tokens))
    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", lambda group: barriers.append(group))

    module.fill_reqs_info([req])
    assert module.shared_req_infos.arr[0, local_dp_rank].tolist() == [local_len, req.req_idx]
    tasks = module.build_shared_kv_trans_tasks([req], req_dp_ranks=[local_dp_rank])
    assert tasks == []
    module.kv_trans(tasks)

    for operation in (allocate_pages, allocate_tokens, copy_resources, allocation_need, available_tokens):
        operation.assert_not_called()
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == local_len
    assert req.hybrid_len_to_big_page_id == owned_pages
    # 即使不接收数据，也必须参加所有屏障，保证其他 DP 能完成读取后再释放来源资源。
    assert barriers == ["node", "node", "node"]


def _build_fixture(monkeypatch: pytest.MonkeyPatch, capacity: int, src_kv_len: int) -> tuple:
    allocator = StatePageAllocator(capacity)
    allocations = []
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=8,
        hold_kv_len=8,
        hybrid_len_to_big_page_id={8: 77},
        get_cur_total_len=lambda: 32,
    )
    req._kv_cache_alloc_need = lambda length: ((length + 3) // 4) * 4 - req.hold_kv_len

    def allocate(cur_req: SimpleNamespace, count: int) -> torch.Tensor:
        allocations.append(count)
        cur_req.hold_kv_len += count
        return torch.arange(100, 100 + count, dtype=torch.int32)

    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = SimpleNamespace(
        is_hybrid_att_model=True,
        dp_world_size=2,
        rank_in_dp=1,
        node_nccl_group=None,
        model=SimpleNamespace(att_resource_manager=SimpleNamespace(bounded_att_big_page_allocator=allocator)),
        dp_req_to_token_indices={1: torch.arange(64, dtype=torch.int32).reshape(2, 32)},
        _alloc_req_kv_mem=allocate,
    )
    module.dp_rank_in_node = 0
    module.big_page_tokens = 8
    module.shared_req_infos = SimpleNamespace(arr=np.array([[[8, 0], [src_kv_len, 1]]], dtype=np.int64))
    module.shared_req_big_page_ids = SimpleNamespace(arr=np.array([[[77, -1, -1, -1], [10, 20, 21, -1]]]))
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", SimpleNamespace(get_can_alloc_token_num=lambda: 100))
    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", lambda group: None)
    return module, req, allocator, allocations
