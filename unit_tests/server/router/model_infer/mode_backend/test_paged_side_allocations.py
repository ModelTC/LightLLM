from types import SimpleNamespace

import numpy as np
import pytest
import torch

# 先加载通用预处理模块，沿用生产初始化顺序，避免 basemodel 与 req_manager
# 在独立运行本测试文件时形成模块级循环导入。
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl
from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl import prefill_impl


def _bind_alloc_need(req, page_size):
    req.args = SimpleNamespace(page_size=page_size)
    req._kv_cache_alloc_need = lambda target_len: InferReq._kv_cache_alloc_need(req, target_len)
    return req


def _reserve_into_table(table, req, alloc_sizes, next_index, alloc_token_num):
    if alloc_token_num == 0:
        return None
    alloc_sizes.append(alloc_token_num)
    new_hold_kv_len = req.hold_kv_len + alloc_token_num
    mem_indexes = torch.arange(next_index[0], next_index[0] + alloc_token_num, dtype=torch.int32)
    table[req.req_idx, req.hold_kv_len : new_hold_kv_len] = mem_indexes
    next_index[0] += alloc_token_num
    req.hold_kv_len = new_hold_kv_len
    return mem_indexes


@pytest.mark.parametrize("is_hybrid_att_model", [False, True])
def test_pd_decode_reserves_model_pages_only_after_first_write(monkeypatch, is_hybrid_att_model):
    from queue import Queue

    table = torch.full((1, 16), -1, dtype=torch.int32)
    alloc_sizes = []
    next_index = [0]
    backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(pd_kv_page_size=4, page_size=4, mtp_step=0)
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=table))
    backend.is_master_in_dp = True
    backend.node_world_size = backend.dp_world_size = 1
    backend.dp_rank_in_node = 0
    backend.info_queue = Queue()
    backend._alloc_req_kv_mem = lambda req, size: _reserve_into_table(table, req, alloc_sizes, next_index, size)
    monkeypatch.setattr(decode_impl.g_infer_context, "is_hybrid_att_model", is_hybrid_att_model)
    monkeypatch.setattr(decode_impl.g_infer_context, "get_can_alloc_token_num", lambda: 100)
    req = _bind_alloc_need(
        SimpleNamespace(
            req_id=1,
            req_idx=0,
            cur_kv_len=0,
            hold_kv_len=0,
            pd_task_num=0,
            pd_task_failed_num=0,
            pd_trans_device_id=-1,
            infer_aborted=False,
            cur_output_len=0,
            shm_req=SimpleNamespace(input_len=10),
            sampling_param=SimpleNamespace(
                pd_master_node_id=0, shm_param=SimpleNamespace(max_new_tokens=1, ignore_eos=True)
            ),
        ),
        page_size=4,
    )
    monkeypatch.setattr(decode_impl.g_infer_context, "requests_mapping", {1: req})

    backend._decode_node_gen_trans_tasks(req)
    group = backend.info_queue.get_nowait()
    assert alloc_sizes == []
    assert req.cur_kv_len == req.hold_kv_len == 0
    assert req.shm_req.pd_waiting_for_kv
    assert all(t.defer_allocation for t in group.task_list)
    backend._allocate_ready_kv([1])
    assert alloc_sizes == []
    req.pd_first_chunk_ready = True
    backend._allocate_ready_kv([1])
    assert alloc_sizes == [12]
    assert req.cur_kv_len == 10
    assert req.hold_kv_len == 12
    update = backend.info_queue.get_nowait()
    assert update.allocation_update
    assert not req.shm_req.pd_waiting_for_kv
    expected_tasks = [(0, 4, [0, 1, 2, 3], "kv"), (4, 8, [4, 5, 6, 7], "kv"), (8, 10, [8, 9], "kv")]
    if is_hybrid_att_model:
        expected_tasks.append((10, 10, [], "att_state"))
    assert [(t.start_kv_index, t.end_kv_index, t.mem_indexes, t.page_kind) for t in update.task_list] == expected_tasks
    assert all(not t.defer_allocation for t in update.task_list)
    assert table[0, :12].tolist() == list(range(12))
    backend._allocate_ready_kv([1])
    assert alloc_sizes == [12]


def test_pd_decode_rejects_unaligned_transfer_page_size():
    backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(pd_kv_page_size=3, page_size=4)
    req = SimpleNamespace(cur_kv_len=4, shm_req=SimpleNamespace(input_len=10))

    with pytest.raises(AssertionError, match="pd_kv_page_size must be divisible by page_size"):
        backend._decode_node_gen_trans_tasks(req)


def test_pd_prefill_rejects_unaligned_transfer_page_size():
    backend = prefill_impl.PDChunkedPrefillForPrefillNode.__new__(prefill_impl.PDChunkedPrefillForPrefillNode)
    backend.args = SimpleNamespace(pd_kv_page_size=3, page_size=4)
    req = SimpleNamespace(cur_kv_len=4, shm_req=SimpleNamespace(input_len=10))

    with pytest.raises(AssertionError, match="pd_kv_page_size must be divisible by page_size"):
        backend._prefill_chuncked_handle_func(req, next_token_id=1, next_token_prob=1.0, output_len=0)


def test_dp_cache_fetch_reserves_pages_and_keeps_logical_transfer_size(monkeypatch):
    table = torch.full((1, 16), -1, dtype=torch.int32)
    table[0, :4] = torch.arange(4, dtype=torch.int32)
    alloc_sizes = []
    next_index = [4]
    req = _bind_alloc_need(
        SimpleNamespace(
            req_idx=0,
            cur_kv_len=4,
            hold_kv_len=4,
            get_cur_total_len=lambda: 20,
        ),
        page_size=4,
    )
    source_table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    backend = SimpleNamespace(
        model=SimpleNamespace(mem_manager=SimpleNamespace()),
        dp_world_size=1,
        rank_in_dp=0,
        node_nccl_group=None,
        mem_managers=[SimpleNamespace(req_to_token_indexs=source_table)] * 2,
    )
    backend._alloc_req_kv_mem = lambda cur_req, size: _reserve_into_table(table, cur_req, alloc_sizes, next_index, size)
    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = backend
    module.dp_rank_in_node = 0
    module.shared_req_infos = SimpleNamespace(arr=np.array([[[4, 0], [10, 0]]], dtype=np.int64))
    context = SimpleNamespace(
        req_manager=SimpleNamespace(req_to_token_indexs=table),
        get_can_alloc_token_num=lambda: 100,
    )
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", context)
    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", lambda group: None)

    tasks = module.build_shared_kv_trans_tasks([req], req_dp_ranks=[0])

    assert alloc_sizes == [8]
    assert req.cur_kv_len == 4
    assert req.hold_kv_len == 12
    assert len(tasks) == 1
    assert tasks[0].mem_indexes.tolist() == list(range(4, 10))
    assert len(tasks[0].max_kv_len_mem_indexes) == 6
