from types import SimpleNamespace

import numpy as np
import torch

# 先加载通用预处理模块，沿用生产初始化顺序，避免 basemodel 与 req_manager
# 在独立运行本测试文件时形成模块级循环导入。
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans
from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl


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


def test_pd_decode_reserves_model_pages_but_transfers_only_logical_kv(monkeypatch):
    table = torch.full((1, 16), -1, dtype=torch.int32)
    table[0, :4] = torch.arange(4, dtype=torch.int32)
    alloc_sizes = []
    next_index = [4]
    tasks = []
    backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(pd_kv_page_size=3)
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=table))
    backend.is_master_in_dp = False
    backend._alloc_req_kv_mem = lambda req, size: _reserve_into_table(table, req, alloc_sizes, next_index, size)

    def create_task(**kwargs):
        tasks.append(
            (
                kwargs["kv_start_index"],
                kwargs["kv_end_index"],
                kwargs["mem_indexes"],
                kwargs.get("page_kind", "kv"),
            )
        )
        kwargs["group"].task_list.append(SimpleNamespace())

    backend._create_pd_trans_task = create_task
    monkeypatch.setattr(decode_impl.g_infer_context, "is_linear_att_mixed_model", False)
    req = _bind_alloc_need(
        SimpleNamespace(
            req_idx=0,
            cur_kv_len=4,
            hold_kv_len=4,
            pd_trans_kv_start_index=0,
            shm_req=SimpleNamespace(input_len=10),
        ),
        page_size=4,
    )

    backend._decode_node_gen_trans_tasks(req)

    assert alloc_sizes == [8]
    assert req.cur_kv_len == 10
    assert req.hold_kv_len == 12
    assert tasks == [
        (4, 7, [4, 5, 6], "kv"),
        (7, 10, [7, 8, 9], "kv"),
    ]
    assert table[0, :12].tolist() == list(range(12))


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
