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


@pytest.mark.parametrize("is_hybrid_att_model", [False, True])
def test_pd_decode_reserves_model_pages_but_transfers_only_logical_kv(monkeypatch, is_hybrid_att_model):
    table = torch.full((1, 16), -1, dtype=torch.int32)
    table[0, :4] = torch.arange(4, dtype=torch.int32)
    alloc_sizes = []
    next_index = [4]
    tasks = []
    backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    backend.args = SimpleNamespace(pd_kv_page_size=4, page_size=4)
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=table))
    backend.is_master_in_dp = False
    backend._alloc_req_kv_mem = lambda req, size: _reserve_into_table(
        table=table,
        req=req,
        alloc_sizes=alloc_sizes,
        next_index=next_index,
        alloc_token_num=size,
    )

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
    monkeypatch.setattr(decode_impl.g_infer_context, "is_hybrid_att_model", is_hybrid_att_model)
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
    expected_tasks = [
        (4, 8, [4, 5, 6, 7], "kv"),
        (8, 10, [8, 9], "kv"),
    ]
    if is_hybrid_att_model:
        expected_tasks.append((10, 10, [], "att_state"))
    assert tasks == expected_tasks
    assert table[0, :12].tolist() == list(range(12))


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
        backend._prefill_chuncked_handle_func(
            req_obj=req,
            next_token_id=1,
            next_token_prob=1.0,
            output_len=0,
        )


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_dp_cache_fetch_reserves_pages_and_keeps_logical_transfer_size(monkeypatch, tp_size, tp_rank):
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
        is_hybrid_att_model=False,
        model=SimpleNamespace(mem_manager=SimpleNamespace()),
        dp_world_size=tp_size,
        rank_in_dp=tp_rank,
        node_nccl_group=None,
        dp_req_to_token_indices={0: source_table, 1: source_table},
        is_hybrid_att_model=False,
    )
    backend._alloc_req_kv_mem = lambda cur_req, size: _reserve_into_table(
        table=table,
        req=cur_req,
        alloc_sizes=alloc_sizes,
        next_index=next_index,
        alloc_token_num=size,
    )
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
    assert tasks[0].src_dp_rank == 1
    assert tasks[0].dst_token_indices.tolist() == list(range(4, 10))
    assert len(tasks[0].src_token_indices) == 6


@pytest.mark.parametrize(
    "node_size,tp_size,rank_in_node,expected_ranks",
    [
        (4, 2, 0, [0, 2]),
        (4, 2, 1, [1, 3]),
        (4, 2, 2, [0, 2]),
        (4, 2, 3, [1, 3]),
        (6, 2, 5, [1, 3, 5]),
        (2, 4, 1, [1]),
    ],
)
def test_dp_imports_only_matching_tp_shards(
    monkeypatch,
    node_size,
    tp_size,
    rank_in_node,
    expected_ranks,
):
    from lightllm.server.router.model_infer.mode_backend import base_backend
    from lightllm.server.router.model_infer import shared_att_resources_utils

    imported_ranks = []
    registered_ranks = []
    managers = {}
    tables = {}
    for rank in expected_ranks:
        managers[rank] = SimpleNamespace(
            register_page_pools_pin_memory=lambda rank=rank: registered_ranks.append(rank),
        )
        tables[rank] = object()

    def load_storage(rank, kind):
        assert kind == "dp"
        imported_ranks.append(rank)
        return managers[rank], tables[rank]

    local_manager = managers[rank_in_node]
    backend = SimpleNamespace(
        args=SimpleNamespace(running_max_req_size=8),
        dp_size_in_node=max(1, node_size // tp_size),
        dp_world_size=tp_size,
        node_world_size=node_size,
        rank_in_node=rank_in_node,
        model=SimpleNamespace(
            att_state_manager=local_manager,
            req_manager=SimpleNamespace(req_to_token_indexs=tables[rank_in_node]),
        ),
    )
    monkeypatch.setattr(base_backend, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(base_backend.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(dp_shared_kv_trans, "DPKVSharedMoudle", lambda **kwargs: object())
    monkeypatch.setattr(shared_att_resources_utils, "load_att_resources", load_storage)

    base_backend.ModeBackend.init_dp_kv_shared(backend)

    remote_ranks = [rank for rank in expected_ranks if rank != rank_in_node]
    assert imported_ranks == registered_ranks == remote_ranks
    assert list(backend.dp_att_state_managers) == list(range(len(expected_ranks)))
    assert list(backend.dp_req_to_token_indices) == list(range(len(expected_ranks)))
    for dp_rank, rank in enumerate(expected_ranks):
        assert backend.dp_att_state_managers[dp_rank] is managers[rank]
        assert backend.dp_req_to_token_indices[dp_rank] is tables[rank]


def _bind_alloc_need(req, page_size):
    req.args = SimpleNamespace(page_size=page_size)
    req._kv_cache_alloc_need = lambda target_len: InferReq._kv_cache_alloc_need(req, target_len)
    return req


def _reserve_into_table(
    table,
    req,
    alloc_sizes,
    next_index,
    alloc_token_num,
):
    if alloc_token_num == 0:
        return None
    alloc_sizes.append(alloc_token_num)
    new_hold_kv_len = req.hold_kv_len + alloc_token_num
    mem_indexes = torch.arange(next_index[0], next_index[0] + alloc_token_num, dtype=torch.int32)
    table[req.req_idx, req.hold_kv_len : new_hold_kv_len] = mem_indexes
    next_index[0] += alloc_token_num
    req.hold_kv_len = new_hold_kv_len
    return mem_indexes
