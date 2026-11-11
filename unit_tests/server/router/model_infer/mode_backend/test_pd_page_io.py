from types import SimpleNamespace

import numpy as np
import torch
import pytest


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_hybrid_pd_uses_the_shared_att_state_task_without_model_flags(monkeypatch, role):
    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl import (
        PDChunkedPrefillForPrefillNode,
    )
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl.decode_impl import PDDecodeNode

    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    cls = PDChunkedPrefillForPrefillNode if role == "prefill" else PDDecodeNode
    backend = cls.__new__(cls)
    backend.args = SimpleNamespace(pd_kv_page_size=256, page_size=256)
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(512).view(1, -1)))
    backend.is_master_in_dp = role == "decode"
    groups, tasks = [], []
    backend.info_queue = SimpleNamespace(put=groups.append)
    req = SimpleNamespace(
        req_idx=0,
        shm_req=SimpleNamespace(input_len=512),
        cur_kv_len=512 if role == "prefill" else 0,
        hold_kv_len=0,
        pd_trans_kv_start_index=0,
        _kv_cache_alloc_need=lambda end: end,
    )

    def create(**kwargs):
        task = SimpleNamespace(
            page_kind=kwargs.get("page_kind", "kv"), first_gen_token_id=None, first_gen_token_logprob=None
        )
        tasks.append(task)
        if "group" in kwargs:
            kwargs["group"].task_list.append(task)
        return task

    backend._create_pd_trans_task = create
    if role == "prefill":
        backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=1)
        assert tasks[-1].first_gen_token_id == 17
    else:
        backend._alloc_req_kv_mem = lambda req, count: torch.arange(count)
        backend._decode_node_gen_trans_tasks(req)
        assert len(groups) == 1
    assert [task.page_kind for task in tasks] == ["kv", "kv", "att_state"]


def test_pd_prefill_reads_ready_kv_indices_once_per_chunk(monkeypatch):
    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl_for_dp import (
        PDDPChunkedForPrefillNode,
    )

    reads, tasks = [], []
    indices = torch.arange(20, dtype=torch.int32).flip(0).view(1, -1)

    class IndexTable:
        def __getitem__(self, key):
            reads.append(key)
            return indices[key]

    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    backend = PDDPChunkedForPrefillNode.__new__(PDDPChunkedForPrefillNode)
    backend.args = SimpleNamespace(pd_kv_page_size=4, page_size=2)
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=IndexTable()))
    backend.is_master_in_dp = True
    backend.is_deepseek_v4, backend.dp_rank_in_node, backend.node_world_size = False, 0, 4
    backend.info_queue = SimpleNamespace(put=tasks.append)
    req = SimpleNamespace(
        req_id=1,
        req_idx=0,
        cur_kv_len=7,
        shm_req=SimpleNamespace(input_len=17),
        pd_trans_kv_start_index=4,
        pd_trans_device_id=-1,
        pd_task_num=0,
        sampling_param=SimpleNamespace(
            pd_master_node_id=0,
            pd_decode_node=SimpleNamespace(agent_name="D", agent_metadata=b"D", num_pages=8, page_reg_desc=b"page"),
        ),
    )

    backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=0)
    assert reads == [] and tasks == []

    req.cur_kv_len = 15
    backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=0)
    assert reads == [(0, slice(4, 12))]
    assert req.pd_trans_kv_start_index == 12

    req.cur_kv_len = 17
    backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=1)
    assert reads == [(0, slice(4, 12)), (0, slice(12, 17))]
    assert [(task.start_kv_index, task.end_kv_index) for task in tasks] == [
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 17),
        (17, 17),
    ]
    assert all(task.mem_indexes.dtype == np.int32 for task in tasks[:-1])
    assert [task.mem_indexes.tolist() for task in tasks[:-1]] == [[15, 14, 13, 12], [11, 10, 9, 8], [7, 6, 5, 4], [3]]
    assert tasks[-1].mem_indexes == []
    assert [task.first_gen_token_id for task in tasks] == [None, None, None, None, 17]
    assert tasks[-1].page_kind == "att_state"
    assert tasks[-1].first_gen_token_logprob == -0.2
    assert req.pd_task_num == len(tasks)
