from types import SimpleNamespace

import numpy as np
import torch
import pytest

from lightllm.common.kv_cache_mem_manager import mem_manager


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
        shm_req=SimpleNamespace(input_len=512, sample_params=SimpleNamespace(pd_decode_allocation_policy=0)),
        cur_kv_len=512 if role == "prefill" else 0,
        hold_kv_len=0,
        pd_trans_kv_start_index=0,
        _kv_cache_alloc_need=lambda end: end,
    )

    def create(**kwargs):
        task = SimpleNamespace(
            page_kind=kwargs.get("page_kind", "kv"),
            mem_indexes=kwargs["mem_indexes"],
            first_gen_token_id=None,
            first_gen_token_logprob=None,
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
        allocator_indexes = torch.arange(512, dtype=torch.int32)
        backend._alloc_req_kv_mem = lambda req, count: allocator_indexes[:count]
        backend._decode_node_gen_trans_tasks(req)
        assert len(groups) == 1
        allocator_indexes.fill_(-1)
        assert all(task.mem_indexes.dtype == np.int32 for task in tasks[:2])
        assert [task.mem_indexes.tolist() for task in tasks[:2]] == [list(range(256)), list(range(256, 512))]
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
    backend.processed_prefill_tokens = 30
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=IndexTable()))
    backend.is_master_in_dp = True
    backend.is_deepseek_v4, backend.dp_rank_in_node, backend.node_world_size = False, 0, 4
    backend.info_queue = SimpleNamespace(put=tasks.append)
    req = SimpleNamespace(
        req_id=1,
        req_idx=0,
        cur_kv_len=7,
        shm_req=SimpleNamespace(input_len=17, sample_params=SimpleNamespace(pd_decode_allocation_policy=0)),
        pd_trans_kv_start_index=4,
        pd_resume_processed_tokens=None,
        pd_trans_device_id=-1,
        pd_task_num=0,
        sampling_param=SimpleNamespace(
            pd_master_node_id=0,
            pd_decode_node=SimpleNamespace(agent_name="D", agent_metadata=b"D", num_pages=8, page_reg_desc=b"page"),
        ),
    )

    backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=0)
    assert reads == [] and tasks == []
    assert req.pd_resume_processed_tokens is None

    req.cur_kv_len = 15
    backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=0)
    assert reads == [(0, slice(4, 12))]
    assert req.pd_trans_kv_start_index == 12
    assert req.pd_resume_processed_tokens == 30

    req.cur_kv_len = 17
    backend._prefill_chuncked_handle_func(req, next_token_id=17, next_token_prob=-0.2, output_len=1)
    assert reads == [(0, slice(4, 12)), (0, slice(12, 17))]
    assert req.pd_resume_processed_tokens is None
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


@pytest.mark.parametrize("mode", ["PREFILL_COMPLETE", "LAST_CHUNK"])
@pytest.mark.parametrize("outcome", ["success", "abort_unbound", "transfer_error"])
def test_deferred_prefill_retains_and_delivers_final_state(monkeypatch, mode, outcome):
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.pd_io_struct import PDAllocationReady, PDDecodeAllocationPolicy
    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    from lightllm.server.router.model_infer.mode_backend.pd.prefill_node_impl.prefill_impl_for_dp import (
        PDDPChunkedForPrefillNode,
    )

    length = 9
    shm = SimpleNamespace(
        input_len=length,
        sample_params=SimpleNamespace(
            pd_decode_allocation_policy=PDDecodeAllocationPolicy[mode], pd_ready_timeout_seconds=600
        ),
        shm_prompt_ids=SimpleNamespace(arr=np.array([0] * length + [17])),
        shm_logprobs=SimpleNamespace(arr={"logprob": np.array([0.0] * length + [-0.125])}),
        candetoken_out_len=0,
    )
    req = SimpleNamespace(
        req_id=32,
        req_idx=0,
        shm_req=shm,
        cur_kv_len=0,
        cur_output_len=0,
        infer_aborted=False,
        pd_allocation_ready_sent=False,
        pd_defer_token_publication=False,
        pd_trans_kv_start_index=0,
        pd_task_num=0,
        pd_task_success_num=0,
        pd_task_failed_num=0,
        pd_trans_device_id=-1,
        finish_status=FinishStatus(),
        sampling_param=SimpleNamespace(pd_decode_node=None, pd_master_node_id=88),
        get_chuncked_input_token_len=lambda: 4,
        tail_small_page_buffer_id=36,
        hybrid_cache_len=8,
    )
    events = []
    backend = PDDPChunkedForPrefillNode.__new__(PDDPChunkedForPrefillNode)
    backend.args = SimpleNamespace(pd_kv_page_size=4, page_size=2)
    backend.model = SimpleNamespace(req_manager=SimpleNamespace(req_to_token_indexs=torch.arange(length)[None]))
    backend.processed_prefill_tokens = 16
    backend.info_queue = SimpleNamespace(put=events.append)
    backend.is_master_in_dp = backend.is_deepseek_v4 = True
    backend.dp_rank_in_node = 0
    backend.node_world_size = 1
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(g_infer_context, "requests_mapping", {req.req_id: req})

    backend._before_pd_prefill([req])
    req.cur_kv_len = 4
    backend._prefill_chuncked_handle_func(req, 0, 0, 0)
    assert req.pd_resume_processed_tokens == 16
    assert not events
    req.get_chuncked_input_token_len = lambda: length
    backend._before_pd_prefill([req])
    backend._before_pd_prefill([req])
    assert len(events) == int(mode == "LAST_CHUNK")
    req.cur_kv_len, req.cur_output_len = length, 1
    req.finish_status.set_status(FinishStatus.FINISHED_LENGTH)
    backend._prefill_chuncked_handle_func(req, 17, -0.125, 1)
    assert len(events) == 1 and isinstance(events[0], PDAllocationReady)
    assert (events[0].group_request_id, events[0].pd_master_node_id) == (32, 88)
    assert backend._filter_not_ready_reqs([32]) == []
    assert req.pd_task_num == shm.candetoken_out_len == 0

    binding = SimpleNamespace(
        ready_kv_len=length - 1, agent_name="D", agent_metadata=b"", num_pages=2, page_reg_desc=b""
    )
    if outcome == "abort_unbound":
        req.infer_aborted = True
        backend._bind_pd_decode_node(req, binding)
        assert req.pd_task_num == 0
    else:
        backend._bind_pd_decode_node(req, binding)
        backend._bind_pd_decode_node(req, binding)
        assert req.pd_task_num == 2
        assert [task.page_kind for task in events[1:]] == ["kv", "att_state"]
        kv, state = events[1:]
        assert (kv.start_kv_index, kv.end_kv_index, kv.mem_indexes.tolist()) == (8, 9, [8])
        assert (state.req_idx, state.checkpoint_buffer_id, state.checkpoint_len) == (0, 36, 8)
        assert (state.first_gen_token_id, state.first_gen_token_logprob) == (17, -0.125)
        assert backend._filter_not_ready_reqs([32]) == []
        req.pd_task_success_num = 1
        assert backend._filter_not_ready_reqs([32]) == []
        if outcome == "success":
            req.pd_task_success_num += 1
        else:
            req.pd_task_failed_num += 1
    assert backend._filter_not_ready_reqs([32]) == [req]
    assert shm.candetoken_out_len == 1
    assert shm.finish_status.status == {
        "abort_unbound": FinishStatus.FINISHED_ABORTED,
        "transfer_error": FinishStatus.FINISHED_ERROR,
        "success": FinishStatus.FINISHED_LENGTH,
    }[outcome]


def test_decode_page_reader_uses_owned_index_buffer_for_numpy_indices(monkeypatch):
    owner = mem_manager.MemoryManager.__new__(mem_manager.MemoryManager)
    owner._buffer_mem_indexes_tensors = [torch.zeros(4, dtype=torch.int64)]
    owner.kv_move_buffer = [object()]
    owner.kv_buffer = object()
    calls = []
    monkeypatch.setattr(torch.Tensor, "cuda", lambda buf, non_blocking: buf)
    monkeypatch.setattr(mem_manager, "page_io", lambda **kwargs: calls.append(kwargs))

    owner.read_page_kv_move_buffer_to_mem(np.array([3, 1], dtype=np.int32), 0, 0, [owner], 1)

    assert len(calls) == 1
    indices = calls[0]["mem_indexes"]
    assert indices.data_ptr() == owner._buffer_mem_indexes_tensors[0].data_ptr()
    assert indices.tolist() == [3, 1]
