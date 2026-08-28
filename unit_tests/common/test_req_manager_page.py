from types import SimpleNamespace

import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process
from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext
from lightllm.server.router.model_infer.mode_backend import base_backend


class _FakeMemManager:
    page_size = 4

    def __init__(self):
        self.next_index = 0
        self.alloc_sizes = []

    def alloc(self, size):
        self.alloc_sizes.append(size)
        result = torch.arange(self.next_index, self.next_index + size, dtype=torch.int32)
        self.next_index += size
        return result


def _make_context(monkeypatch):
    mem_manager = _FakeMemManager()
    req_manager = SimpleNamespace(
        mem_manager=mem_manager,
        req_to_token_indexs=torch.full((2, 16), -1, dtype=torch.int32),
    )
    context = SimpleNamespace(
        args=SimpleNamespace(page_size=4),
        req_manager=req_manager,
        radix_cache=None,
    )
    monkeypatch.setattr(generic_pre_process, "g_infer_context", context)
    monkeypatch.setattr(base_backend, "g_infer_context", context)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.args = context.args
    return context, backend


def _make_req(req_idx):
    req = SimpleNamespace(req_idx=req_idx, cur_kv_len=0, hold_kv_len=0)
    req._kv_cache_alloc_need = lambda target_len: (target_len + 3) // 4 * 4 - req.hold_kv_len
    return req


def test_request_reuses_reserved_page_tail_before_allocating_next_page(monkeypatch):
    context, backend = _make_context(monkeypatch)
    req = _make_req(0)

    backend._alloc_req_kv_mem(req, req._kv_cache_alloc_need(3))
    assert req.hold_kv_len == 4
    assert context.req_manager.mem_manager.alloc_sizes == [4]
    assert context.req_manager.req_to_token_indexs[0, :4].tolist() == [0, 1, 2, 3]

    req.cur_kv_len = 3
    backend._alloc_req_kv_mem(req, req._kv_cache_alloc_need(4))
    assert context.req_manager.mem_manager.alloc_sizes == [4]

    req.cur_kv_len = 4
    backend._alloc_req_kv_mem(req, req._kv_cache_alloc_need(6))
    assert req.hold_kv_len == 8
    assert context.req_manager.mem_manager.alloc_sizes == [4, 4]
    assert context.req_manager.req_to_token_indexs[0, :8].tolist() == list(range(8))


def test_reservation_fills_each_request_table_row(monkeypatch):
    _, backend = _make_context(monkeypatch)
    req0 = _make_req(0)
    req1 = _make_req(1)

    backend._alloc_req_kv_mem(req0, req0._kv_cache_alloc_need(2))
    backend._alloc_req_kv_mem(req1, req1._kv_cache_alloc_need(3))

    assert generic_pre_process.g_infer_context.req_manager.req_to_token_indexs[0, :4].tolist() == [0, 1, 2, 3]
    assert generic_pre_process.g_infer_context.req_manager.req_to_token_indexs[1, :4].tolist() == [4, 5, 6, 7]
    assert req0.hold_kv_len == req1.hold_kv_len == 4


def test_decode_reserves_mtp_headroom(monkeypatch):
    context, backend = _make_context(monkeypatch)
    req = _make_req(0)
    req.cur_kv_len = 3
    req.hold_kv_len = 4
    req.mtp_step = 2
    req.multimodal_params = {"images": [], "audios": []}
    req.shared_kv_node = None
    req.get_cur_total_len = lambda: 4
    req.get_radix_cache_shared_len = lambda: 0
    req.args = context.args

    context.req_manager.req_to_token_indexs[0, :4] = torch.arange(4, dtype=torch.int32)
    context.req_manager.mem_manager.next_index = 4
    alloc_token_num = InferReq._mtp_decode_need_token_num(req)
    backend._alloc_req_kv_mem(req, alloc_token_num)

    model_input, run_reqs = generic_pre_process.prepare_decode_inputs([req])

    assert run_reqs == [req, req, req]
    assert model_input.b_seq_len.tolist() == [4, 5, 6]
    assert model_input.mem_indexes_cpu is None
    assert req.hold_kv_len == 12
    assert context.req_manager.mem_manager.alloc_sizes == [8]
    assert context.req_manager.req_to_token_indexs[0, :12].tolist() == list(range(12))


def test_page_size_one_uses_the_same_scheduler_preallocation(monkeypatch):
    context, backend = _make_context(monkeypatch)
    context.args.page_size = 1
    req = _make_req(0)
    req.cur_kv_len = 3
    req.hold_kv_len = 3
    req.mtp_step = 2
    req.multimodal_params = {"images": [], "audios": []}
    req.shared_kv_node = None
    req.get_cur_total_len = lambda: 4
    req.get_radix_cache_shared_len = lambda: 0
    req.args = context.args
    req._kv_cache_alloc_need = lambda target_len: InferReq._kv_cache_alloc_need(req, target_len)

    context.req_manager.req_to_token_indexs[0, :3] = torch.arange(3, dtype=torch.int32)
    context.req_manager.mem_manager.next_index = 3
    alloc_token_num = InferReq._mtp_decode_need_token_num(req)
    backend._alloc_req_kv_mem(req, alloc_token_num)

    model_input, _ = generic_pre_process.prepare_decode_inputs([req])

    assert model_input.mem_indexes_cpu is None
    assert model_input.mem_indexes_from_req_table is True
    assert req.hold_kv_len == 12
    assert context.req_manager.mem_manager.alloc_sizes == [9]
    assert context.req_manager.req_to_token_indexs[0, :12].tolist() == list(range(12))


def test_page_size_one_frees_all_preallocated_indexes():
    infer_context = InferenceContext.__new__(InferenceContext)
    infer_context.args = SimpleNamespace(page_size=1)
    infer_context.radix_cache = None
    infer_context.req_manager = SimpleNamespace(req_to_token_indexs=torch.arange(16, dtype=torch.int32)[None, :])
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=3,
        hold_kv_len=12,
        shm_req=SimpleNamespace(shm_cur_kv_len=3),
    )
    free_token_indexes = []

    infer_context.free_a_req_mem(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == list(range(12))
    assert req.cur_kv_len == req.hold_kv_len == 0
