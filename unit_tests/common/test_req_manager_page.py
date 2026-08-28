from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process


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
    return context


def _make_req(req_idx):
    req = SimpleNamespace(req_idx=req_idx, cur_kv_len=0, hold_kv_len=0)
    req._kv_cache_alloc_need = lambda target_len: (target_len + 3) // 4 * 4 - req.hold_kv_len
    return req


def test_request_reuses_reserved_page_tail_before_allocating_next_page(monkeypatch):
    context = _make_context(monkeypatch)
    req = _make_req(0)

    indexes = generic_pre_process._alloc_kv_mem_indexes([req], [3], 3)
    assert indexes.tolist() == [0, 1, 2]
    assert req.hold_kv_len == 4
    assert context.req_manager.mem_manager.alloc_sizes == [4]
    assert context.req_manager.req_to_token_indexs[0, :4].tolist() == [0, 1, 2, 3]

    req.cur_kv_len = 3
    indexes = generic_pre_process._alloc_kv_mem_indexes([req], [4], 1)
    assert indexes.tolist() == [3]
    assert context.req_manager.mem_manager.alloc_sizes == [4]

    req.cur_kv_len = 4
    indexes = generic_pre_process._alloc_kv_mem_indexes([req], [6], 2)
    assert indexes.tolist() == [4, 5]
    assert req.hold_kv_len == 8
    assert context.req_manager.mem_manager.alloc_sizes == [4, 4]
    assert context.req_manager.req_to_token_indexs[0, :8].tolist() == list(range(8))


def test_step_indexes_follow_request_order(monkeypatch):
    _make_context(monkeypatch)
    req0 = _make_req(0)
    req1 = _make_req(1)

    indexes = generic_pre_process._alloc_kv_mem_indexes([req0, req1], [2, 3], 5)

    assert indexes.tolist() == [0, 1, 4, 5, 6]
    assert req0.hold_kv_len == req1.hold_kv_len == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_indexes_are_assembled_from_gpu_request_table(monkeypatch):
    context = _make_context(monkeypatch)
    context.req_manager.req_to_token_indexs = context.req_manager.req_to_token_indexs.cuda()
    req = _make_req(0)

    indexes = generic_pre_process._alloc_kv_mem_indexes([req], [3], 3)

    assert indexes.is_cuda
    assert indexes.cpu().tolist() == [0, 1, 2]
