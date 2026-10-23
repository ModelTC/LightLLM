from types import MethodType, SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process
from lightllm.common.req_manager import ReqManager
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext
from lightllm.server.router.model_infer.mode_backend import base_backend
from lightllm.server.multi_level_kv_cache import CacheTier
from unit_tests.common.cpu_index_helpers import make_cpu_index_manager, bind_cpu_indexes


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


def test_bind_mem_manager_initializes_hold_request_with_reserved_page():
    req_manager = ReqManager.__new__(ReqManager)
    req_manager.HOLD_REQUEST_ID = 2
    req_manager.req_list = SimpleNamespace(is_all_free=lambda: True)
    req_manager.page_size = 4
    req_manager.req_to_token_indexs = torch.full((3, 8), -1, dtype=torch.int32)
    mem_manager = SimpleNamespace(page_size=4, HOLD_TOKEN_MEMINDEXES=(100, 101, 102, 103))

    req_manager.bind_mem_manager(mem_manager)

    assert req_manager.mem_manager is mem_manager
    assert req_manager.req_to_token_indexs[2].tolist() == [100, 101, 102, 103] * 2
    assert req_manager.req_to_token_indexs[:2].eq(-1).all()


def test_hold_request_indexes_require_all_requests_released():
    req_manager = ReqManager.__new__(ReqManager)
    req_manager.req_list = SimpleNamespace(is_all_free=lambda: False)

    with pytest.raises(AssertionError, match="all requests are released"):
        req_manager.init_hold_request_indexs()


def _make_context(monkeypatch):
    mem_manager = _FakeMemManager()
    req_manager = make_cpu_index_manager(torch.full((2, 16), -1, dtype=torch.int32), page_size=4)
    req_manager.mem_manager = mem_manager
    req_manager.max_request_num = 2
    req_manager.page_size = 4

    context = SimpleNamespace(
        args=SimpleNamespace(page_size=4),
        req_manager=req_manager,
        radix_cache=None,
        get_overlap_stream=lambda: None,
    )
    monkeypatch.setattr(generic_pre_process, "g_infer_context", context)
    monkeypatch.setattr(base_backend, "g_infer_context", context)
    monkeypatch.setattr(infer_batch, "g_infer_context", context)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.args = context.args

    def copy_indexes(run_reqs):
        for req in run_reqs:
            start, end = req.token_index.gpu_index_submitted_len, req.hold_kv_len
            req_manager.req_to_token_indexs[req.req_idx, start:end].copy_(
                req.token_index.get_cpu_token_indexes(start, end)
            )
            req.token_index.gpu_index_submitted_len = end

    backend._copy_req_token_indexes_to_gpu = copy_indexes
    return context, backend


def _make_req(req_idx):
    req = InferReq.__new__(InferReq)
    req.req_idx = req_idx
    bind_cpu_indexes(req, infer_batch.g_infer_context.req_manager)
    req.cur_kv_len = req.hold_kv_len = 0
    req._kv_cache_alloc_need = lambda target_len: InferReq._kv_cache_alloc_need(req, target_len)
    return req


def test_request_reuses_reserved_page_tail_before_allocating_next_page(monkeypatch):
    context, backend = _make_context(monkeypatch)
    req = _make_req(0)

    mem_indexes = backend._alloc_req_kv_mem(req, alloc_token_num=4)
    assert mem_indexes.tolist() == [0, 1, 2, 3]
    assert req.hold_kv_len == 4
    assert context.req_manager.mem_manager.alloc_sizes == [4]
    assert context.req_manager.get_cpu_token_indexes(0, 0, 4).tolist() == [0, 1, 2, 3]

    req.cur_kv_len = 3
    assert backend._alloc_req_kv_mem(req, alloc_token_num=0) is None
    assert context.req_manager.mem_manager.alloc_sizes == [4]

    req.cur_kv_len = 4
    backend._alloc_req_kv_mem(req, alloc_token_num=4)
    assert req.hold_kv_len == 8
    assert context.req_manager.mem_manager.alloc_sizes == [4, 4]
    assert context.req_manager.get_cpu_token_indexes(0, 0, 8).tolist() == list(range(8))


def test_reservation_fills_each_request_table_row(monkeypatch):
    _, backend = _make_context(monkeypatch)
    req0 = _make_req(0)
    req1 = _make_req(1)

    backend._alloc_req_kv_mem(req0, alloc_token_num=4)
    backend._alloc_req_kv_mem(req1, alloc_token_num=4)

    assert generic_pre_process.g_infer_context.req_manager.get_cpu_token_indexes(0, 0, 4).tolist() == [0, 1, 2, 3]
    assert generic_pre_process.g_infer_context.req_manager.get_cpu_token_indexes(1, 0, 4).tolist() == [4, 5, 6, 7]
    assert req0.hold_kv_len == req1.hold_kv_len == 4


def test_alloc_req_kv_mem_defers_gpu_table_update(monkeypatch):
    context, backend = _make_context(monkeypatch)
    req = _make_req(0)

    backend._alloc_req_kv_mem(req, alloc_token_num=4)
    backend._alloc_req_kv_mem(req, alloc_token_num=4)

    assert context.req_manager.get_cpu_token_indexes(0, 0, 8).tolist() == list(range(8))
    assert context.req_manager.req_to_token_indexs.eq(-1).all()
    assert req.token_index.gpu_index_submitted_len == 0


def test_need_token_num_distinguishes_compute_and_page_allocation():
    req = SimpleNamespace(
        args=SimpleNamespace(page_size=4),
        cur_kv_len=3,
        hold_kv_len=4,
        mtp_step=0,
        get_chuncked_input_token_len=lambda: 6,
        get_cur_total_len=lambda: 10,
    )
    req._kv_cache_alloc_need = lambda target_len: InferReq._kv_cache_alloc_need(req, target_len)

    assert InferReq.prefill_need_token_num(req, is_chuncked_prefill=True) == (3, 4)
    assert InferReq.prefill_need_token_num(req, is_chuncked_prefill=False) == (7, 8)
    assert InferReq.decode_need_token_num(req) == (1, 0)

    req.cur_kv_len = 4
    assert InferReq.decode_need_token_num(req) == (1, 4)

    req.hold_kv_len = 8
    assert InferReq.prefill_need_token_num(req, is_chuncked_prefill=True) == (2, 0)
    assert InferReq.prefill_need_token_num(req, is_chuncked_prefill=False) == (6, 4)

    req.mtp_step = 2
    assert InferReq.decode_need_token_num(req) == (6, 4)
    req.hold_kv_len = 12
    assert InferReq.decode_need_token_num(req) == (6, 0)


@pytest.mark.parametrize(
    "hold_kv_len, can_alloc_token_num, batch_max_tokens, expected_alloc_sizes, second_req_wait_pause",
    [
        (4, 8, 3, [4], False),
        (8, 0, 3, [], False),
        (4, 4, 6, [4], True),
    ],
)
def test_prefill_scheduler_checks_compute_and_kv_budgets_separately(
    monkeypatch, hold_kv_len, can_alloc_token_num, batch_max_tokens, expected_alloc_sizes, second_req_wait_pause
):
    context, backend = _make_context(monkeypatch)
    backend.args.enable_cpu_cache = False
    backend.args.enable_prefill_decode_mixed = False
    backend.support_overlap = False
    backend.disable_chunked_prefill = False
    backend.batch_max_tokens = batch_max_tokens
    backend.is_master_in_dp = True
    backend._timer_merge_radix_tree = lambda: None
    backend._reorder_pd_high_priority_reqs = lambda reqs: reqs
    backend._reorder_long_prefill_reqs = lambda reqs: reqs
    context.get_can_alloc_token_num = lambda: can_alloc_token_num
    context.cache_placement_controller = SimpleNamespace(set_req_cache_way=lambda reqs: None)
    context.filter_reqs = lambda finished_reqs: None
    context.pause_reqs = lambda reqs, is_master_in_dp: None

    reqs = [_make_req(0), _make_req(1)]
    for req in reqs:
        req.args = context.args
        req.cur_kv_len = 3
        req.hold_kv_len = hold_kv_len
        req.filter_mark = False
        req.wait_pause = False
        req.paused = False
        req.infer_aborted = False
        req.finish_status = infer_batch.FinishStatus()
        req.is_slave_req = lambda: False
        req.get_cur_total_len = lambda: 6
        req.get_chuncked_input_token_len = lambda: 6
        req.prefill_need_token_num = MethodType(InferReq.prefill_need_token_num, req)
        start_index = req.req_idx * hold_kv_len
        context.req_manager.append_token_indexes(
            req.req_idx, torch.arange(start_index, start_index + hold_kv_len, dtype=torch.int32)
        )
    context.req_manager.mem_manager.next_index = 2 * hold_kv_len
    backend._filter_not_ready_reqs = lambda req_ids: reqs
    prefill_reqs, decode_reqs = backend._get_classed_reqs(req_ids=[0, 1])

    assert prefill_reqs == [reqs[0]]
    assert decode_reqs == []
    assert context.req_manager.mem_manager.alloc_sizes == expected_alloc_sizes
    assert reqs[0].hold_kv_len == 8
    assert not reqs[0].wait_pause
    assert reqs[1].hold_kv_len == hold_kv_len
    assert reqs[1].wait_pause is second_req_wait_pause


def test_decode_scheduler_reserves_cpu_request_table(monkeypatch):
    context, backend = _make_context(monkeypatch)
    backend.args.enable_cpu_cache = False
    backend.args.enable_prefill_decode_mixed = False
    backend.args.run_mode = "normal"
    backend.support_overlap = False
    backend.batch_max_tokens = 8
    backend.is_master_in_dp = True
    backend._timer_merge_radix_tree = lambda: None
    backend._reorder_pd_high_priority_reqs = lambda reqs: reqs
    backend._reorder_long_prefill_reqs = lambda reqs: reqs
    context.get_can_alloc_token_num = lambda: 12
    context.cache_placement_controller = SimpleNamespace(set_req_cache_way=lambda reqs: None)
    context.filter_reqs = lambda finished_reqs: None
    context.pause_reqs = lambda reqs, is_master_in_dp: None

    req = _make_req(0)
    req.args = context.args
    req.cur_kv_len = 4
    req.hold_kv_len = 4
    req.mtp_step = 0
    req.filter_mark = False
    req.wait_pause = False
    req.paused = False
    req.infer_aborted = False
    req.finish_status = infer_batch.FinishStatus()
    req.get_cur_total_len = lambda: 5
    req.decode_need_token_num = MethodType(InferReq.decode_need_token_num, req)
    context.req_manager.append_token_indexes(0, torch.arange(4, dtype=torch.int32))
    context.req_manager.mem_manager.next_index = 4
    backend._filter_not_ready_reqs = lambda req_ids: [req]

    prefill_reqs, decode_reqs = backend._get_classed_reqs(req_ids=[0])

    assert prefill_reqs == []
    assert decode_reqs == [req]
    assert req.hold_kv_len == 16
    assert context.req_manager.mem_manager.alloc_sizes == [12]


def _make_capacity_scheduler(monkeypatch, page_size, available, run_mode, support_overlap=False):
    context, backend = _make_context(monkeypatch)
    context.args.page_size = context.req_manager.page_size = page_size
    context.req_manager.token_indexes_cpu = torch.full((2, 256), -1, dtype=torch.int32)
    context.req_manager.req_to_token_indexs = torch.full((2, 256), -1, dtype=torch.int32)
    backend.args.enable_cpu_cache = False
    backend.args.enable_prefill_decode_mixed = False
    backend.args.run_mode = run_mode
    backend.support_overlap = support_overlap
    backend.disable_chunked_prefill = True
    backend.batch_max_tokens = 256
    backend.is_master_in_dp = True
    backend.logger = SimpleNamespace(info=lambda message: None)
    backend._timer_merge_radix_tree = lambda: None
    backend._reorder_pd_high_priority_reqs = lambda reqs: reqs
    backend._reorder_long_prefill_reqs = lambda reqs: reqs
    context.get_can_alloc_token_num = lambda: available
    context.cache_placement_controller = SimpleNamespace(set_req_cache_way=lambda reqs: None)
    context.finished_reqs = []
    context.filter_reqs = lambda finished_reqs: context.finished_reqs.extend(finished_reqs)
    context.pause_reqs = lambda reqs, is_master_in_dp: None
    return context, backend


def _make_capacity_req(req_idx, context, is_decode=True):
    req = _make_req(req_idx)
    req.req_id = req_idx
    req.args = context.args
    req.cur_kv_len = req.hold_kv_len = context.args.page_size
    req.mtp_step = 0
    req.filter_mark = req.wait_pause = req.paused = req.infer_aborted = False
    req.finish_status = infer_batch.FinishStatus()
    req.cpu_cache_task_status = SimpleNamespace(is_not_started=lambda: True)
    req.is_slave_req = lambda: False
    req.get_cur_total_len = lambda: req.cur_kv_len + (1 if is_decode else 2 * context.args.page_size)
    req.decode_need_token_num = MethodType(InferReq.decode_need_token_num, req)
    req.prefill_need_token_num = MethodType(InferReq.prefill_need_token_num, req)
    context.req_manager.append_token_indexes(req_idx, torch.arange(req.hold_kv_len, dtype=torch.int32))
    return req


@pytest.mark.parametrize("page_size", [1, 3, 4, 7, 8, 64])
@pytest.mark.parametrize("run_mode", ["normal", "decode"])
@pytest.mark.parametrize("extra_budget", [0, 1, 8])
@pytest.mark.parametrize("request_types", [(True,), (True, True), (True, False), (False, True)])
def test_decode_headroom_only_uses_budget_after_all_required_allocations(
    monkeypatch, page_size, run_mode, extra_budget, request_types
):
    required = sum(page_size if is_decode else 2 * page_size for is_decode in request_types)
    context, backend = _make_capacity_scheduler(monkeypatch, page_size, required + extra_budget, run_mode)
    reqs = [_make_capacity_req(index, context, is_decode) for index, is_decode in enumerate(request_types)]
    context.req_manager.mem_manager.next_index = len(reqs) * page_size
    backend._filter_not_ready_reqs = lambda req_ids: reqs

    prefill, decode = backend._get_classed_reqs(req_ids=list(range(len(reqs))))

    assert prefill == [req for req, is_decode in zip(reqs, request_types) if not is_decode]
    assert decode == [req for req, is_decode in zip(reqs, request_types) if is_decode]
    assert context.finished_reqs == []
    remaining = extra_budget
    for req, is_decode in zip(reqs, request_types):
        extra = min(8, remaining) // page_size * page_size if is_decode and page_size < 8 else 0
        remaining -= extra
        assert req.hold_kv_len == page_size + (page_size if is_decode else 2 * page_size) + extra
        assert req.token_index.token_index_len == req.hold_kv_len
        assert req.token_index.gpu_index_submitted_len == req.hold_kv_len
        assert not req.wait_pause and not req.filter_mark
        assert not hasattr(req, "finished_by_pd_decode_capacity")
    # 必需量和可选预留合并申请，不额外增加分配次数，且不会超出总预算。
    allocations = context.req_manager.mem_manager.alloc_sizes
    assert len(allocations) == len(reqs)
    assert sum(allocations) == required + extra_budget - remaining


@pytest.mark.parametrize("page_size", [1, 4, 64])
@pytest.mark.parametrize("run_mode", ["normal", "decode"])
@pytest.mark.parametrize("support_overlap", [False, True])
def test_decode_still_rejects_requests_when_required_capacity_is_insufficient(
    monkeypatch, page_size, run_mode, support_overlap
):
    context, backend = _make_capacity_scheduler(monkeypatch, page_size, page_size - 1, run_mode, support_overlap)
    req = _make_capacity_req(0, context)
    backend._filter_not_ready_reqs = lambda req_ids: [req]

    assert backend._get_classed_reqs(req_ids=[0]) == ([], [])

    assert context.req_manager.mem_manager.alloc_sizes == []
    assert req.hold_kv_len == page_size
    if run_mode == "normal":
        assert req.wait_pause
        assert context.finished_reqs == []
    else:
        assert req.finished_by_pd_decode_capacity
        assert req.filter_mark is support_overlap
        assert context.finished_reqs == ([] if support_overlap else [req])


@pytest.mark.parametrize(
    "page_size, expected_alloc_token_num",
    [(1, 9), (3, 9), (4, 12), (7, 14), (8, 8)],
)
def test_decode_small_page_preallocation_reuses_one_allocation_for_multiple_steps(
    monkeypatch, page_size, expected_alloc_token_num
):
    context, backend = _make_context(monkeypatch)
    context.args.page_size = page_size
    context.req_manager.page_size = page_size
    context.req_manager.token_indexes_cpu = torch.full((2, (64 + page_size - 1) // page_size), -1, dtype=torch.int32)
    context.req_manager.req_to_token_indexs = torch.full((2, 64), -1, dtype=torch.int32)
    backend.args.enable_cpu_cache = False
    backend.args.enable_prefill_decode_mixed = False
    backend.args.run_mode = "normal"
    backend.support_overlap = False
    backend.batch_max_tokens = 8
    backend.is_master_in_dp = True
    backend._timer_merge_radix_tree = lambda: None
    backend._reorder_pd_high_priority_reqs = lambda reqs: reqs
    backend._reorder_long_prefill_reqs = lambda reqs: reqs
    context.get_can_alloc_token_num = lambda: 32
    context.cache_placement_controller = SimpleNamespace(set_req_cache_way=lambda reqs: None)
    context.filter_reqs = lambda finished_reqs: None
    context.pause_reqs = lambda reqs, is_master_in_dp: None

    req = _make_req(0)
    req.args = context.args
    req.cur_kv_len = page_size
    req.hold_kv_len = page_size
    req.mtp_step = 0
    req.filter_mark = False
    req.wait_pause = False
    req.paused = False
    req.infer_aborted = False
    req.finish_status = infer_batch.FinishStatus()
    req.get_cur_total_len = lambda: req.cur_kv_len + 1
    req.decode_need_token_num = MethodType(InferReq.decode_need_token_num, req)
    context.req_manager.append_token_indexes(0, torch.arange(page_size, dtype=torch.int32))
    context.req_manager.mem_manager.next_index = page_size
    backend._filter_not_ready_reqs = lambda req_ids: [req]

    for _ in range(expected_alloc_token_num):
        _, decode_reqs = backend._get_classed_reqs(req_ids=[0])
        assert decode_reqs == [req]
        req.cur_kv_len += 1

    assert req.hold_kv_len == page_size + expected_alloc_token_num
    assert context.req_manager.mem_manager.alloc_sizes == [expected_alloc_token_num]


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

    context.req_manager.append_token_indexes(0, torch.arange(4, dtype=torch.int32))
    context.req_manager.mem_manager.next_index = 4
    token_num, alloc_token_num = InferReq.decode_need_token_num(req)
    assert token_num == 6
    assert alloc_token_num == 8
    backend._alloc_req_kv_mem(req, alloc_token_num)

    model_input, run_reqs = generic_pre_process.prepare_decode_inputs([req])

    assert run_reqs == [req, req, req]
    assert model_input.b_seq_len.tolist() == [4, 5, 6]
    assert req.hold_kv_len == 12
    assert context.req_manager.mem_manager.alloc_sizes == [8]
    assert context.req_manager.get_cpu_token_indexes(0, 0, 12).tolist() == list(range(12))


def test_page_size_one_uses_the_same_scheduler_preallocation(monkeypatch):
    context, backend = _make_context(monkeypatch)
    context.args.page_size = 1
    context.req_manager.page_size = 1
    context.req_manager.token_indexes_cpu = torch.full((2, 16), -1, dtype=torch.int32)
    req = _make_req(0)
    req.cur_kv_len = 3
    req.hold_kv_len = 3
    req.mtp_step = 2
    req.multimodal_params = {"images": [], "audios": []}
    req.shared_kv_node = None
    req.get_cur_total_len = lambda: 4
    req.get_radix_cache_shared_len = lambda: 0
    req.args = context.args
    context.req_manager.append_token_indexes(0, torch.arange(3, dtype=torch.int32))
    context.req_manager.mem_manager.next_index = 3
    token_num, alloc_token_num = InferReq.decode_need_token_num(req)
    assert token_num == alloc_token_num == 6
    backend._alloc_req_kv_mem(req, alloc_token_num)

    model_input, _ = generic_pre_process.prepare_decode_inputs([req])

    assert req.hold_kv_len == 9
    assert context.req_manager.mem_manager.alloc_sizes == [6]
    assert context.req_manager.get_cpu_token_indexes(0, 0, 9).tolist() == list(range(9))


def test_page_size_one_frees_all_preallocated_indexes():
    infer_context = InferenceContext.__new__(InferenceContext)
    infer_context.args = SimpleNamespace(page_size=1)
    infer_context.radix_cache = None
    infer_context.req_manager = make_cpu_index_manager(
        token_table=torch.arange(16, dtype=torch.int32)[None, :], page_size=1
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=3,
        hold_kv_len=12,
        shm_req=SimpleNamespace(shm_cur_kv_len=3),
    )
    bind_cpu_indexes(req, infer_context.req_manager)
    free_token_indexes = []

    infer_context.free_a_req_mem(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == list(range(12))
    assert req.cur_kv_len == req.hold_kv_len == 0
    assert req.token_index.token_index_len == req.token_index.gpu_index_submitted_len == 0


@pytest.mark.parametrize("attention", ["full", "hybrid_small", "hybrid_big"])
def test_cache_release_preserves_snapshots_and_reclaims_tail_duplicates_and_reference(monkeypatch, attention):
    hybrid = attention != "full"
    context = InferenceContext()
    context.args = SimpleNamespace(page_size=4, linear_att_hash_page_size=4, linear_att_page_block_num=2)
    context.is_hybrid_att_model = hybrid
    context.req_manager = make_cpu_index_manager(torch.arange(16, dtype=torch.int32)[None, :], page_size=4)
    cached = []
    released_refs = []

    def insert(key, indexes, *, block_hashs=None, block_state_idxs=None, len_to_big_page_id=None):
        cached.append((key, indexes, block_hashs, block_state_idxs))
        # 模拟 radix cache 接管请求已生成的大页 checkpoint。
        if hybrid:
            len_to_big_page_id.clear()
        return 8, None

    context.radix_cache = SimpleNamespace(insert=insert, dec_node_ref_counter=released_refs.append)
    shared_node = SimpleNamespace(node_prefix_total_len=4)
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=13 if hybrid else 11,
        hold_kv_len=16,
        cache_tiers=(CacheTier.GPU,),
        shared_kv_node=shared_node,
        hybrid_cache_len=12,
        tail_small_page_buffer_id=7 if attention == "hybrid_small" else None,
        hybrid_len_to_big_page_id={8: 9} if hybrid else {},
        get_input_token_ids=lambda: list(range(13)),
        shm_req=SimpleNamespace(
            shm_cur_kv_len=13,
            hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [100, 200, 300]),
        ),
    )
    bind_cpu_indexes(req, context.req_manager)
    monkeypatch.setattr(infer_batch, "g_infer_context", context)
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: context.args)
    released = []

    context.free_a_req_mem(released, req)

    cache_len = 12 if attention == "hybrid_small" else 8
    assert [indexes.tolist() for indexes in released] == [list(range(cache_len, 16)), list(range(4, 8))]
    assert released_refs == [shared_node]
    assert req.shared_kv_node is None
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == 0
    assert req.token_index.token_index_len == req.token_index.gpu_index_submitted_len == 0
    assert req.tail_small_page_buffer_id is None and req.hybrid_len_to_big_page_id == {}
    context.req_manager.token_indexes_cpu.fill_(-99)
    key, indexes, block_hashs, block_state_idxs = cached[0]
    assert key.tolist() == indexes.tolist() == list(range(cache_len))
    if hybrid:
        assert block_hashs == [100, 200, 300][: cache_len // 4]
        assert block_state_idxs == ([None, None, 7] if attention == "hybrid_small" else [None, None])
    else:
        assert block_hashs is None and block_state_idxs is None


def _insert_empty_hybrid_prefix(key, indexes, *, block_hashs, block_state_idxs, len_to_big_page_id):
    assert key.numel() == indexes.numel() == 0
    assert block_hashs == block_state_idxs == []
    assert not len_to_big_page_id
    return 0, None


def test_hybrid_release_without_new_big_page_rejects_pending_checkpoints(monkeypatch):
    context = InferenceContext()
    context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    context.is_hybrid_att_model = True
    context.req_manager = make_cpu_index_manager(torch.arange(8, dtype=torch.int32)[None, :])
    freed_states = []
    context.radix_cache = SimpleNamespace(
        insert=_insert_empty_hybrid_prefix,
        big_page_buffers=SimpleNamespace(free_state_cache=freed_states.extend),
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=5,
        hold_kv_len=8,
        hybrid_cache_len=0,
        tail_small_page_buffer_id=None,
        hybrid_len_to_big_page_id={8: 9},
        shared_kv_node=None,
        get_input_token_ids=lambda: list(range(8)),
        shm_req=SimpleNamespace(hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [])),
    )
    bind_cpu_indexes(req, context.req_manager)
    monkeypatch.setattr(infer_batch, "g_infer_context", context)
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: context.args)
    released = []

    with pytest.raises(AssertionError):
        context._hybrid_att_free_req(released, req)

    assert released == []
    assert freed_states == []
    assert req.hybrid_len_to_big_page_id == {8: 9}
    assert req.cur_kv_len == 5 and req.hold_kv_len == 8


def test_linear_attention_frees_reserved_page_tail(monkeypatch):
    infer_context = InferenceContext.__new__(InferenceContext)
    infer_context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    infer_context.radix_cache = SimpleNamespace(insert=_insert_empty_hybrid_prefix)
    infer_context.req_manager = make_cpu_index_manager(
        token_table=torch.arange(16, dtype=torch.int32)[None, :], page_size=1
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=5,
        hold_kv_len=8,
        hybrid_cache_len=0,
        tail_small_page_buffer_id=None,
        hybrid_len_to_big_page_id={},
        shared_kv_node=None,
        get_input_token_ids=lambda: list(range(8)),
        shm_req=SimpleNamespace(hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [])),
    )
    bind_cpu_indexes(req, infer_context.req_manager)
    free_token_indexes = []
    monkeypatch.setattr(infer_batch.g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: infer_context.args)

    infer_context._hybrid_att_free_req(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == list(range(8))


def test_linear_attention_frees_preallocated_pages_before_first_forward(monkeypatch):
    infer_context = InferenceContext.__new__(InferenceContext)
    infer_context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    infer_context.radix_cache = SimpleNamespace(insert=_insert_empty_hybrid_prefix)
    infer_context.req_manager = make_cpu_index_manager(
        token_table=torch.arange(8, dtype=torch.int32)[None, :], page_size=1
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=0,
        hold_kv_len=8,
        hybrid_cache_len=0,
        tail_small_page_buffer_id=None,
        hybrid_len_to_big_page_id={},
        shared_kv_node=None,
        get_input_token_ids=lambda: list(range(8)),
        shm_req=SimpleNamespace(hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [])),
    )
    bind_cpu_indexes(req, infer_context.req_manager)
    free_token_indexes = []
    monkeypatch.setattr(infer_batch.g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: infer_context.args)

    infer_context._hybrid_att_free_req(free_token_indexes, req)

    assert free_token_indexes[0].tolist() == list(range(8))


def _make_paused_req(req_idx, target_kv_len, page_size, match_kv_len=0, chunk_kv_len=None):
    if chunk_kv_len is None:
        chunk_kv_len = target_kv_len
    req = SimpleNamespace(
        args=SimpleNamespace(page_size=page_size),
        req_id=req_idx,
        req_idx=req_idx,
        cur_kv_len=0,
        hold_kv_len=0,
        paused=True,
        match_call_count=0,
        shared_kv_node=None,
        shm_req=SimpleNamespace(is_paused=True, shm_cur_kv_len=0),
        get_cur_total_len=lambda: target_kv_len,
        get_chuncked_input_token_len=lambda: chunk_kv_len,
    )

    def match_radix_cache():
        req.match_call_count += 1
        req.cur_kv_len = match_kv_len
        req.hold_kv_len = match_kv_len
        req.shm_req.shm_cur_kv_len = match_kv_len

    req._match_radix_cache = match_radix_cache
    req._kv_cache_alloc_need = lambda target_len: InferReq._kv_cache_alloc_need(req, target_len)
    req.prefill_need_token_num = MethodType(InferReq.prefill_need_token_num, req)
    return req


def test_recover_paused_reqs_uses_page_allocation_need(monkeypatch):
    infer_context = InferenceContext.__new__(InferenceContext)
    infer_context.args = SimpleNamespace(page_size=4)
    infer_context.backend = SimpleNamespace(disable_chunked_prefill=False)
    infer_context.radix_cache = None
    freed_indexes = []
    infer_context.req_manager = make_cpu_index_manager(
        token_table=torch.arange(16, dtype=torch.int32).reshape(2, 8),
        free_token=lambda indexes: freed_indexes.extend(indexes.tolist()),
        page_size=1,
    )
    infer_context.get_can_alloc_token_num = lambda: 5
    large_req = _make_paused_req(req_idx=0, target_kv_len=5, page_size=4)
    small_req = _make_paused_req(req_idx=1, target_kv_len=4, page_size=4)
    monkeypatch.setattr(infer_batch.g_infer_context, "is_hybrid_att_model", False)

    infer_context.recover_paused_reqs([large_req, small_req], is_master_in_dp=True)

    assert large_req.paused is True
    assert large_req.shm_req.is_paused is True
    assert large_req.match_call_count == 0
    assert small_req.paused is True
    assert small_req.shm_req.is_paused is True
    assert small_req.match_call_count == 0
    assert freed_indexes == []


def test_recover_paused_reqs_requires_capacity_for_the_full_sequence(monkeypatch):
    infer_context = InferenceContext.__new__(InferenceContext)
    infer_context.args = SimpleNamespace(page_size=4)
    infer_context.radix_cache = None
    infer_context.req_manager = make_cpu_index_manager(
        token_table=torch.arange(8, dtype=torch.int32)[None, :], free_token=lambda indexes: None, page_size=1
    )
    infer_context.get_can_alloc_token_num = lambda: 4
    req = _make_paused_req(req_idx=0, target_kv_len=8, page_size=4, chunk_kv_len=4)
    monkeypatch.setattr(infer_batch.g_infer_context, "is_hybrid_att_model", False)

    infer_context.recover_paused_reqs([req], is_master_in_dp=True)

    assert req.paused is True
    assert req.shm_req.is_paused is True
    assert req.match_call_count == 0


def test_recover_paused_reqs_accounts_for_rematched_prefix(monkeypatch):
    infer_context = InferenceContext.__new__(InferenceContext)
    infer_context.args = SimpleNamespace(page_size=4)
    infer_context.backend = SimpleNamespace(disable_chunked_prefill=False)
    infer_context.get_can_alloc_token_num = lambda: 12
    req = _make_paused_req(req_idx=0, target_kv_len=9, page_size=4, match_kv_len=8)
    monkeypatch.setattr(infer_batch.g_infer_context, "is_hybrid_att_model", False)

    infer_context.recover_paused_reqs([req], is_master_in_dp=True)

    assert req.cur_kv_len == req.hold_kv_len == 8
    assert req.paused is False
    assert req.shm_req.is_paused is False
    assert req.match_call_count == 1
