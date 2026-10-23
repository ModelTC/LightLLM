from types import SimpleNamespace
from contextlib import nullcontext
import builtins

import pytest
import torch

# Use the production import order to avoid the existing package initialization cycle.
from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.req_manager.base import ReqManager, _ReqLinkedList
from lightllm.common.cuda_wrapper import cuda_memcpy_pinned_h2d_batch_async
from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext
from lightllm.server.router.model_infer.infer_req_ext1 import create_req_token_index
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from unit_tests.common.cpu_index_helpers import bind_cpu_indexes, make_cpu_index_manager


@pytest.fixture(autouse=True)
def reset_batch_copy_availability(monkeypatch):
    from lightllm.common import cuda_wrapper

    monkeypatch.setattr(cuda_wrapper.PinnedH2DBatchCopier, "_native_batch_available", True)
    monkeypatch.setattr(infer_batch.g_infer_context, "req_manager", None)


def _make_manager(device="cpu"):
    manager = make_cpu_index_manager(
        torch.full((3, 16), -1, dtype=torch.int32, pin_memory=torch.device(device).type == "cuda"),
        req_to_token_indexs=torch.full((3, 16), -1, dtype=torch.int32, device=device),
    )
    infer_batch.g_infer_context.req_manager = manager
    manager.max_request_num = 2
    manager.req_list = _ReqLinkedList(2)
    return manager


def test_req_index_allocation_preserves_order_capacity_and_reuse():
    indexes = _ReqLinkedList(3)
    assert indexes.is_all_free() and indexes.can_alloc_size == 3
    assert [indexes.alloc() for _ in range(3)] == [0, 1, 2]
    assert indexes.can_alloc_size == 0 and not indexes.is_all_free()
    assert indexes.alloc() is None

    indexes.free(1)
    indexes.free(0)
    assert indexes.can_alloc_size == 2
    assert [indexes.alloc(), indexes.alloc()] == [0, 1]
    assert indexes.alloc() is None
    for index in range(3):
        indexes.free(index)
    assert indexes.is_all_free() and indexes.can_alloc_size == 3


def test_req_index_free_rejects_unallocated_and_duplicate_indexes():
    indexes = _ReqLinkedList(2)
    with pytest.raises(AssertionError):
        indexes.free(0)
    assert indexes.alloc() == 0
    indexes.free(0)
    with pytest.raises(AssertionError):
        indexes.free(0)
    assert indexes.can_alloc_size == 2
    assert [indexes.alloc(), indexes.alloc()] == [0, 1]
    assert indexes.alloc() is None


def test_cpu_updates_own_allocator_values_and_defer_gpu_publication():
    manager = _make_manager()
    indexes = torch.tensor([20, 21, 22, 23], dtype=torch.int32)
    manager.token_indexes_cpu[0, 4:8].copy_(indexes)
    indexes.fill_(99)
    manager.token_indexes_cpu[0, :4].copy_(torch.arange(4, dtype=torch.int32))

    assert manager.token_indexes_cpu[0, :8].tolist() == [0, 1, 2, 3, 20, 21, 22, 23]
    assert manager.req_to_token_indexs.eq(-1).all()


def test_releasing_request_memory_resets_uploaded_length():
    manager = _make_manager()
    manager.token_indexes_cpu[0, :8].copy_(torch.arange(8, dtype=torch.int32))
    context = InferenceContext()
    context.req_manager = manager
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=6,
        hold_kv_len=8,
        shm_req=SimpleNamespace(shm_cur_kv_len=6),
    )
    bind_cpu_indexes(req, manager, gpu_index_submitted_len=8)
    released = []
    context.free_a_req_mem(released, req)

    assert released[0].tolist() == list(range(8))
    assert req.cur_kv_len == req.hold_kv_len == req.token_index.gpu_index_submitted_len == 0
    assert req.token_index.token_index_len == 0
    assert req.token_index._cpu_buffer is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_export_keeps_cpu_mirror_local_and_pinned(monkeypatch):
    from lightllm.common.kv_cache_mem_manager import mem_manager as impl

    req_manager = _make_manager("cuda")
    mem_manager = impl.MemoryManager.__new__(impl.MemoryManager)
    source = req_manager.token_indexes_cpu
    source_ptr = source.data_ptr()
    exports = []

    def serialize(manager):
        assert manager.req_to_token_indexs is req_manager.req_to_token_indexs
        assert not hasattr(manager, "token_indexes_cpu")
        exports.append(manager)
        return memoryview(b"test")

    def forbid_cpu_export(*args, **kwargs):
        pytest.fail("CPU mirror export and host registration must not run")

    monkeypatch.setattr(torch.Tensor, "share_memory_", forbid_cpu_export)
    monkeypatch.setattr(torch.cuda, "cudart", forbid_cpu_export)
    monkeypatch.setattr(impl, "ForkingPickler", SimpleNamespace(dumps=serialize))
    monkeypatch.setattr(impl, "kv_trans_use_p2p", lambda: False)
    monkeypatch.setattr(impl, "FileLock", lambda path: nullcontext())
    monkeypatch.setattr(impl, "get_node_world_size", lambda: 1)
    monkeypatch.setattr(impl, "get_current_rank_in_node", lambda: 0)
    monkeypatch.setattr(impl, "get_service_shm_name", lambda name: name)
    monkeypatch.setattr(
        impl,
        "create_or_link_shm",
        lambda name, expected_size, **kwargs: SimpleNamespace(
            name=name, size=expected_size, buf=bytearray(expected_size)
        ),
    )

    mem_manager.write_to_shm(req_manager)
    mem_manager.write_to_shm(req_manager)

    assert len(exports) == 4
    assert req_manager.token_indexes_cpu is source
    assert source.data_ptr() == source_ptr
    assert source.is_pinned() and not source.is_shared()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["pin_mem_cache", "pure_gpu"])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("resuming", [False, True])
def test_prefix_match_appends_to_initialized_or_released_index_state(monkeypatch, mode, hybrid, resuming):
    manager = _make_manager("cuda")
    req = InferReq.__new__(InferReq)
    req.req_idx = 0
    req.req_manager = manager
    req.args = SimpleNamespace(
        page_size=4,
        req_token_index_mode=mode,
        linear_att_hash_page_size=2,
        linear_att_page_block_num=2,
        max_req_total_len=16,
    )
    req.cur_kv_len = req.hold_kv_len = 0
    req.shared_kv_node = req.tail_small_page_buffer_id = None
    req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
    req.shm_req = SimpleNamespace(
        input_len=8,
        shm_prompt_ids=SimpleNamespace(arr=list(range(8))),
        hybrid_token_hash_list=SimpleNamespace(size=3, get_all=lambda: [1, 2, 3]),
    )
    req.token_index = create_req_token_index(req, index_stream=torch.cuda.current_stream())
    req.get_cur_total_len = lambda: 8
    if resuming:
        req.token_index.append_token_indexes(torch.arange(8, dtype=torch.int32))
        req.token_index.get_gpu_token_indexes(0, 8)
        req.cur_kv_len, req.hold_kv_len = 6, 8
        release_context = InferenceContext()
        released = []
        release_context.free_a_req_mem(released, req)
        assert released[0].cpu().tolist() == list(range(8))
        assert req.cur_kv_len == req.hold_kv_len == 0
        assert req.token_index.token_index_len == req.token_index.gpu_index_submitted_len == 0

    node = SimpleNamespace(node_prefix_total_len=4, big_page_buffer_idx=0, is_big_page_node=lambda: True)
    indexes = torch.arange(32, 36, dtype=torch.int32)
    manager.restore_big_page_state = lambda **kwargs: None
    context = SimpleNamespace(
        is_hybrid_att_model=hybrid,
        req_manager=manager,
        radix_cache=SimpleNamespace(match_prefix=lambda key, **kwargs: (node, 4, indexes)),
    )
    monkeypatch.setattr(infer_batch, "g_infer_context", context)

    def unexpected_reset():
        pytest.fail("Prefix matching starts with empty indexes and must not reset them again")

    monkeypatch.setattr(req.token_index, "reset", unexpected_reset)
    if hybrid:
        req._hybrid_match_radix_cache()
    else:
        req._match_radix_cache()

    assert req.token_index.token_index_len == 4
    assert req.token_index.gpu_index_submitted_len == (0 if mode == "pin_mem_cache" else 4)
    assert req.cur_kv_len == req.hold_kv_len == 4
    assert req.shared_kv_node is node
    assert req.shm_req.prompt_cache_len == req.shm_req.shm_cur_kv_len == 4
    assert req.token_index.get_cpu_token_indexes(0, 4).tolist() == [32, 33, 34, 35]
    assert req.token_index.get_gpu_token_indexes(0, 4).cpu().tolist() == [32, 33, 34, 35]


def test_radix_insertion_owns_cpu_indexes_before_request_row_reuse():
    manager = _make_manager()
    manager.token_indexes_cpu[0, :8].copy_(torch.arange(8, dtype=torch.int32))
    cached = []

    def insert(key, indexes):
        cached.append(indexes)
        return 0, None

    context = InferenceContext()
    context.args = SimpleNamespace(page_size=4)
    context.req_manager = manager
    context.radix_cache = SimpleNamespace(insert=insert)
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=6,
        hold_kv_len=8,
        shared_kv_node=None,
        get_input_token_ids=lambda: list(range(6)),
    )
    bind_cpu_indexes(req, manager)
    released = []
    context._full_att_free_req(released, req)
    assert released[0].tolist() == [4, 5, 6, 7]
    manager.token_indexes_cpu[0, :8].fill_(99)

    assert cached[0].tolist() == [0, 1, 2, 3]
    assert manager.req_to_token_indexs.eq(-1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_batch_reads_source_after_prior_stream_writes_and_preserves_untouched_ranges():
    manager = _make_manager("cuda")
    incoming_indexes = torch.tensor([40, 41, 42, 43], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        manager.token_indexes_cpu[0, 4:8].copy_(incoming_indexes, non_blocking=True)
        cuda_memcpy_pinned_h2d_batch_async(
            [manager.req_to_token_indexs[0, 4:8]],
            [manager.token_indexes_cpu[0, 4:8]],
        )
    stream.synchronize()

    assert manager.req_to_token_indexs[0, 4:8].cpu().tolist() == [40, 41, 42, 43]
    assert manager.req_to_token_indexs[0, :4].eq(-1).all()
    assert manager.req_to_token_indexs[0, 8:].eq(-1).all()
    assert manager.req_to_token_indexs[1:].eq(-1).all()
    manager.token_indexes_cpu[0, 4:8].copy_(torch.tensor([80, 81, 82, 83], dtype=torch.int32))
    with torch.cuda.stream(stream):
        cuda_memcpy_pinned_h2d_batch_async(
            [manager.req_to_token_indexs[0, 4:8]],
            [manager.token_indexes_cpu[0, 4:8]],
        )
    stream.synchronize()
    assert manager.req_to_token_indexs[0, 4:8].cpu().tolist() == [80, 81, 82, 83]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("default_stream", [False, True])
def test_native_batch_uses_stream_order_and_handles_disjoint_ranges(monkeypatch, default_stream):
    from cuda.bindings import runtime as cuda_runtime

    manager = _make_manager("cuda")
    manager.token_indexes_cpu.copy_(torch.arange(48, dtype=torch.int32).reshape(3, 16))
    ranges = [(0, 4, 8), (1, 1, 5), (0, 12, 16)]
    expected = torch.full_like(manager.token_indexes_cpu, -1)
    for row, start, end in ranges:
        expected[row, start:end].copy_(manager.token_indexes_cpu[row, start:end])
    calls = []
    native_copy = cuda_runtime.cudaMemcpyBatchAsync

    def record_native(*args):
        calls.append((args[3], args[2], args[4][0].srcAccessOrder))
        return native_copy(*args)

    monkeypatch.setattr(cuda_runtime, "cudaMemcpyBatchAsync", record_native)
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        cuda_memcpy_pinned_h2d_batch_async(
            [manager.req_to_token_indexs[row, start:end] for row, start, end in ranges],
            [manager.token_indexes_cpu[row, start:end] for row, start, end in ranges],
        )
        assert torch.cuda.current_stream() == stream
        result = manager.req_to_token_indexs.clone()
    stream.synchronize()

    assert calls == [(3, [16, 16, 16], cuda_runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream)]
    assert torch.equal(result.cpu(), expected)


def test_empty_batch_does_not_call_cuda(monkeypatch):
    original_import = builtins.__import__

    def check_import(name, *args, **kwargs):
        if name == "cuda.bindings":
            pytest.fail("empty batch must not import CUDA bindings")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    cuda_memcpy_pinned_h2d_batch_async([], [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("failure", ["missing_binding", "missing_api", "native_error", "exception"])
def test_batch_fallback_uses_current_stream_without_synchronizing(monkeypatch, failure):
    from cuda.bindings import runtime as cuda_runtime
    from lightllm.common import cuda_wrapper as base

    manager = _make_manager("cuda")
    manager.token_indexes_cpu.copy_(torch.arange(48, dtype=torch.int32).reshape(3, 16))
    ranges = [(0, 4, 8), (1, 1, 5), (0, 12, 16)]
    expected = torch.full_like(manager.token_indexes_cpu, -1)
    for row, start, end in ranges:
        expected[row, start:end].copy_(manager.token_indexes_cpu[row, start:end])

    batch_attempts = []
    if failure == "missing_binding":
        original_import = builtins.__import__

        def missing_import(name, *args, **kwargs):
            if name == "cuda.bindings":
                batch_attempts.append(name)
                raise ModuleNotFoundError("cuda.bindings is unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", missing_import)
    elif failure == "missing_api":
        monkeypatch.delattr(cuda_runtime, "cudaMemcpyBatchAsync")
    else:

        def failed_batch(*args):
            batch_attempts.append(args)
            if failure == "exception":
                raise RuntimeError("batch API could not be loaded")
            return (cuda_runtime.cudaError_t.cudaErrorNotSupported,)

        monkeypatch.setattr(cuda_runtime, "cudaMemcpyBatchAsync", failed_batch)

    loop_calls = []
    torch_copy = torch.Tensor.copy_

    def record_copy(dst, src, *args, **kwargs):
        if dst.is_cuda and src.device.type == "cpu":
            assert kwargs["non_blocking"] is True
            loop_calls.append((dst.numel() * dst.element_size(), torch.cuda.current_stream(dst.device).cuda_stream))
        return torch_copy(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    dsts = [manager.req_to_token_indexs[row, start:end] for row, start, end in ranges]
    srcs = [manager.token_indexes_cpu[row, start:end] for row, start, end in ranges]

    def unexpected_sync(*args, **kwargs):
        pytest.fail("copy submission must not synchronize")

    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        manager.req_to_token_indexs.fill_(-1)
        with monkeypatch.context() as no_sync:
            no_sync.setattr(torch.cuda.Stream, "synchronize", unexpected_sync)
            no_sync.setattr(torch.cuda, "synchronize", unexpected_sync)
            cuda_memcpy_pinned_h2d_batch_async(dsts, srcs)
        result = manager.req_to_token_indexs.clone()
    stream.synchronize()

    assert torch.equal(result.cpu(), expected)
    assert [size for size, _ in loop_calls] == [16] * 3
    assert all(copy_stream == stream.cuda_stream for _, copy_stream in loop_calls)
    assert not base.PinnedH2DBatchCopier._native_batch_available
    attempts_after_failure = len(batch_attempts)
    manager.token_indexes_cpu.fill_(99)
    with torch.cuda.stream(stream):
        cuda_memcpy_pinned_h2d_batch_async(dsts, srcs)
    stream.synchronize()
    assert len(batch_attempts) == attempts_after_failure
    assert len(loop_calls) == 6
    assert manager.req_to_token_indexs[0, 4:8].cpu().tolist() == [99] * 4


@pytest.mark.parametrize("entry", ["forward", "microbatch_overlap_prefill", "microbatch_overlap_decode"])
def test_model_entries_do_not_manage_request_upload_progress(entry):
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    events = []
    model.req_manager = SimpleNamespace()
    model.args = SimpleNamespace(enable_prefill_decode_mixed=False)
    model._prefill = lambda model_input: events.append("infer")
    model._decode = lambda model_input: events.append("infer")
    model._microbatch_overlap_prefill_cuda = lambda *inputs: events.append("infer")
    model._microbatch_overlap_decode_cuda = lambda *inputs: events.append("infer")
    model_input = SimpleNamespace(
        is_prefill=True,
        input_ids=torch.ones(1, dtype=torch.int64),
        to_cuda=lambda: events.append("input"),
    )

    if entry == "forward":
        model.forward(model_input)
        assert events == ["input", "infer"]
    else:
        getattr(model, entry)(model_input, model_input)
        assert events == ["input", "input", "infer"]


def _make_scheduler(monkeypatch, manager, reqs, budget=16, mixed=False):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    args = SimpleNamespace(page_size=4, enable_cpu_cache=False, enable_prefill_decode_mixed=mixed, run_mode="normal")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    context = SimpleNamespace(
        req_manager=manager,
        radix_cache=None,
        get_can_alloc_token_num=lambda: 32,
        get_overlap_stream=lambda: stream,
        cache_placement_controller=SimpleNamespace(set_req_cache_way=lambda reqs: None),
        filter_reqs=lambda finished_reqs: None,
        pause_reqs=lambda reqs, is_master_in_dp: None,
    )
    monkeypatch.setattr(base_backend, "g_infer_context", context)
    manager.mem_manager = SimpleNamespace(
        alloc=lambda size: torch.arange(100, 100 + size, dtype=torch.int32, pin_memory=True)
    )
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = args
    backend.support_overlap = False
    backend.disable_chunked_prefill = False
    backend.batch_max_tokens = budget
    backend.is_master_in_dp = True
    backend._timer_merge_radix_tree = lambda: None
    backend._filter_not_ready_reqs = lambda req_ids: reqs
    backend._reorder_pd_high_priority_reqs = lambda reqs: reqs
    backend._reorder_long_prefill_reqs = lambda reqs: reqs
    for req in reqs:
        req.args = args
    copies = []
    copy_indexes = backend._copy_req_token_indexes_to_gpu

    def record_copy(run_reqs):
        ranges = [
            (req.req_idx, req.token_index.gpu_index_submitted_len, req.hold_kv_len)
            for req in run_reqs
            if req.token_index.gpu_index_submitted_len < req.hold_kv_len
        ]
        if ranges:
            copies.append(ranges)
        copy_indexes(run_reqs)

    backend._copy_req_token_indexes_to_gpu = record_copy
    return backend, stream, copies


def _make_scheduled_req(row, total_len=12, gpu_index_submitted_len=4, hold_kv_len=4):
    req = InferReq.__new__(InferReq)
    req.req_idx = req.req_id = row
    req.cur_kv_len = 3
    req.hold_kv_len = hold_kv_len
    bind_cpu_indexes(req, infer_batch.g_infer_context.req_manager, gpu_index_submitted_len=gpu_index_submitted_len)
    req.mtp_step = 0
    req.filter_mark = req.wait_pause = req.paused = req.infer_aborted = False
    req.finish_status = infer_batch.FinishStatus()
    req.shm_req = SimpleNamespace(input_len=total_len)
    req.is_slave_req = lambda: False
    req.get_cur_total_len = lambda: total_len
    req.get_chuncked_input_token_len = lambda: min(total_len, req.cur_kv_len + 3)
    return req


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scheduler_submits_native_batch_on_upload_stream_and_returns_to_default_stream(monkeypatch):
    from cuda.bindings import runtime as cuda_runtime

    manager = _make_manager("cuda")
    req = _make_scheduled_req(0, gpu_index_submitted_len=0)
    backend, stream, _ = _make_scheduler(monkeypatch, manager, [req])
    submitted_streams = []
    native_copy = cuda_runtime.cudaMemcpyBatchAsync

    def record_batch(*args):
        submitted_streams.append(args[-1])
        return native_copy(*args)

    monkeypatch.setattr(cuda_runtime, "cudaMemcpyBatchAsync", record_batch)
    with torch.cuda.stream(torch.cuda.default_stream()):
        backend._get_classed_reqs(req_ids=[0])
        assert torch.cuda.current_stream().cuda_stream == 0
    torch.cuda.default_stream().synchronize()

    assert len(submitted_streams) == 1
    assert submitted_streams[0] not in (0, stream.cuda_stream)
    assert req.token_index.gpu_index_submitted_len == req.hold_kv_len
    assert torch.equal(
        manager.req_to_token_indexs[0, : req.hold_kv_len].cpu(), manager.token_indexes_cpu[0, : req.hold_kv_len]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scheduler_uploads_only_new_tail_and_reuses_uploaded_page(monkeypatch):
    manager = _make_manager("cuda")
    manager.token_indexes_cpu[0, :4].copy_(torch.arange(4, dtype=torch.int32))
    manager.req_to_token_indexs[0, :4].copy_(manager.token_indexes_cpu[0, :4])
    req = _make_scheduled_req(0)
    backend, stream, copies = _make_scheduler(monkeypatch, manager, [req])

    with torch.cuda.stream(torch.cuda.default_stream()):
        prefill, decode = backend._get_classed_reqs(req_ids=[0])
    torch.cuda.default_stream().synchronize()
    assert prefill == [req] and decode == []
    assert copies == [[(0, 4, 8)]]
    assert req.token_index.gpu_index_submitted_len == req.hold_kv_len == 8
    assert manager.req_to_token_indexs[0, :8].cpu().tolist() == [0, 1, 2, 3, 100, 101, 102, 103]

    req.cur_kv_len = 5  # The next chunk still fits into the reserved page.
    with torch.cuda.stream(torch.cuda.default_stream()):
        backend._get_classed_reqs(req_ids=[0])
    assert copies == [[(0, 4, 8)]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scheduler_leaves_budget_skipped_request_unpublished(monkeypatch):
    manager = _make_manager("cuda")
    manager.token_indexes_cpu[0, :8].copy_(torch.arange(8, dtype=torch.int32))
    req = _make_scheduled_req(0, gpu_index_submitted_len=0, hold_kv_len=8)
    backend, stream, copies = _make_scheduler(monkeypatch, manager, [req], budget=2)

    assert backend._get_classed_reqs(req_ids=[0]) == ([], [])
    assert req.token_index.gpu_index_submitted_len == 0
    assert copies == []
    assert manager.req_to_token_indexs.eq(-1).all()

    backend.batch_max_tokens = 3
    with torch.cuda.stream(torch.cuda.default_stream()):
        backend._get_classed_reqs(req_ids=[0])
    torch.cuda.default_stream().synchronize()
    assert copies == [[(0, 0, 8)]]
    assert req.token_index.gpu_index_submitted_len == 8


@pytest.mark.parametrize("shared_prefix_len", [0, 4])
@pytest.mark.parametrize("is_master_in_dp", [False, True])
def test_pd_transfer_failure_defers_owned_tail_cleanup(monkeypatch, shared_prefix_len, is_master_in_dp):
    from lightllm.server.router.model_infer.mode_backend.pd.decode_node_impl import decode_impl

    manager = _make_manager()
    manager.token_indexes_cpu[0, :12].copy_(torch.arange(12, dtype=torch.int32))
    shared_node = SimpleNamespace(node_prefix_total_len=shared_prefix_len) if shared_prefix_len else None
    simulated_finishes = []
    req = SimpleNamespace(
        req_idx=0,
        req_id=0,
        infer_aborted=False,
        pd_task_num=2,
        pd_task_failed_num=1,
        pd_task_success_num=0,
        finish_status=infer_batch.FinishStatus(),
        shared_kv_node=shared_node,
        cur_kv_len=10,
        hold_kv_len=12,
        cur_output_len=3,
        cache_tiers=(infer_batch.CacheTier.GPU,),
        shm_req=SimpleNamespace(
            shm_cur_kv_len=10,
            mark_simulated_finished=lambda status, output_len: simulated_finishes.append((status, output_len)),
        ),
        get_input_token_ids=lambda: list(range(12)),
    )
    bind_cpu_indexes(req, manager, gpu_index_submitted_len=12)
    original_token_index_len = req.token_index.token_index_len
    original_buffer = req.token_index._cpu_buffer
    monkeypatch.setattr(decode_impl, "g_infer_context", SimpleNamespace(requests_mapping={0: req}))
    freed = []
    manager.mem_manager = SimpleNamespace(free=lambda indexes: freed.extend(indexes.tolist()))
    backend = decode_impl.PDDecodeNode.__new__(decode_impl.PDDecodeNode)
    backend.is_master_in_dp = is_master_in_dp
    backend.model = SimpleNamespace(req_manager=manager, mem_manager=manager.mem_manager)

    # 任意传输任务仍未结束时，不能结束请求、回退逻辑长度或释放物理容量。
    assert backend._filter_not_ready_reqs([0]) == []
    assert not req.finish_status.is_finished()
    assert req.cur_kv_len == req.shm_req.shm_cur_kv_len == 10
    assert freed == simulated_finishes == []

    req.pd_task_success_num = 1
    for _ in range(2):
        assert backend._filter_not_ready_reqs([0]) == [req]
        assert req.finish_status.get_status() == infer_batch.FinishStatus.FINISHED_ERROR
        assert req.cur_kv_len == shared_prefix_len
        assert req.shm_req.shm_cur_kv_len == (shared_prefix_len if is_master_in_dp else 10)
        assert req.hold_kv_len == req.token_index.gpu_index_submitted_len == 12
        assert req.token_index.token_index_len == original_token_index_len
        assert req.token_index._cpu_buffer is original_buffer
        assert freed == []
    assert simulated_finishes == ([(infer_batch.FinishStatus.FINISHED_ERROR, 3)] if is_master_in_dp else [])

    # 走真实统一释放路径：只保留可信共享前缀，失败的完整页和页内预留尾部都应回收。
    cached, released_refs = [], []

    def insert(key, indexes):
        cached.append((key, indexes))
        return len(key), shared_node

    context = InferenceContext()
    context.args = SimpleNamespace(page_size=4)
    context.req_manager = manager
    context.radix_cache = SimpleNamespace(insert=insert, dec_node_ref_counter=released_refs.append)
    released = []
    context.free_a_req_mem(released, req)
    manager.free_token(torch.cat(released))

    assert freed == list(range(shared_prefix_len, 12))
    assert len(cached) == 1
    assert cached[0][0].tolist() == cached[0][1].tolist() == list(range(shared_prefix_len))
    assert released_refs == ([shared_node] if shared_node is not None else [])
    assert req.shared_kv_node is None
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == 0
    assert req.token_index.token_index_len == req.token_index.gpu_index_submitted_len == 0
    assert req.token_index._cpu_buffer is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mixed", [False, True])
def test_scheduler_publishes_prefill_and_decode_before_return(monkeypatch, mixed):
    manager = _make_manager("cuda")
    for row in (0, 1):
        manager.token_indexes_cpu[row, :4].copy_(torch.arange(row * 4, row * 4 + 4, dtype=torch.int32))
    prefill_req = _make_scheduled_req(0, gpu_index_submitted_len=0)
    decode_req = _make_scheduled_req(1, total_len=4, gpu_index_submitted_len=0)
    backend, stream, copies = _make_scheduler(monkeypatch, manager, [prefill_req, decode_req], mixed=mixed)

    with torch.cuda.stream(torch.cuda.default_stream()):
        prefill, decode = backend._get_classed_reqs(req_ids=[0, 1])
    torch.cuda.default_stream().synchronize()

    assert copies == [[(0, 0, 8), (1, 0, 4)]]
    assert prefill_req.token_index.gpu_index_submitted_len == 8
    assert decode_req.token_index.gpu_index_submitted_len == 4
    assert prefill == ([prefill_req, decode_req] if mixed else [prefill_req])
    assert decode == ([] if mixed else [decode_req])
    for req in (prefill_req, decode_req):
        assert torch.equal(
            manager.req_to_token_indexs[req.req_idx, : req.hold_kv_len].cpu(),
            manager.token_indexes_cpu[req.req_idx, : req.hold_kv_len],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("loop_fails", [False, True])
def test_request_upload_lengths_follow_fallback_copy_result(monkeypatch, loop_fails):
    from cuda.bindings import runtime as cuda_runtime

    manager = _make_manager("cuda")
    reqs = [
        _make_scheduled_req(0, gpu_index_submitted_len=0),
        _make_scheduled_req(1, total_len=4, gpu_index_submitted_len=0),
    ]
    backend, stream, _ = _make_scheduler(monkeypatch, manager, reqs)
    monkeypatch.setattr(
        cuda_runtime,
        "cudaMemcpyBatchAsync",
        lambda *args: (cuda_runtime.cudaError_t.cudaErrorInvalidValue,),
    )

    if loop_fails:
        torch_copy = torch.Tensor.copy_

        def failed_copy(dst, src, *args, **kwargs):
            if dst.is_cuda and src.device.type == "cpu":
                raise RuntimeError("individual copy failed")
            return torch_copy(dst, src, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "copy_", failed_copy)
        with torch.cuda.stream(torch.cuda.default_stream()), pytest.raises(
            RuntimeError, match="individual copy failed"
        ):
            backend._get_classed_reqs(req_ids=[0, 1])
        assert [req.token_index.gpu_index_submitted_len for req in reqs] == [0, 0]
    else:
        with torch.cuda.stream(torch.cuda.default_stream()):
            backend._get_classed_reqs(req_ids=[0, 1])
        torch.cuda.default_stream().synchronize()
        for req in reqs:
            assert req.token_index.gpu_index_submitted_len == req.hold_kv_len
            assert torch.equal(
                manager.req_to_token_indexs[req.req_idx, : req.hold_kv_len].cpu(),
                manager.token_indexes_cpu[req.req_idx, : req.hold_kv_len],
            )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("batch_available", [False, True])
def test_copy_uses_current_stream_on_current_device(monkeypatch, batch_available):
    from lightllm.common import cuda_wrapper as base

    monkeypatch.setattr(base.PinnedH2DBatchCopier, "_native_batch_available", batch_available)
    manager = _make_manager("cuda:1")
    manager.token_indexes_cpu[0, :4].copy_(torch.arange(4, dtype=torch.int32))
    stream = torch.cuda.Stream(device=1)
    stream.wait_stream(torch.cuda.current_stream(1))
    with torch.cuda.device(0):
        with torch.cuda.stream(stream):
            cuda_memcpy_pinned_h2d_batch_async(
                [manager.req_to_token_indexs[0, :4]],
                [manager.token_indexes_cpu[0, :4]],
            )
            assert torch.cuda.current_device() == 1
            assert torch.cuda.current_stream().cuda_stream == stream.cuda_stream
        assert torch.cuda.current_device() == 0
    stream.synchronize()
    assert manager.req_to_token_indexs[0, :4].cpu().tolist() == list(range(4))
