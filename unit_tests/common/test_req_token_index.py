from types import SimpleNamespace
import weakref

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.req_manager import base
from lightllm.server.router.model_infer.infer_req_ext1 import (
    ReqTokenIndex,
    PinMemCacheReqTokenIndex,
    PinMemPageCacheReqTokenIndex,
    PureGpuReqTokenIndex,
    create_req_token_index,
)
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs


MODES = ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"]
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and pinned CPU memory")


@pytest.fixture(autouse=True)
def reset_page_offsets(monkeypatch):
    # 各用例模拟不同启动配置；隔离进程级页偏移，同一用例内的请求仍共享。
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_page_offsets", None)


def make_req(token_table, mode="pin_mem_cache", req_idx=0, input_len=0, page_size=1):
    req = InferReq.__new__(InferReq)
    req.req_idx = req_idx
    req.req_manager = SimpleNamespace(req_to_token_indexs=token_table, page_size=page_size)
    req.args = (
        StartArgs(page_size=page_size) if mode is None else StartArgs(req_token_index_mode=mode, page_size=page_size)
    )
    req.shm_req = SimpleNamespace(input_len=input_len)
    return req


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [
        (None, PinMemCacheReqTokenIndex),
        ("pin_mem_cache", PinMemCacheReqTokenIndex),
        ("pin_mem_page_cache", PinMemPageCacheReqTokenIndex),
        ("pure_gpu", PureGpuReqTokenIndex),
    ],
)
def test_factory_uses_request_mode_and_row_view_without_retaining_request(mode, expected_type):
    table = torch.arange(24, dtype=torch.int32).view(3, 8)
    req = make_req(table, mode, req_idx=1)
    # 构造只保存传入的流，不执行 CUDA 操作；用占位对象验证透传即可。
    index_stream = object()
    helper = create_req_token_index(req, index_stream=index_stream)

    assert not hasattr(InferReq, "move_to_gpu_index_len")
    assert isinstance(helper, expected_type)
    assert hasattr(helper, "_alloc_pinned_tensor") == (expected_type is PinMemPageCacheReqTokenIndex)
    assert not hasattr(ReqTokenIndex, "_pinned_buffer")
    assert not hasattr(ReqTokenIndex, "_pinned_offset")
    assert helper._index_stream is index_stream
    assert helper.req_idx == req.req_idx
    assert helper.gpu_indexes.data_ptr() == table[req.req_idx].data_ptr()
    expected = table.clone()
    expected[req.req_idx, 3] = -99
    helper.gpu_indexes[3] = -99
    assert torch.equal(table, expected)
    # 模拟真实持有关系；保留 Helper 时，请求也应能立即释放，无需循环 GC。
    req.token_index = helper
    request_ref = weakref.ref(req)
    del req
    assert request_ref() is None


@pytest.mark.parametrize("mode", MODES)
def test_start_args_and_cli_select_mode(mode):
    parser = make_argument_parser()
    assert parser.parse_args([]).req_token_index_mode == "pin_mem_cache"
    assert StartArgs().req_token_index_mode == "pin_mem_cache"
    assert parser.parse_args(["--req_token_index_mode", mode]).req_token_index_mode == mode
    assert StartArgs(req_token_index_mode=mode).req_token_index_mode == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["--req_token_index_mode", "unknown"])
    with pytest.raises(ValueError, match="Unknown request token index mode"):
        create_req_token_index(make_req(torch.empty((1, 8), dtype=torch.int32), "unknown"), index_stream=object())


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pinned", [False, True])
def test_append_accepts_cpu_input_and_allocates_pinned_storage_when_needed(monkeypatch, mode, pinned):
    table = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    helper = create_req_token_index(make_req(table, mode), index_stream=torch.cuda.current_stream())
    indexes = torch.arange(4, dtype=torch.int32, pin_memory=pinned)
    alloc_pinned = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor
    pinned_sizes = []

    def record_alloc_pinned(need_size):
        pinned_sizes.append(need_size)
        return alloc_pinned(need_size)

    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_alloc_pinned_tensor", staticmethod(record_alloc_pinned))
    helper.append_token_indexes(indexes)
    # 普通 CPU 输入已经复制到自有 storage，可在 append 返回后立即改写。
    if not pinned:
        indexes.fill_(-99)

    assert pinned_sizes == []
    assert helper.token_index_len == 4
    assert helper.get_cpu_token_indexes(0, 4).tolist() == [0, 1, 2, 3]
    assert helper.get_gpu_token_indexes(0, 4).cpu().tolist() == [0, 1, 2, 3]
    if mode != "pure_gpu":
        assert helper._cpu_buffer.is_pinned()
        assert helper._cpu_buffer.untyped_storage().nbytes() == helper._cpu_buffer.nbytes


@requires_cuda
def test_shared_pinned_allocator_preserves_slices_across_slab_replacement(monkeypatch):
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_pinned_buffer", None)
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_pinned_offset", 0)
    buffer_token_num = 1024 * 1024
    first = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(5)
    second = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(3)
    first.fill_(11)
    second.fill_(22)
    assert second.data_ptr() == first.data_ptr() + first.nbytes
    assert first.is_pinned() and second.is_pinned()
    assert first.dtype == second.dtype == torch.int32
    assert not first.requires_grad and not second.requires_grad

    buffer = PinMemPageCacheReqTokenIndex._pinned_buffer
    empty = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(0)
    oversized = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(buffer_token_num + 1)
    assert empty.numel() == 0 and oversized.numel() == buffer_token_num + 1 and oversized.is_pinned()
    assert PinMemPageCacheReqTokenIndex._pinned_buffer is buffer
    assert PinMemPageCacheReqTokenIndex._pinned_offset == 8
    assert oversized.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()

    replacement = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(buffer_token_num - 7)
    replacement.fill_(33)
    assert replacement.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
    assert first.tolist() == [11] * 5 and second.tolist() == [22] * 3
    assert PinMemPageCacheReqTokenIndex._pinned_buffer.numel() == buffer_token_num
    assert PinMemPageCacheReqTokenIndex._pinned_offset == buffer_token_num - 7


@requires_cuda
def test_shared_pinned_allocator_rollover_keeps_released_h2d_source_valid(monkeypatch):
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_pinned_buffer", None)
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_pinned_offset", 0)
    buffer_token_num = 1024 * 1024
    source = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(buffer_token_num)
    source.fill_(11)
    source_ref = weakref.ref(source)
    target = torch.empty(buffer_token_num, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        target.copy_(source, non_blocking=True)
    del source
    assert source_ref() is None and not stream.query()

    # 连续换掉多个 slab，旧 storage 释放后仍须等在途 H2D 结束才能被重新使用。
    for value in range(22, 26):
        replacement = PinMemPageCacheReqTokenIndex._alloc_pinned_tensor(buffer_token_num)
        replacement.fill_(value)
        del replacement
    stream.synchronize()
    assert target.eq(11).all().item()


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_append_rejects_gpu_input_without_changing_index_state(mode):
    table = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    helper = create_req_token_index(make_req(table, mode), index_stream=torch.cuda.current_stream())

    with pytest.raises(AssertionError, match="mem_indexes must be a CPU Tensor"):
        helper.append_token_indexes(torch.arange(4, dtype=torch.int32, device="cuda"))

    assert helper.token_index_len == helper.gpu_index_submitted_len == 0
    assert table.eq(-1).all()


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_gpu_getter_rejects_unrecorded_indexes_without_advancing_progress(mode):
    table = torch.full((1, 8), -7, dtype=torch.int32, device="cuda")
    helper = create_req_token_index(make_req(table, mode), index_stream=torch.cuda.current_stream())
    helper.append_token_indexes(torch.tensor([10, 11], dtype=torch.int32, pin_memory=True))
    submitted_len = helper.gpu_index_submitted_len

    with pytest.raises(AssertionError, match="Read range exceeds"):
        helper.get_gpu_token_indexes(0, 4)

    assert helper.token_index_len == 2
    assert helper.gpu_index_submitted_len == submitted_len
    assert helper.get_gpu_token_indexes(0, 2).cpu().tolist() == [10, 11]


@requires_cuda
def test_batch_upload_rejects_missing_gpu_submission_without_advancing_progress(monkeypatch):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    table = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    req = make_req(table, "pure_gpu")
    req.token_index = create_req_token_index(req, index_stream=torch.cuda.current_stream())
    # 模拟不一致状态：记录了索引，但没有提交 GPU 写入；批量入口不能替它补写进度。
    req.token_index.token_index_len = 4
    monkeypatch.setattr(base_backend, "g_infer_context", SimpleNamespace(req_manager=req.req_manager))
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)

    with pytest.raises(AssertionError, match="GPU index submission progress does not match recorded length"):
        backend._copy_req_token_indexes_to_gpu([req])

    assert req.token_index.gpu_index_submitted_len == 0
    assert not hasattr(backend, "_index_upload_stream")


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_backend_rejects_requests_with_different_index_streams(monkeypatch, mode):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    table = torch.full((2, 8), -1, dtype=torch.int32, device="cuda")
    reqs = []
    for row in range(2):
        index_stream = torch.cuda.Stream()
        index_stream.wait_stream(torch.cuda.current_stream())
        req = make_req(table, mode, req_idx=row)
        req.token_index = create_req_token_index(req, index_stream=index_stream)
        req.token_index.append_token_indexes(torch.arange(4, dtype=torch.int32, pin_memory=True))
        reqs.append(req)

    def unexpected_upload(*args):
        pytest.fail("A batch with mismatched index streams must be rejected before upload")

    monkeypatch.setattr(base_backend, "cuda_memcpy_pinned_h2d_batch_async", unexpected_upload)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    with pytest.raises(AssertionError, match="All requests in a batch must use the same index stream"):
        backend._copy_req_token_indexes_to_gpu(reqs)
    assert [req.token_index.gpu_index_submitted_len for req in reqs] == [4 if mode == "pure_gpu" else 0] * 2


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
def test_token_units_and_snapshot_ownership(mode, page_size):
    row = torch.full((4 * page_size,), -1, dtype=torch.int32, device="cuda")
    req = make_req(row.unsqueeze(0), mode, page_size=page_size)
    helper = create_req_token_index(req, index_stream=torch.cuda.default_stream())
    req.token_index = helper
    count = row.numel()
    positions = torch.arange(count, dtype=torch.int32)
    expected = positions.div(page_size, rounding_mode="floor") * (3 * page_size) + positions.remainder(page_size)
    original = expected.pin_memory()
    split = page_size if mode == "pin_mem_page_cache" else page_size + 1
    req.token_index.append_token_indexes(original[:split])
    req.token_index.append_token_indexes(torch.empty(0, dtype=torch.int32))
    assert helper.token_index_len == split
    req.token_index.append_token_indexes(original[split:])
    assert helper.token_index_len == count
    ranges = (
        [(0, count), (page_size, 3 * page_size), (count, count)]
        if mode == "pin_mem_page_cache"
        else [(0, count), (1, count - 1), (page_size, page_size + 2), (count, count)]
    )
    for start, end in ranges:
        assert torch.equal(req.token_index.get_cpu_token_indexes(start, end), expected[start:end])
    snapshot = req.token_index.get_cpu_token_indexes(0, count)
    # pure_gpu 的阻塞 CPU getter 已等待 H2D 完成，此后才可主动改写上传源。
    original.fill_(99)
    # GPU 读取区间只决定返回视图；首次读取同时提交内部记录的全部待上传索引。
    assert torch.equal(req.token_index.get_gpu_token_indexes(1, 3).cpu(), expected[1:3])
    assert helper.gpu_index_submitted_len == count
    gpu = req.token_index.get_gpu_token_indexes(0, count)
    assert gpu.data_ptr() == row.data_ptr()
    assert torch.equal(gpu.cpu(), expected)
    if mode != "pure_gpu":
        cpu_buffer = helper._cpu_buffer
    helper.reset()
    assert helper.token_index_len == helper.gpu_index_submitted_len == 0
    if mode != "pure_gpu":
        assert helper._cpu_buffer is None
    else:
        assert not hasattr(helper, "_cpu_buffer")
    replacement = torch.arange(count, dtype=torch.int32, pin_memory=True)
    helper.append_token_indexes(replacement)
    if mode != "pure_gpu":
        assert helper._cpu_buffer is not cpu_buffer
    assert torch.equal(helper.get_gpu_token_indexes(0, count).cpu(), replacement)
    assert torch.equal(snapshot, expected)


@requires_cuda
@pytest.mark.parametrize("page_size", [1, 3, 4, 64])
def test_page_cache_compresses_page_starts_and_uploads_aligned_token_ranges(page_size):
    table = torch.full((1, 8 * page_size), -1, dtype=torch.int32, device="cuda")
    req = make_req(table, "pin_mem_page_cache", input_len=4 * page_size, page_size=page_size)
    helper = create_req_token_index(req, index_stream=torch.cuda.default_stream())
    page_starts = torch.tensor([9, 2, 15, 4], dtype=torch.int32) * page_size
    expected = (page_starts[:, None] + torch.arange(page_size, dtype=torch.int32)).reshape(-1)
    source = expected.clone()
    start = 0
    for end in (page_size, 2 * page_size, 3 * page_size, expected.numel()):
        helper.append_token_indexes(source[start:end])
        start = end
    source.fill_(-99)

    assert helper._cpu_buffer.is_pinned() and helper._cpu_buffer.numel() == 8
    assert helper._cpu_buffer.nbytes * page_size == table.nbytes
    assert torch.equal(helper._cpu_buffer[:4], page_starts)
    for start, end in ((0, expected.numel()), (page_size, 3 * page_size), (page_size, page_size)):
        assert torch.equal(helper.get_cpu_token_indexes(start, end), expected[start:end])
        assert torch.equal(helper.get_free_token_indexes(start, end), expected[start:end])

    snapshot = helper.get_cpu_token_indexes(0, expected.numel())
    upload = helper.get_upload_indexes()
    assert upload.is_pinned() and torch.equal(upload, expected)
    assert helper.gpu_index_submitted_len == 0
    if page_size > 1:
        with pytest.raises(AssertionError, match="start must be page-aligned"):
            helper.get_cpu_token_indexes(1, page_size)
        with pytest.raises(AssertionError, match="end must be page-aligned"):
            helper.get_free_token_indexes(0, page_size - 1)
        with pytest.raises(AssertionError, match="start must be page-aligned"):
            helper.get_cpu_token_indexes(1, 1)

    # 模拟完整前缀页已经提交，待上传接口从下一个页边界开始展开。
    submitted = page_size
    table[0, :submitted].copy_(upload[:submitted], non_blocking=True)
    helper.gpu_index_submitted_len = submitted
    pending = helper.get_upload_indexes()
    assert pending.is_pinned() and torch.equal(pending, expected[submitted:])
    assert helper.gpu_index_submitted_len == submitted
    assert torch.equal(helper.get_gpu_token_indexes(0, expected.numel()).cpu(), expected)
    assert helper.get_upload_indexes().numel() == 0
    assert table[0, expected.numel() :].eq(-1).all()

    helper.reset()
    helper.append_token_indexes(torch.arange(expected.numel(), dtype=torch.int32))
    assert torch.equal(snapshot, expected) and torch.equal(upload, expected)


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_free_getter_borrows_gpu_indexes_but_metadata_getter_keeps_cpu_snapshot(monkeypatch, mode):
    table = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    req = make_req(table, mode)
    req.token_index = create_req_token_index(req, index_stream=torch.cuda.default_stream())
    req.token_index.append_token_indexes(torch.arange(6, dtype=torch.int32, pin_memory=True))
    snapshot = req.token_index.get_cpu_token_indexes(1, 5)

    def unexpected_snapshot(*args):
        pytest.fail("The pure_gpu free getter must not call the CPU snapshot getter")

    with monkeypatch.context() as no_snapshot:
        if mode == "pure_gpu":
            no_snapshot.setattr(req.token_index, "get_cpu_token_indexes", unexpected_snapshot)
        free_indexes = req.token_index.get_free_token_indexes(1, 5)
        empty = req.token_index.get_free_token_indexes(3, 3)

    expected_device = "cuda" if mode == "pure_gpu" else "cpu"
    assert free_indexes.device.type == empty.device.type == expected_device
    assert empty.numel() == 0
    if mode == "pure_gpu":
        assert free_indexes.data_ptr() == table[0, 1:5].data_ptr()
    updated = torch.arange(101, 105, dtype=torch.int32)
    replacement = torch.arange(6, dtype=torch.int32, pin_memory=True)
    replacement[1:5] = updated
    req.token_index.reset()
    req.token_index.append_token_indexes(replacement)

    assert snapshot.device.type == "cpu"
    assert snapshot.tolist() == [1, 2, 3, 4]
    assert free_indexes.cpu().tolist() == (updated.tolist() if mode == "pure_gpu" else [1, 2, 3, 4])


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_reset_rebuild_keeps_cpu_appends_and_gpu_submission_separate(mode):
    helper = create_req_token_index(
        make_req(torch.zeros((1, 8), dtype=torch.int32, device="cuda"), mode), index_stream=torch.cuda.default_stream()
    )
    pending = helper.get_upload_indexes()
    if mode == "pure_gpu":
        assert pending is None
    else:
        assert pending.numel() == 0
    helper.append_token_indexes(torch.arange(4, dtype=torch.int32, pin_memory=True))
    pending = helper.get_upload_indexes()
    if mode == "pure_gpu":
        assert pending is None
    else:
        assert pending.tolist() == [0, 1, 2, 3]
    assert helper.gpu_index_submitted_len == (4 if mode == "pure_gpu" else 0)
    before = helper.get_cpu_token_indexes(0, 4)
    helper.get_gpu_token_indexes(0, 4)
    pending = helper.get_upload_indexes()
    if mode == "pure_gpu":
        assert pending is None
    else:
        assert pending.numel() == 0
    helper.reset()
    helper.append_token_indexes(torch.tensor([0, 31], dtype=torch.int32, pin_memory=True))
    assert helper.token_index_len == 2
    assert helper.gpu_index_submitted_len == (2 if mode == "pure_gpu" else 0)
    if mode != "pure_gpu":
        assert helper.gpu_indexes[:4].cpu().tolist() == [0, 1, 2, 3]
    helper.append_token_indexes(torch.tensor([32, 3], dtype=torch.int32, pin_memory=True))
    assert helper.token_index_len == 4
    assert helper.gpu_index_submitted_len == (4 if mode == "pure_gpu" else 0)
    assert helper.get_gpu_token_indexes(0, 4).cpu().tolist() == [0, 31, 32, 3]
    assert helper.gpu_index_submitted_len == 4
    assert before.tolist() == [0, 1, 2, 3]


@requires_cuda
@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache"])
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize(
    ("input_len", "max_length", "initial_capacity", "next_end", "grown_capacity"),
    [
        (1023, 16384, 3072, 3073, 6144),
        (1024, 16384, 3072, 3073, 6144),
        (1025, 16384, 4096, 4097, 7168),
        (2048, 16384, 4096, 5000, 7168),
        (1025, 5500, 4096, 5000, 5500),
    ],
)
def test_pin_buffer_capacity_and_lifecycle(
    mode, page_size, input_len, max_length, initial_capacity, next_end, grown_capacity
):
    req = make_req(
        torch.empty((1, max_length), dtype=torch.int32, device="cuda"), mode, input_len=input_len, page_size=page_size
    )
    helper = create_req_token_index(req, index_stream=torch.cuda.default_stream())
    storage_page_size = page_size if mode == "pin_mem_page_cache" else 1
    if mode == "pin_mem_page_cache":
        next_end = (next_end + page_size - 1) // page_size * page_size
        initial_elements = grown_elements = (max_length + page_size - 1) // page_size
    else:
        initial_elements = initial_capacity
        grown_elements = grown_capacity
    assert helper.input_len == input_len
    assert helper._cpu_buffer is None
    helper.append_token_indexes(torch.empty(0, dtype=torch.int32))
    assert helper._cpu_buffer is None
    # 首次追加最小存储单位；完整模式预留 2K 输出，页模式一次申请到最大长度。
    first_index = torch.arange(storage_page_size, dtype=torch.int32, pin_memory=True)
    helper.append_token_indexes(first_index)
    assert helper._cpu_buffer.numel() == initial_elements
    assert helper._cpu_buffer.is_pinned() and helper._cpu_buffer.data_ptr() != first_index.data_ptr()
    helper.append_token_indexes(torch.arange(storage_page_size, initial_capacity, dtype=torch.int32, pin_memory=True))
    old_buffer = helper._cpu_buffer
    helper.append_token_indexes(torch.arange(initial_capacity, next_end, dtype=torch.int32, pin_memory=True))
    assert helper._cpu_buffer.numel() == grown_elements and helper._cpu_buffer.is_pinned()
    if mode == "pin_mem_page_cache":
        assert helper._cpu_buffer is old_buffer
    else:
        assert helper._cpu_buffer.data_ptr() != old_buffer.data_ptr()
    snapshot = helper.get_cpu_token_indexes(0, next_end)
    assert snapshot.tolist() == list(range(next_end))
    grown_buffer = helper._cpu_buffer
    helper.reset()
    assert helper.input_len == input_len and helper._cpu_buffer is None
    assert helper.token_index_len == helper.gpu_index_submitted_len == 0
    replacement = torch.arange(99, 99 + storage_page_size, dtype=torch.int32, pin_memory=True)
    helper.append_token_indexes(replacement)
    assert helper._cpu_buffer is not grown_buffer
    assert helper._cpu_buffer.numel() == initial_elements
    assert helper.token_index_len == storage_page_size
    assert torch.equal(helper.get_cpu_token_indexes(0, helper.token_index_len), replacement)
    assert snapshot.tolist() == list(range(next_end))


@requires_cuda
@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache"])
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("use_default_stream", [False, True])
def test_pin_buffer_reset_during_upload_keeps_old_data_until_h2d_completes(mode, page_size, use_default_stream):
    count = 8192
    row = torch.full((count,), -1, dtype=torch.int32, device="cuda")
    index_stream = torch.cuda.default_stream() if use_default_stream else torch.cuda.Stream()
    index_stream.wait_stream(torch.cuda.current_stream())
    helper = create_req_token_index(
        make_req(row.unsqueeze(0), mode, input_len=count, page_size=page_size), index_stream=index_stream
    )
    original = torch.arange(count, dtype=torch.int32, pin_memory=True)
    replacement = (torch.arange(count, dtype=torch.int32).remainder(page_size) + count * 2).pin_memory()
    helper.append_token_indexes(original)
    source_ref = weakref.ref(helper._cpu_buffer)

    with torch.cuda.stream(index_stream):
        torch.cuda._sleep(200_000_000)
    helper.get_gpu_token_indexes(0, count)
    assert not index_stream.query(), "Delayed H2D must remain pending through reset and append"
    helper.reset()
    assert helper._cpu_buffer is None and source_ref() is None
    helper.append_token_indexes(replacement)

    # reset 已释放旧 Tensor 的 Python 引用；新缓存写入不能污染仍在传输的旧数据。
    index_stream.synchronize()
    assert torch.equal(row.cpu(), original)
    assert torch.equal(helper.get_gpu_token_indexes(0, count).cpu(), replacement)


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pre_upload", [False, True])
@pytest.mark.parametrize("use_default_stream", [False, True])
def test_gpu_getter_uses_injected_stream_and_orders_calling_stream(monkeypatch, mode, pre_upload, use_default_stream):
    row = torch.full((16,), -1, dtype=torch.int32, device="cuda")
    index_stream = torch.cuda.default_stream() if use_default_stream else torch.cuda.Stream()
    index_stream.wait_stream(torch.cuda.current_stream())
    reader = torch.cuda.Stream()
    with torch.cuda.stream(reader):
        # 构造时的当前流与传入的索引流不同，Helper 必须始终使用外部指定的流。
        helper = create_req_token_index(make_req(row.unsqueeze(0), mode), index_stream=index_stream)
        assert helper._index_stream is index_stream
        assert torch.cuda.current_stream() == reader
    copy = torch.Tensor.copy_
    cpu = torch.Tensor.cpu
    uploads = []

    def record_copy(dst, src, *args, **kwargs):
        if dst.is_cuda and src.device.type == "cpu":
            assert torch.cuda.current_stream() == index_stream
            uploads.append(src.numel())
        return copy(dst, src, *args, **kwargs)

    def check_cpu_read(src, *args, **kwargs):
        if src.is_cuda:
            assert torch.cuda.current_stream() == index_stream
        return cpu(src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    monkeypatch.setattr(torch.Tensor, "cpu", check_cpu_read)
    with torch.cuda.stream(index_stream):
        torch.cuda._sleep(30_000_000)
    with torch.cuda.stream(reader):
        helper.append_token_indexes(torch.arange(8, dtype=torch.int32, pin_memory=True))
        assert torch.cuda.current_stream() == reader
    if pre_upload:
        # 索引流已提交上传，但尚未完成；另一个流读取时仍必须承接依赖。
        helper.get_gpu_token_indexes(0, 8)
    with torch.cuda.stream(reader):
        indexes = helper.get_gpu_token_indexes(0, 8)
        assert torch.cuda.current_stream() == reader
        actual = indexes.clone()
        assert helper.get_cpu_token_indexes(0, 8).tolist() == list(range(8))
        assert torch.cuda.current_stream() == reader
    reader.synchronize()
    assert uploads == [8]
    with torch.cuda.stream(index_stream):
        assert actual.cpu().tolist() == list(range(8))


@requires_cuda
def test_pure_gpu_slot_reuse_keeps_old_and_new_helper_writes_in_order():
    row = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    torch.cuda._sleep(30_000_000)
    old = create_req_token_index(make_req(row.unsqueeze(0), "pure_gpu"), index_stream=torch.cuda.default_stream())
    old.append_token_indexes(torch.full((8,), 11, dtype=torch.int32, pin_memory=True))
    # 源 Tensor 引用已释放；先保留旧写入结果，验证后续 pinned 分配不会覆盖在途 H2D。
    first_write = row.clone()
    new = create_req_token_index(make_req(row.unsqueeze(0), "pure_gpu"), index_stream=torch.cuda.default_stream())
    new.append_token_indexes(torch.full((8,), 22, dtype=torch.int32, pin_memory=True))
    assert new.get_cpu_token_indexes(0, 8).tolist() == [22] * 8
    assert first_write.cpu().tolist() == [11] * 8


@requires_cuda
@pytest.mark.parametrize("page_size", [1, 4, 64])
def test_manager_owns_only_gpu_table_and_initializes_hold_page(monkeypatch, page_size):
    from lightllm.common.req_manager import req_sampling_params

    monkeypatch.setattr(base, "get_env_start_args", lambda: SimpleNamespace(page_size=page_size))
    monkeypatch.setattr(req_sampling_params, "ReqSamplingParamsManager", lambda count: None)
    mem_manager = SimpleNamespace(page_size=page_size, HOLD_TOKEN_MEMINDEXES=tuple(range(page_size * 8, page_size * 9)))
    manager = base.ReqManager(2, page_size * 2 + 1, mem_manager)
    assert not hasattr(manager, "req_token_index_mode")
    assert not hasattr(manager, "req_to_page_indexs_cpu")
    assert not hasattr(manager, "get_cpu_token_indexes")
    assert manager.req_to_token_indexs.shape == (3, 3 * page_size)
    assert (
        manager.req_to_token_indexs[manager.HOLD_REQUEST_ID].cpu().tolist()
        == list(mem_manager.HOLD_TOKEN_MEMINDEXES) * 3
    )


@requires_cuda
@pytest.mark.parametrize("batch_available", [False, True])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("index_stream_kind", ["default", "shared"])
def test_backend_uploads_pinned_views_without_request_events_or_keepalive_list(
    monkeypatch, mode, page_size, batch_available, index_stream_kind
):
    from lightllm.common import cuda_wrapper
    from lightllm.server.router.model_infer.mode_backend import base_backend

    count = max(8, page_size)
    table = torch.full((2, 2 * count), -1, dtype=torch.int32, device="cuda")
    expected = (torch.arange(2 * count, dtype=torch.int32).view(2, count) + 32).pin_memory()
    index_streams = [torch.cuda.default_stream()] * 2
    if index_stream_kind == "shared":
        index_streams = [torch.cuda.Stream()] * 2
    for index_stream in set(index_streams):
        index_stream.wait_stream(torch.cuda.default_stream())
    reqs = []
    for row, index_stream in enumerate(index_streams):
        req = make_req(table, mode, req_idx=row, page_size=page_size)
        req.token_index = create_req_token_index(req, index_stream=index_stream)
        with torch.cuda.stream(index_stream):
            # 批量上传须先等待旧写入；pure_gpu 在本流追加，调用流尚未承接依赖。
            torch.cuda._sleep(60_000_000)
            table[row, :count].fill_(-2)
            req.token_index.append_token_indexes(expected[row])
        req.hold_kv_len = count
        reqs.append(req)
    monkeypatch.setattr(base_backend, "g_infer_context", SimpleNamespace(req_manager=reqs[0].req_manager))
    monkeypatch.setattr(cuda_wrapper.PinnedH2DBatchCopier, "_native_batch_available", batch_available)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    stream = index_streams[0]
    stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(30_000_000)
    events = []
    batch_calls = []
    copy_batch = base_backend.cuda_memcpy_pinned_h2d_batch_async
    event_type = torch.cuda.Event

    def record_batch(dst_tensors, src_tensors):
        # 完整缓存直接借用视图，页缓存先展开；两者都应一次批量提交正确的 token 索引。
        assert all(tensor.is_pinned() for tensor in src_tensors)
        assert all(torch.equal(tensor, expected[row]) for row, tensor in enumerate(src_tensors))
        if mode == "pin_mem_cache":
            assert [tensor.data_ptr() for tensor in src_tensors] == [
                req.token_index.get_upload_indexes().data_ptr() for req in reqs
            ]
        batch_calls.append(len(src_tensors))
        return copy_batch(dst_tensors, src_tensors)

    def record_event(*args, **kwargs):
        event = event_type(*args, **kwargs)
        events.append(event)
        return event

    with monkeypatch.context() as batch_events:
        batch_events.setattr(torch.cuda, "Event", record_event)
        batch_events.setattr(base_backend, "cuda_memcpy_pinned_h2d_batch_async", record_batch)
        backend._copy_req_token_indexes_to_gpu(reqs)
        observed = []
        for req, index_stream in zip(reqs, index_streams):
            with torch.cuda.stream(index_stream):
                # 不等待后台线程：已标记提交的索引必须能在请求自己的索引流立即使用。
                observed.append(req.token_index.get_gpu_token_indexes(0, count).clone())
        if mode != "pure_gpu":
            for req in reqs:
                req.token_index.reset()
                req.token_index.append_token_indexes(torch.arange(count, dtype=torch.int32, pin_memory=True))
        # 两个请求只建立一个 batch 保活事件；pure_gpu 与 PyTorch 回退无需该事件。
        assert len(events) == (1 if mode != "pure_gpu" and batch_available else 0)
        assert batch_calls == ([] if mode == "pure_gpu" else [2])
        pool = cuda_wrapper.PinnedH2DBatchCopier._release_pool
        if pool is not None:
            pool.submit(lambda: None).result(timeout=10)
    assert [req.token_index.gpu_index_submitted_len for req in reqs] == [count if mode == "pure_gpu" else 0] * 2
    assert not hasattr(backend, "_pending_cpu_index_uploads")
    assert not hasattr(backend, "_index_upload_stream")
    assert all(not hasattr(req.token_index, "_gpu_write_event") for req in reqs)
    assert torch.equal(table[:, :count].cpu(), expected)
    for actual, expected_row, index_stream in zip(observed, expected, index_streams):
        with torch.cuda.stream(index_stream):
            assert torch.equal(actual.cpu(), expected_row)
    backend._copy_req_token_indexes_to_gpu([])


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_backend_waits_for_submitted_indexes_without_pending_cpu_uploads(monkeypatch, mode):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    table = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    index_stream = torch.cuda.default_stream()
    reader_stream = torch.cuda.Stream()
    reader_stream.wait_stream(index_stream)
    req = make_req(table, mode)
    req.token_index = create_req_token_index(req, index_stream=index_stream)
    with torch.cuda.stream(index_stream):
        torch.cuda._sleep(200_000_000)
        req.token_index.append_token_indexes(torch.arange(8, dtype=torch.int32, pin_memory=True))
        req.token_index.get_gpu_token_indexes(0, 8)
    assert req.token_index.gpu_index_submitted_len == req.token_index.token_index_len == 8
    assert not index_stream.query(), "Submitted indexes must still be pending with no new CPU upload range"

    def unexpected_upload(*args):
        pytest.fail("No copy should be submitted when there are no pending upload ranges")

    monkeypatch.setattr(base_backend, "cuda_memcpy_pinned_h2d_batch_async", unexpected_upload)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    with torch.cuda.stream(reader_stream):
        backend._copy_req_token_indexes_to_gpu([req])
        # 直接读取请求表，避免 GPU getter 自带的等待掩盖 batch 入口遗漏的依赖。
        observed = table.clone()
    reader_stream.synchronize()
    assert observed.cpu().tolist() == [list(range(8))]


@requires_cuda
@pytest.mark.parametrize("mode", MODES)
def test_dp_publication_waits_for_the_selected_index_mode(monkeypatch, mode):
    from lightllm.server.router.model_infer.mode_backend import base_backend
    from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans

    table = torch.full((1, 8), -1, dtype=torch.int32, device="cuda")
    torch.cuda._sleep(30_000_000)
    table.fill_(-2)
    req = make_req(table, mode)
    req.token_index = create_req_token_index(req, index_stream=torch.cuda.default_stream())
    req.token_index.append_token_indexes(torch.arange(8, dtype=torch.int32, pin_memory=True))
    req.cur_kv_len = req.hold_kv_len = 8
    context = SimpleNamespace(req_manager=req.req_manager)
    monkeypatch.setattr(base_backend, "g_infer_context", context)
    monkeypatch.setattr(dp_shared_kv_trans, "g_infer_context", context)
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.node_nccl_group = object()
    backend.is_master_in_dp = False

    def barrier(group):
        torch.cuda.current_stream().synchronize()
        assert table.cpu().tolist() == [list(range(8))]

    monkeypatch.setattr(dp_shared_kv_trans.dist, "barrier", barrier)
    module = dp_shared_kv_trans.DPKVSharedMoudle.__new__(dp_shared_kv_trans.DPKVSharedMoudle)
    module.backend = backend
    module.fill_reqs_info([req])
