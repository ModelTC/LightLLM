"""验证 raw H2D 的源引用由完成事件和后台线程保活，不依赖服务启动。"""

import gc
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import weakref

import pytest
import torch

from lightllm.common import cuda_wrapper

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and pinned CPU memory")


@pytest.fixture(autouse=True)
def release_futures(monkeypatch):
    monkeypatch.setattr(cuda_wrapper.PinnedH2DBatchCopier, "_native_batch_available", True)
    pool_factory = cuda_wrapper.ThreadPoolExecutor
    pools = {}
    futures = []

    def track_pool(pool):
        original_submit = pool.submit
        pools[pool] = original_submit

        def record_submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            futures.append(future)
            return future

        monkeypatch.setattr(pool, "submit", record_submit)

    def create_pool(*args, **kwargs):
        pool = pool_factory(*args, **kwargs)
        track_pool(pool)
        return pool

    # 观察已有或后续实际创建的线程池，夹具自身不触发延迟初始化。
    pool = cuda_wrapper.PinnedH2DBatchCopier._release_pool
    if pool is not None:
        track_pool(pool)
    monkeypatch.setattr(cuda_wrapper, "ThreadPoolExecutor", create_pool)
    yield futures
    # 先收尾所有真实等待任务，再撤销本例 monkeypatch；生产池可被后续测试继续复用。
    for future in futures:
        future.result(timeout=10)
    for submit in pools.values():
        submit(lambda: None).result(timeout=10)


def test_empty_batch_and_fallback_do_not_create_release_pool(monkeypatch):
    copier = cuda_wrapper.PinnedH2DBatchCopier
    monkeypatch.setattr(copier, "_release_pool", None)
    monkeypatch.setattr(copier, "_native_batch_available", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))

    def unexpected_pool(*args, **kwargs):
        pytest.fail("空批次和 copy_ 回退不能创建保活线程池")

    monkeypatch.setattr(cuda_wrapper, "ThreadPoolExecutor", unexpected_pool)
    cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([], [])
    source = torch.arange(8, dtype=torch.int32)
    target = torch.empty_like(source)
    cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([target], [source])
    assert copier._release_pool is None
    assert torch.equal(target, source)


def test_release_pool_is_initialized_once_for_concurrent_callers(monkeypatch):
    copier = cuda_wrapper.PinnedH2DBatchCopier
    monkeypatch.setattr(copier, "_release_pool", None)
    entered = threading.Event()
    allow_creation = threading.Event()
    start = threading.Barrier(8)
    shared_pool = object()
    creations = []

    def create_pool(**kwargs):
        creations.append(kwargs)
        entered.set()
        assert allow_creation.wait(timeout=5)
        return shared_pool

    def get_pool():
        start.wait(timeout=5)
        return copier._get_release_pool()

    monkeypatch.setattr(cuda_wrapper, "ThreadPoolExecutor", create_pool)
    with ThreadPoolExecutor(max_workers=8) as callers:
        futures = [callers.submit(get_pool) for _ in range(8)]
        try:
            assert entered.wait(timeout=5)
        finally:
            allow_creation.set()
        assert all(future.result(timeout=5) is shared_pool for future in futures)
    assert creations == [{"max_workers": 1, "thread_name_prefix": "lightllm-pinned-release"}]


@requires_cuda
@pytest.mark.parametrize("native_batch", [False, True])
def test_default_stream_copy_orders_reads_and_reuses_upload_stream(monkeypatch, native_batch):
    runtime = pytest.importorskip("cuda.bindings.runtime")
    copier = cuda_wrapper.PinnedH2DBatchCopier
    monkeypatch.setattr(copier, "_native_batch_available", native_batch)
    stream = torch.cuda.default_stream()
    source = torch.arange(8192, dtype=torch.int32, pin_memory=True)
    target = torch.empty(8192, dtype=torch.int32, device="cuda")
    native_copy = runtime.cudaMemcpyBatchAsync
    submitted_streams, observed = [], []

    def record_batch(*args):
        submitted_streams.append(args[-1])
        return native_copy(*args)

    monkeypatch.setattr(runtime, "cudaMemcpyBatchAsync", record_batch)
    with torch.cuda.stream(stream):
        for _ in range(2):
            torch.cuda._sleep(20_000_000)
            target.fill_(-1)
            cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([target], [source])
            assert torch.cuda.current_stream() == stream
            observed.append(target.clone())
    for result in observed:
        assert torch.equal(result.cpu(), source)
    if native_batch:
        assert len(submitted_streams) == 2
        assert submitted_streams[0] == submitted_streams[1] != 0
    else:
        assert submitted_streams == []


@requires_cuda
def test_raw_batch_records_one_completion_event_without_small_copy(monkeypatch, release_futures):
    runtime = pytest.importorskip("cuda.bindings.runtime")
    source = torch.arange(2048, dtype=torch.int32, pin_memory=True)
    another_source = torch.arange(2048, dtype=torch.int32, pin_memory=True).add_(10000)
    sources = [source[128:384], source[1000:1256], another_source[17:273]]
    target = torch.full((1024,), -1, dtype=torch.int32, device="cuda")
    targets = [target[2:258], target[300:556], target[600:856]]
    expected = torch.full((1024,), -1, dtype=torch.int32)
    expected[2:258], expected[300:556], expected[600:856] = sources
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    calls = []
    native_copy, event_record = runtime.cudaMemcpyBatchAsync, torch.cuda.Event.record

    def record_native(*args):
        calls.append(("batch", args[3], args[-1]))
        return native_copy(*args)

    def record_event(event, on_stream=None):
        calls.append(("event", 1, on_stream.cuda_stream))
        return event_record(event, on_stream)

    def unexpected_copy(*args, **kwargs):
        pytest.fail("raw batch 保活不应额外提交小片段 copy_ 或进入逐项回退")

    def unexpected_sync(*args, **kwargs):
        pytest.fail("正常 raw batch 不应在提交线程同步 CUDA 流")

    monkeypatch.setattr(runtime, "cudaMemcpyBatchAsync", record_native)
    monkeypatch.setattr(torch.cuda.Event, "record", record_event)
    monkeypatch.setattr(torch.Tensor, "copy_", unexpected_copy)
    with torch.cuda.stream(stream), monkeypatch.context() as no_sync:
        no_sync.setattr(torch.cuda.Stream, "synchronize", unexpected_sync)
        no_sync.setattr(torch.cuda, "synchronize", unexpected_sync)
        cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async(targets, sources)
        assert len(release_futures) == 1
        release_futures[0].result(timeout=10)
    assert cuda_wrapper.PinnedH2DBatchCopier._native_batch_available
    assert calls == [("batch", 3, stream.cuda_stream), ("event", 1, stream.cuda_stream)]
    assert torch.equal(target.cpu(), expected)


@requires_cuda
def test_background_wait_holds_tensor_references_without_blocking_submitter(monkeypatch, release_futures):
    pytest.importorskip("cuda.bindings.runtime")
    source = torch.arange(8192, dtype=torch.int32, pin_memory=True)
    source_views = [source[64:1088], source[1088:4160]]
    expected = torch.cat(source_views)
    source_ref = weakref.ref(source)
    view_refs = [weakref.ref(view) for view in source_views]
    target = torch.empty(4096, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    worker_wait_started = threading.Event()
    allow_worker_finish = threading.Event()
    main_thread = threading.get_ident()
    event_synchronize = torch.cuda.Event.synchronize

    def controlled_wait(event):
        assert threading.get_ident() != main_thread, "CUDA 完成事件只能由后台等待线程同步"
        worker_wait_started.set()
        assert allow_worker_finish.wait(timeout=10), "测试未允许后台等待结束"
        event_synchronize(event)

    monkeypatch.setattr(torch.cuda.Event, "synchronize", controlled_wait)
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([target[:1024], target[1024:]], source_views)
        # gate 仍关闭时提交调用已返回，且后台任务正在等待，而非阻塞主线程。
        assert worker_wait_started.wait(timeout=5)
        assert len(release_futures) == 1 and not release_futures[0].done()
        source_views.clear()
        del source_views, source
        gc.collect()
        assert source_ref() is not None and all(reference() is not None for reference in view_refs)
    finally:
        allow_worker_finish.set()
        for future in release_futures:
            assert future.result(timeout=10) is None
        stream.synchronize()
    gc.collect()
    assert source_ref() is None and all(reference() is None for reference in view_refs)
    assert torch.equal(target.cpu(), expected)


@requires_cuda
@pytest.mark.parametrize("native_batch", [True, False])
@pytest.mark.parametrize("shared_storage", [True, False])
def test_async_copy_keeps_freed_source_storage_out_of_reuse(monkeypatch, native_batch, shared_storage):
    if native_batch:
        pytest.importorskip("cuda.bindings.runtime")
    monkeypatch.setattr(cuda_wrapper.PinnedH2DBatchCopier, "_native_batch_available", native_batch)
    source = torch.arange(8192, dtype=torch.int32, pin_memory=True)
    another_source = source if shared_storage else torch.arange(8192, dtype=torch.int32, pin_memory=True).add_(10000)
    source_ptrs = {source.data_ptr(), another_source.data_ptr()}
    source_views = [source[64:1088], another_source[1088:4160]]
    expected = torch.cat(source_views)
    target = torch.empty(4096, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())

    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)
        cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([target[:1024], target[1024:]], source_views)
    assert cuda_wrapper.PinnedH2DBatchCopier._native_batch_available is native_batch
    del source_views, source, another_source

    pending_checks = 0
    for _ in range(64):
        still_pending = not stream.query()
        replacement = torch.empty(8192, dtype=torch.int32, pin_memory=True)
        if still_pending:
            assert replacement.data_ptr() not in source_ptrs
            pending_checks += 1
        replacement.fill_(-1)
        del replacement
    stream.synchronize()
    assert pending_checks > 0, "延时传输应覆盖源释放和 pinned 内存复用检查"
    assert torch.equal(target.cpu(), expected)


@requires_cuda
@pytest.mark.parametrize("failure", ["record", "pool", "submit"])
def test_event_setup_failure_propagates_without_sync_or_fallback(monkeypatch, failure):
    pytest.importorskip("cuda.bindings.runtime")
    source = torch.arange(1024, dtype=torch.int32, pin_memory=True)
    target = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    synchronize = torch.cuda.Stream.synchronize

    def fail_setup(*args, **kwargs):
        raise RuntimeError(f"test {failure} failure")

    def unexpected_sync(*args, **kwargs):
        pytest.fail("保活接管异常应直接传播，不应同步收尾")

    def unexpected_copy(*args, **kwargs):
        pytest.fail("raw batch 已提交后的保护失败不能被 copy_ 回退掩盖")

    if failure == "record":
        monkeypatch.setattr(torch.cuda.Event, "record", fail_setup)
    elif failure == "pool":
        monkeypatch.setattr(cuda_wrapper.PinnedH2DBatchCopier, "_release_pool", None)
        monkeypatch.setattr(cuda_wrapper, "ThreadPoolExecutor", fail_setup)
    else:
        monkeypatch.setattr(cuda_wrapper.PinnedH2DBatchCopier._get_release_pool(), "submit", fail_setup)
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", unexpected_sync)
    monkeypatch.setattr(torch.Tensor, "copy_", unexpected_copy)
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            with pytest.raises(RuntimeError, match=f"test {failure} failure"):
                cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([target], [source])
    finally:
        # 测试捕获注入异常后仍要继续运行，自行持有源并等待在途传输结束。
        synchronize(stream)
    assert cuda_wrapper.PinnedH2DBatchCopier._native_batch_available
    assert torch.equal(target.cpu(), source)


@requires_cuda
def test_pytorch_fallback_needs_no_completion_event_or_pool_task(monkeypatch, release_futures):
    monkeypatch.setattr(cuda_wrapper.PinnedH2DBatchCopier, "_native_batch_available", False)
    source = torch.arange(1024, dtype=torch.int32, pin_memory=True)
    target = torch.empty_like(source, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    copy_sizes = []
    copy = torch.Tensor.copy_

    def unexpected_event(*args, **kwargs):
        pytest.fail("普通 PyTorch copy_ 已保护源 storage，不应建立额外完成事件")

    def record_copy(dst, src, *args, **kwargs):
        copy_sizes.append(src.numel() * src.element_size())
        return copy(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Event", unexpected_event)
    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    with torch.cuda.stream(stream):
        cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([target], [source])
    stream.synchronize()
    assert not cuda_wrapper.PinnedH2DBatchCopier._native_batch_available
    assert release_futures == []
    assert copy_sizes == [source.numel() * source.element_size()]
    assert torch.equal(target.cpu(), source)


@requires_cuda
def test_source_release_pool_is_reused_across_batches(release_futures):
    pytest.importorskip("cuda.bindings.runtime")
    pool = cuda_wrapper.PinnedH2DBatchCopier._get_release_pool()
    source = torch.arange(1024, dtype=torch.int32, pin_memory=True)
    target = torch.empty((2, 1024), dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(stream):
        for row in range(2):
            cuda_wrapper.cuda_memcpy_pinned_h2d_batch_async([target[row]], [source])
    assert cuda_wrapper.PinnedH2DBatchCopier._get_release_pool() is pool
    assert len(release_futures) == 2
    for future in release_futures:
        assert future.result(timeout=10) is None
    assert torch.equal(target.cpu(), source.expand(2, -1))


def test_background_wait_failure_propagates():
    source = torch.arange(8, dtype=torch.int32)
    sources = [source]

    def failed_wait():
        raise RuntimeError("test background wait failure")

    with pytest.raises(RuntimeError, match="test background wait failure"):
        cuda_wrapper.PinnedH2DBatchCopier._wait_and_release_pinned_sources(
            SimpleNamespace(synchronize=failed_wait), sources
        )
