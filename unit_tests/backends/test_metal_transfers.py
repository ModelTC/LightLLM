"""Private host snapshot lifetime across mutable and concurrent producers."""

from concurrent.futures import ThreadPoolExecutor
import weakref

import numpy as np
import pytest
import torch

from lightllm.backends.metal.runtime import MetalRuntime
from lightllm.backends.metal.transfers import HostTransferQueue


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")


def _delay_gpu():
    # Queue sufficient work to expose source reuse before a transfer completes.
    x = torch.ones((768, 768), device="mps") * 0.001
    for _ in range(24):
        x = torch.mm(x, x)
    return x


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float16, torch.float32, torch.bool])
@pytest.mark.parametrize("strided", [False, True])
def test_transfer_keeps_private_snapshot_after_shared_numpy_source_reused(dtype, strided):
    runtime = MetalRuntime()
    source = torch.arange(513).to(dtype)
    shared = source.numpy()
    view = torch.from_numpy(shared)[1::2] if strided else torch.from_numpy(shared)
    expected = view.clone()
    delayed = _delay_gpu()
    result = runtime.move(view, non_blocking=True)
    shared[...] = False if dtype == torch.bool else -9
    del view, source, shared
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    runtime.synchronize()
    assert not runtime._host_transfers._pending
    assert runtime._host_transfers._bytes == 0
    assert delayed.shape == (768, 768)


def test_transfer_retains_snapshots_until_event_and_bounds_pending_memory():
    queue = HostTransferQueue(max_pending=3, max_bytes=192)
    delayed = _delay_gpu()
    source = torch.arange(16, dtype=torch.int32)
    expected = source.clone()
    results = []
    refs = []
    for i in range(200):
        source.copy_(expected + i)
        result = queue.move(source)
        refs.append(weakref.ref(queue._pending[-1][1]))
        assert refs[-1]() is not None
        results.append(result)
        source.fill_(-1)
        assert len(queue._pending) <= 3
        assert queue._bytes <= 192
        assert queue._bytes == sum(item[2] for item in queue._pending)
    queue.synchronize()
    for i, result in enumerate(results):
        torch.testing.assert_close(result.cpu(), expected + i, atol=0, rtol=0)
    assert all(ref() is None for ref in refs)
    assert not queue._pending and queue._bytes == 0
    assert delayed.shape == (768, 768)


def test_concurrent_producers_and_synchronization_do_not_release_pending_sources():
    runtime = MetalRuntime()

    def produce(worker):
        source = torch.empty(33, dtype=torch.int64)
        outputs = []
        for i in range(100):
            expected = worker * 1000 + i
            source.fill_(expected)
            outputs.append((runtime.move(source, non_blocking=True), expected))
            source.fill_(-7)
            if i % 7 == 0:
                runtime.current_stream().synchronize()
        return outputs

    delayed = _delay_gpu()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(produce, range(4)))
    runtime.synchronize()
    for worker_outputs in outputs:
        for result, expected in worker_outputs:
            assert torch.equal(result.cpu(), torch.full((33,), expected, dtype=torch.int64))
    assert not runtime._host_transfers._pending
    assert delayed.shape == (768, 768)


def test_synchronous_large_empty_scalar_autograd_and_device_inputs():
    runtime = MetalRuntime()
    source = torch.arange(16, dtype=torch.float32)
    expected = source.clone()
    result = runtime.move(source)
    source.fill_(-1)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    assert not runtime._host_transfers._pending
    queue = HostTransferQueue(max_bytes=8)
    assert queue.move(source) is None
    assert queue.move(torch.empty(0)) is None
    assert queue.move(torch.ones(1, requires_grad=True)) is None
    assert queue.move(result) is None
    scalar = torch.tensor(17)
    moved = runtime.move(scalar, non_blocking=True)
    scalar.fill_(-3)
    assert moved.cpu().item() == 17
    runtime.empty_cache()
    assert not runtime._host_transfers._pending


def test_transfer_preserves_channels_last_format_and_noncontiguous_storage():
    runtime = MetalRuntime()
    source = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5).transpose(2, 3)
    expected = source.clone()
    result = runtime.move(source, non_blocking=True, memory_format=torch.channels_last)
    source.fill_(-1)
    assert result.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    runtime.synchronize()
