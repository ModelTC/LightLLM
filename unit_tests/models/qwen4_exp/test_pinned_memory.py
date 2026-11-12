import gc
import weakref

import pytest
import torch
import triton
import triton.language as tl

from lightllm.models.qwen4_exp import pinned_memory

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@triton.jit
def read_ends(Source, Output, SIZE: tl.constexpr):
    row = tl.arange(0, 2)
    position = tl.where(row == 0, 0, SIZE - 1)
    tl.store(Output + row, tl.load(Source + position))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_registered_snapshot_views_graph_and_lifetime(dtype, monkeypatch):
    monkeypatch.setattr(pinned_memory, "_LARGE_BYTES", 1)
    monkeypatch.setattr(pinned_memory, "_CHUNK_BYTES", 4096)
    value = pinned_memory.pinned_zeros((6144,), dtype=dtype)
    assert value.is_pinned() and value.is_contiguous()
    assert torch.count_nonzero(value) == 0
    value[0], value[-1] = 3, 7
    output = torch.empty(2, device="cuda", dtype=torch.float32)
    read_ends[(1,)](value, output, value.numel())
    torch.testing.assert_close(output, torch.tensor([3, 7], device="cuda", dtype=torch.float32), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        read_ends[(1,)](value, output, value.numel())
    value[-1] = -7
    graph.replay()
    torch.testing.assert_close(output, torch.tensor([3, -7], device="cuda", dtype=torch.float32), atol=0, rtol=0)
    graph.reset()
    owner = weakref.ref(value._qwen4_pinned_allocation)
    view = value[4096:]
    del value
    gc.collect()
    assert owner() is not None and view.is_pinned()
    read_ends[(1,)](view, output, view.numel())
    torch.testing.assert_close(output, torch.tensor([0, -7], device="cuda", dtype=torch.float32), atol=0, rtol=0)
    del view
    gc.collect()
    assert owner() is None


def test_partial_registration_failure_releases_registered_ranges(monkeypatch):
    monkeypatch.setattr(pinned_memory, "_LARGE_BYTES", 1)
    monkeypatch.setattr(pinned_memory, "_CHUNK_BYTES", 4096)
    runtime = torch.cuda.cudart()
    registered, released = [], []

    class FailingRuntime:
        def cudaHostRegister(self, address, size, flags):
            if registered:
                return 1
            status = runtime.cudaHostRegister(address, size, flags)
            assert int(status) == 0
            registered.append(address)
            return status

        def cudaHostUnregister(self, address):
            released.append(address)
            return runtime.cudaHostUnregister(address)

        def cudaGetErrorString(self, status):
            return "intentional registration failure"

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FailingRuntime())

    def allocate():
        try:
            pinned_memory.pinned_zeros((6144,), dtype=torch.float32)
        except RuntimeError as error:
            assert "intentional registration failure" in str(error)
        else:
            pytest.fail("Expected registration failure")

    allocate()
    gc.collect()
    assert registered and released == registered


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_snapshot_dma_restore_never_crosses_registration(dtype, monkeypatch):
    monkeypatch.setattr(pinned_memory, "_LARGE_BYTES", 1)
    monkeypatch.setattr(pinned_memory, "_CHUNK_BYTES", 16384)
    value = pinned_memory.pinned_zeros((7, 3, 512), dtype=dtype)
    cell_bytes = value[0].numel() * value.element_size()
    starts = value._qwen4_pinned_allocation.addresses
    assert len(starts) > 1
    assert all((address - value.data_ptr()) % cell_bytes == 0 for address in starts)
    ends = starts[1:] + [value.data_ptr() + value.numel() * value.element_size()]
    assert all(end - start <= 16384 for start, end in zip(starts, ends))
    destination = torch.empty((3, 5, 512), dtype=dtype, device="cuda")
    for index in range(value.shape[0]):
        value[index].fill_(index + 1)

        destination[:, 1, ...] = value[index]
        torch.cuda.synchronize()
        torch.testing.assert_close(destination[:, 1].cpu(), value[index], atol=0, rtol=0)


def test_snapshot_larger_than_registration_bound_is_rejected(monkeypatch):
    monkeypatch.setattr(pinned_memory, "_LARGE_BYTES", 1)
    monkeypatch.setattr(pinned_memory, "_CHUNK_BYTES", 16384)
    with pytest.raises(ValueError, match="one bounded host registration"):
        pinned_memory.pinned_zeros((2, 8192), dtype=torch.float32)
