import ctypes
import mmap
import threading

import pytest
import torch

from lightllm.utils import host_memory


class CudaCall:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


@pytest.fixture
def runtime(monkeypatch):
    from types import SimpleNamespace

    registered, removed = [], []
    failures = set()
    context = threading.local()

    def set_device(device):
        context.device = device

    def register(base, size, flags):
        assert context.device == 3
        if base.value in failures:
            return 2
        registered.append((base.value, size.value, flags))
        return 0

    def unregister(base):
        assert context.device == 3
        if base.value in failures:
            return 3
        removed.append(base.value)
        return 0

    def device_pointer(out, base, flags):
        assert context.device == 3
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = base
        return 0

    cuda = SimpleNamespace(
        cudaHostRegister=CudaCall(register),
        cudaHostUnregister=CudaCall(unregister),
        cudaHostGetDevicePointer=CudaCall(device_pointer),
    )
    monkeypatch.setattr(host_memory.ctypes, "CDLL", lambda _: cuda)
    monkeypatch.setattr(host_memory, "get_current_device_id", lambda: 3)
    monkeypatch.setattr(host_memory.torch.cuda, "set_device", set_device)
    return cuda, registered, removed, failures


def test_all_original_segment_bases_are_released_once(runtime):
    _, registered, removed, _ = runtime
    registration = host_memory.HostMemoryRegistration(4096, 10240, 4096, workers=3)
    assert sorted(registered) == [(4096, 4096, 3), (8192, 4096, 3), (12288, 2048, 3)]
    assert registration.device_ptr == 4096
    registration.close()
    registration.close()
    assert sorted(removed) == [4096, 8192, 12288]


def test_partial_parallel_registration_failure_rolls_back_all_successes(runtime):
    _, registered, removed, failures = runtime
    failures.add(8192)
    with pytest.raises(RuntimeError, match="cudaHostRegister"):
        host_memory.HostMemoryRegistration(4096, 16384, 4096, workers=4)
    assert {base for base, _, _ in registered} == {4096, 12288, 16384}
    assert set(removed) == {4096, 12288, 16384}


def test_mapping_failure_rolls_back_registrations(runtime):
    cuda, _, removed, _ = runtime
    cuda.cudaHostGetDevicePointer.fn = lambda *args: 7
    with pytest.raises(RuntimeError, match="cudaHostGetDevicePointer"):
        host_memory.HostMemoryRegistration(4096, 8192, 4096, workers=2)
    assert set(removed) == {4096, 8192}


def test_unregister_failure_keeps_only_failed_bases_for_retry(runtime):
    _, _, removed, failures = runtime
    registration = host_memory.HostMemoryRegistration(4096, 8192, 4096, workers=2)
    failures.add(4096)
    with pytest.raises(RuntimeError, match="cudaHostUnregister"):
        registration.close()
    assert registration.bases == [4096]
    assert removed == [8192]
    failures.clear()
    registration.close()
    assert removed == [8192, 4096]


def test_cleanup_progress_requires_a_successful_unregister(runtime):
    _, _, removed, failures = runtime
    registration = host_memory.HostMemoryRegistration(4096, 8192, 4096, workers=2)
    progress = []
    failures.add(4096)
    with pytest.raises(RuntimeError, match="cudaHostUnregister"):
        registration.close(progress=lambda: progress.append(len(removed)))
    assert progress == [1]
    failures.clear()
    registration.close(progress=lambda: progress.append(len(removed)))
    assert progress == [1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_segmented_uva_reads_writes_and_explicit_release(monkeypatch):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    monkeypatch.setattr(host_memory, "get_current_device_id", lambda: torch.cuda.current_device())
    with mmap.mmap(-1, 3 * mmap.PAGESIZE) as mapping:
        tensor = torch.frombuffer(mapping, dtype=torch.uint8)
        tensor.fill_(11)
        registration = host_memory.HostMemoryRegistration(tensor.data_ptr(), tensor.numel(), mmap.PAGESIZE, 2)
        view = get_accelerator_view_from_cpu_tensor(tensor)
        torch.testing.assert_close(view.cuda(), torch.full_like(view, 11))
        view.fill_(29)
        torch.cuda.synchronize()
        assert tensor.tolist() == [29] * tensor.numel()
        del view
        registration.close()
        registration.close()
        assert registration.bases == []
        del tensor
