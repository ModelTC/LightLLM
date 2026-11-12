from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest
import torch
import torch.multiprocessing as mp

from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNPagesPool, KDAPagesPool
from lightllm.common.attention_state.bounded_att import utils

if TYPE_CHECKING:
    from easydict import EasyDict
    from multiprocessing.queues import Queue


@pytest.mark.parametrize("pool_cls", [KDAPagesPool, GDNPagesPool])
@pytest.mark.parametrize("dp_fetch", [False, True])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_page_pool_allocation_strategy(
    cpu_args: EasyDict,
    pool_cls: type[KDAPagesPool] | type[GDNPagesPool],
    dp_fetch: bool,
    pin_memory: bool,
) -> None:
    if pin_memory and not torch.cuda.is_available():
        pytest.skip("CUDA required for pinned memory")
    cpu_args.enable_dp_prompt_cache_fetch = dp_fetch
    config = ConvSSMStateConfig(
        logical_layer_ids=(0, 2),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=4,
        value_dim=4,
        conv_width=3,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )
    pool = pool_cls(size=2, config=config, pin_memory=pin_memory)
    assert pool.conv_state.shape == (2, 2, 12, 3)
    assert pool.ssm_state.shape == (2, 2, 1, 4, 4)
    assert pool.conv_state.dtype == torch.bfloat16
    assert pool.ssm_state.dtype == torch.float32
    for tensor in (pool.conv_state, pool.ssm_state):
        assert tensor.is_shared() == dp_fetch
        assert tensor.is_pinned() == pin_memory
        tensor.fill_(7)
        assert torch.all(tensor == 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_registration_lives_until_last_view(
    cpu_args: EasyDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_args.enable_dp_prompt_cache_fetch = True
    unregistered = []
    unregister = utils._unregister_page_pool_storage

    def record_unregister(addresses: tuple[int, ...], device_id: int) -> None:
        unregistered.extend(addresses)
        unregister(addresses=addresses, device_id=device_id)

    monkeypatch.setattr(utils, "_unregister_page_pool_storage", record_unregister)
    tensor = utils.create_page_pool_tensor(size=(4096,), dtype=torch.float32)
    ptr = tensor.data_ptr()
    view = tensor[1:]
    del tensor
    gc.collect()
    assert not unregistered
    assert view.is_pinned()
    del view
    gc.collect()
    assert unregistered == [ptr]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_chunked_shared_pages_keep_dma_boundaries_and_lifetime(cpu_args, monkeypatch, dtype) -> None:
    cpu_args.enable_dp_prompt_cache_fetch = True
    monkeypatch.setattr(utils, "_HOST_REGISTRATION_BYTES", 16384)
    runtime = torch.cuda.cudart()
    registered, released = [], []

    class RecordingRuntime:
        def cudaHostRegister(self, address, size, flags):
            status = runtime.cudaHostRegister(address, size, flags)
            if int(status) == 0:
                registered.append((address, size))
            return status

        def cudaHostUnregister(self, address):
            released.append(address)
            return runtime.cudaHostUnregister(address)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: RecordingRuntime())
    tensor = utils.create_page_pool_tensor(size=(7, 3, 512), dtype=dtype)
    cell_bytes = tensor[0].numel() * tensor.element_size()
    assert len(registered) > 1
    assert all((address - tensor.data_ptr()) % cell_bytes == 0 for address, _ in registered)
    assert all(size <= 16384 for _, size in registered)
    destination = torch.empty((3, 5, 512), dtype=dtype, device="cuda")
    for index in range(tensor.shape[0]):
        tensor[index].fill_(index + 1)
        destination[:, 1].copy_(tensor[index], non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(destination[:, 1].cpu(), tensor[index])

    view = tensor[-1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        destination[:, 1].copy_(view, non_blocking=True)
    view.fill_(17)
    del tensor
    gc.collect()
    assert not released
    assert view.is_shared() and view.is_pinned()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(destination[:, 1].cpu(), view)
    graph.reset()
    del view
    gc.collect()
    assert released == [address for address, _ in reversed(registered)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_private_storage_keeps_whole_tensor_dma(monkeypatch) -> None:
    monkeypatch.setattr(utils, "_HOST_REGISTRATION_BYTES", 16384)
    tensor = torch.full((7, 3, 512), 7, dtype=torch.float32)
    utils.register_page_pool_tensor(tensor)
    destination = tensor.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(destination.cpu(), tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_pool_partial_registration_failure_rolls_back(cpu_args, monkeypatch) -> None:
    cpu_args.enable_dp_prompt_cache_fetch = True
    monkeypatch.setattr(utils, "_HOST_REGISTRATION_BYTES", 16384)
    runtime = torch.cuda.cudart()
    registered, released = [], []

    class FailingRuntime:
        def cudaHostRegister(self, address, size, flags):
            if registered:
                return 1
            status = runtime.cudaHostRegister(address, size, flags)
            if int(status) == 0:
                registered.append(address)
            return status

        def cudaHostUnregister(self, address):
            released.append(address)
            return runtime.cudaHostUnregister(address)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: FailingRuntime())
    with pytest.raises(torch.cuda.CudaError):
        utils.create_page_pool_tensor(size=(7, 3, 512), dtype=torch.float32)
    assert registered and released == registered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_page_pool_keeps_device_allocation(cpu_args: EasyDict) -> None:
    cpu_args.enable_dp_prompt_cache_fetch = True
    tensor = utils.create_page_pool_tensor(size=(8,), dtype=torch.float32, device="cuda")
    assert tensor.device.type == "cuda"
    assert not tensor.is_pinned()


def test_empty_shared_pool_skips_cuda_registration(
    cpu_args: EasyDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_args.enable_dp_prompt_cache_fetch = True

    def unexpected_cuda_init() -> None:
        pytest.fail("empty shared storage must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_cuda_init)
    tensor = utils.create_page_pool_tensor(size=(0, 4), dtype=torch.float32)
    assert tensor.shape == (0, 4)
    assert tensor.is_shared()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("pool_cls", [KDAPagesPool, GDNPagesPool])
def test_pinned_page_pool_survives_queue_on_another_gpu(
    cpu_args: EasyDict,
    pool_cls: type[KDAPagesPool] | type[GDNPagesPool],
) -> None:
    cpu_args.enable_dp_prompt_cache_fetch = True
    torch.cuda.set_device(0)
    config = ConvSSMStateConfig(
        logical_layer_ids=(0, 2),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=4,
        value_dim=4,
        conv_width=3,
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
    )
    pool = pool_cls(size=2, config=config)
    tensors = (pool.conv_state, pool.ssm_state)
    pointers = [tensor.data_ptr() for tensor in tensors]
    for tensor in tensors:
        values = torch.arange(tensor.numel(), device="cuda", dtype=tensor.dtype).reshape(tensor.shape)
        tensor.copy_(values, non_blocking=True)
    torch.cuda.synchronize()
    # 创建端也允许重复调用，不得重复注册或复制存储。
    pool.register_pin_memory()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    result = ctx.Queue()
    worker = ctx.Process(target=_read_page_pool_on_other_gpu, args=(queue, result))
    worker.start()
    try:
        queue.put(pool)
        assert result.get(timeout=45) == [(True, False, True), (True, False, True)]
        worker.join(timeout=15)
        assert worker.exitcode == 0
        for tensor, ptr in zip(tensors, pointers):
            assert tensor.data_ptr() == ptr
            assert tensor.is_shared() and tensor.is_pinned()
            assert tensor.flatten()[0].item() == 1234
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=5)
        queue.close()
        result.close()


def _read_page_pool_on_other_gpu(queue: Queue, result: Queue) -> None:
    torch.cuda.set_device(1)
    pool = queue.get(timeout=30)
    tensors = (pool.conv_state, pool.ssm_state)
    was_pinned = [tensor.is_pinned() for tensor in tensors]
    pointers = [tensor.data_ptr() for tensor in tensors]
    pool.register_pin_memory()
    pool.register_pin_memory()
    states = []
    for tensor, ptr, before in zip(tensors, pointers, was_pinned):
        assert tensor.data_ptr() == ptr
        values = tensor.to(device="cuda", non_blocking=True)
        torch.cuda.synchronize()
        expected = torch.arange(tensor.numel(), device="cuda", dtype=tensor.dtype).reshape(tensor.shape)
        assert torch.equal(values, expected)
        tensor.flatten()[0] = 1234
        states.append((tensor.is_shared(), before, tensor.is_pinned()))
    result.put(states)
