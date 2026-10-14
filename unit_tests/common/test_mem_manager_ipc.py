from multiprocessing.reduction import ForkingPickler

import pytest
import torch
import torch.multiprocessing as mp

from lightllm.common.kv_cache_mem_manager import allocator as allocator_module
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.operator import NormalMemOperator


class _Counter:
    def __init__(self, *args):
        pass

    def set_value(self, value):
        self.value = value


def _make_manager(monkeypatch, size, device="cpu"):
    monkeypatch.setattr(allocator_module, "SharedInt", _Counter)
    monkeypatch.setattr(allocator_module, "get_current_rank_in_node", lambda: 0)
    manager = MemoryManager.__new__(MemoryManager)
    manager.size = size
    manager.page_size = 1
    manager.allocator = allocator_module.KvCacheAllocator(size)
    manager.kv_buffer = torch.arange(32, dtype=torch.float32, device=device).reshape(1, 4, 2, 4)
    manager.req_to_token_indexs = torch.arange(4, dtype=torch.int32, device=device).reshape(1, 4)
    manager.operator = NormalMemOperator(manager)
    return manager


def test_ipc_descriptor_size_does_not_scale_with_local_allocator(monkeypatch):
    small = _make_manager(monkeypatch, 256)
    large = _make_manager(monkeypatch, 131072)
    small_payload = ForkingPickler.dumps(small)
    large_payload = ForkingPickler.dumps(large)
    print(f"IPC pickle sizes: small={len(small_payload)} large={len(large_payload)}")
    assert len(large_payload) < 4096
    assert abs(len(large_payload) - len(small_payload)) < 128


def test_ipc_roundtrip_retains_shared_buffers_and_local_owner(monkeypatch):
    source = _make_manager(monkeypatch, 256)
    allocator = source.allocator
    exported = ForkingPickler.loads(ForkingPickler.dumps(source))
    assert source.allocator is allocator
    assert allocator.alloc(1).tolist() == [0]
    assert exported.allocator is None
    assert exported.operator.mem_manager is exported
    exported.kv_buffer.fill_(17)
    exported.req_to_token_indexs.fill_(23)
    assert torch.all(source.kv_buffer == 17)
    assert torch.all(source.req_to_token_indexs == 23)


def _update_exported_buffers(payload, connection):
    try:
        exported = ForkingPickler.loads(payload)
        assert exported.allocator is None
        assert exported.operator.mem_manager is exported
        exported.kv_buffer.fill_(17)
        exported.req_to_token_indexs.fill_(23)
        if exported.kv_buffer.is_cuda:
            torch.cuda.synchronize()
        connection.send("complete")
    finally:
        connection.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ipc_buffers_remain_shared_between_processes(monkeypatch, device):
    source = _make_manager(monkeypatch, 256, device)
    allocator = source.allocator
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_update_exported_buffers, args=(ForkingPickler.dumps(source).tobytes(), sender))
    try:
        process.start()
        sender.close()
        assert receiver.poll(60), "IPC consumer did not complete"
        assert receiver.recv() == "complete"
        process.join(10)
        assert process.exitcode == 0
        assert source.allocator is allocator
        assert torch.all(source.kv_buffer == 17)
        assert torch.all(source.req_to_token_indexs == 23)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        receiver.close()
        sender.close()
