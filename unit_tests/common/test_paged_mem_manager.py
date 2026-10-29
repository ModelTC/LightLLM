from types import SimpleNamespace
import pickle

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.common.kv_cache_mem_manager.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.common.kv_cache_mem_manager.ppl_int4kv_mem_manager import PPLINT4KVMemoryManager
from lightllm.common.kv_cache_mem_manager.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager


def test_shared_memory_exports_preserve_variable_pickle_lengths(monkeypatch, tmp_path):
    from lightllm.common.kv_cache_mem_manager import mem_manager as module

    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", f"{tmp_path.parent.name}_{tmp_path.name}")
    monkeypatch.setattr(module, "get_node_world_size", lambda: 4)
    monkeypatch.setattr(module, "get_current_rank_in_node", lambda: 0)
    monkeypatch.setattr(module, "kv_trans_use_p2p", lambda: False)
    # CUDA IPC ref-counter offsets can cross pickle's one-byte integer boundary
    # between exports. Each consumer still needs its own reference-counter token.
    offsets = [254, 255, 256, 257, 258, 259, 260, 261]
    payloads = [pickle.dumps({"ref_counter_offset": offset}) for offset in offsets]
    serialized = iter(payloads)
    monkeypatch.setattr(module.ForkingPickler, "dumps", lambda obj: memoryview(next(serialized)))
    manager = MemoryManager.__new__(MemoryManager)
    try:
        manager.write_to_shm(SimpleNamespace(req_to_token_indexs=None))
        for offset in reversed(offsets):
            assert manager.loads_from_shm(0) == {"ref_counter_offset": offset}
        with pytest.raises(AssertionError):
            manager.loads_from_shm(0)
    finally:
        shm = module.create_or_link_shm("mem_manager_0", -1, force_mode="link")
        shm.close()
        shm.unlink()


def test_kv_cache_allocation_requires_complete_pages():
    manager = MemoryManager.__new__(MemoryManager)
    manager.page_size = 4
    manager.allocator = SimpleNamespace(alloc=lambda size: torch.arange(size, dtype=torch.int32))

    with pytest.raises(AssertionError, match="must be a multiple of page_size 4"):
        manager.alloc(5)

    assert manager.alloc(8).tolist() == list(range(8))


@pytest.mark.parametrize(
    "manager_class",
    [MemoryManager, Deepseek2MemoryManager, PPLINT4KVMemoryManager, PPLINT8KVMemoryManager],
)
def test_kv_cache_buffer_size_requires_complete_pages(manager_class):
    manager = manager_class.__new__(manager_class)
    manager.page_size = 4

    with pytest.raises(AssertionError, match="KV cache size 5 must be a multiple of page_size 4"):
        manager._init_buffers(size=5, dtype=torch.float16, head_num=1, head_dim=8, layer_num=1)
