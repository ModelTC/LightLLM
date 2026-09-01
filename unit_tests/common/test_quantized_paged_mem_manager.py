from types import SimpleNamespace

import torch

from lightllm.common.kv_cache_mem_manager.ppl_int4kv_mem_manager import PPLINT4KVMemoryManager
from lightllm.common.kv_cache_mem_manager.ppl_int8kv_mem_manager import PPLINT8KVMemoryManager


def test_quantized_kv_buffers_include_the_full_hold_page(monkeypatch):
    allocations = []

    def fake_empty(shape, **kwargs):
        allocations.append(shape)
        return SimpleNamespace(shape=shape)

    monkeypatch.setattr(torch, "empty", fake_empty)

    int8_manager = PPLINT8KVMemoryManager.__new__(PPLINT8KVMemoryManager)
    int8_manager.page_size = 4
    int8_manager.group_quant_size = 8
    int8_manager._init_buffers(size=32, dtype=torch.float16, head_num=2, head_dim=16, layer_num=3)

    int4_manager = PPLINT4KVMemoryManager.__new__(PPLINT4KVMemoryManager)
    int4_manager.page_size = 4
    int4_manager.group_quant_size = 8
    int4_manager._init_buffers(size=32, dtype=torch.float16, head_num=2, head_dim=16, layer_num=3)

    assert allocations == [
        (3, 36, 4, 16),
        (3, 36, 4, 2),
        (3, 36, 4, 8),
        (3, 36, 4, 2),
    ]
