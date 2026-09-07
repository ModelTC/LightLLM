import pytest

from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
    DeepseekV4MemoryManager,
)


def _manager(ratio=0.1, mtp_step=0, nc4=21, nc128=20):
    manager = DeepseekV4MemoryManager.__new__(DeepseekV4MemoryManager)
    manager.layer_num = 46
    manager.n_c4 = nc4
    manager.n_c128 = nc128
    manager.head_dim = 512
    manager.indexer_head_dim = 128
    manager.mla_scale_bytes = 8
    manager.c4_state_ring = 8 + mtp_step
    manager.c128_state_ring = ((128 + mtp_step + 3) // 4) * 4
    manager.max_request_num = 256
    manager.swa_full_tokens_ratio = ratio
    return manager


def test_dsv4_exact_profile_payload_known_prefill_layout():
    assert _manager().get_kv_memory_size(1622608) == 15597055852


@pytest.mark.parametrize("ratio", [0.05, 0.1])
@pytest.mark.parametrize("mtp_step", [0, 4])
@pytest.mark.parametrize("target", [127, 128, 255, 256, 257, 8192, 123456])
def test_dsv4_exact_profile_selects_maximum_token_count(ratio, mtp_step, target):
    manager = _manager(ratio, mtp_step)
    budget = manager.get_kv_memory_size(target)
    assert manager.get_profiled_size(budget - manager.get_fixed_memory_size()) == target
    assert manager.get_kv_memory_size(target + 1) > budget


def test_dsv4_ratio_and_mtp_rings_change_exact_payload():
    assert _manager(0.05, 0).get_kv_memory_size(8192) < _manager(0.1, 0).get_kv_memory_size(8192)
    assert _manager(0.1, 4).get_kv_memory_size(8192) > _manager(0.1, 0).get_kv_memory_size(8192)


@pytest.mark.parametrize("nc4,nc128", [(0, 0), (21, 0), (0, 20), (21, 20)])
def test_dsv4_pool_presence_keeps_exact_boundary(nc4, nc128):
    manager = _manager(nc4=nc4, nc128=nc128)
    budget = manager.get_kv_memory_size(257)
    assert manager.get_profiled_size(budget - manager.get_fixed_memory_size()) == 257
    assert manager.get_kv_memory_size(258) > budget


def test_dsv4_fixed_payload_over_budget_rejected():
    manager = _manager()
    with pytest.raises(RuntimeError):
        manager.get_profiled_size(manager.get_kv_memory_size(0) - 1 - manager.get_fixed_memory_size())


import pytest
