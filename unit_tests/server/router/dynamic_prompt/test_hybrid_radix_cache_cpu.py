import torch
import pytest


class TestHybridRadixCacheCpuBuffers:
    """Test that HybridRadixCache correctly manages CPU buffer slots."""

    def _make_cpu_manager(self, size=16, layer_num=2):
        from lightllm.common.mamba_cache_mem_manager.cpu_cache_manager import CpuMambaCacheManager

        return CpuMambaCacheManager(
            size=size,
            layer_num=layer_num,
            conv_state_dtype=torch.bfloat16,
            ssm_state_dtype=torch.float32,
            conv_kernel_size=4,
            num_linear_k_heads=4,
            num_linear_v_heads=4,
            head_linear_k_dim=64,
            head_linear_v_dim=32,
        )

    def _make_cache(self, cpu_mgr):
        from unittest.mock import MagicMock
        from lightllm.server.router.dynamic_prompt.hybrid_radix_cache import HybridRadixCache

        kv_mem = MagicMock()
        kv_mem.mamba_cache_mem_manager = cpu_mgr
        kv_mem.can_use_mem_size = 1000
        cache = HybridRadixCache("test_cpu", 100, 0, kv_mem)
        return cache

    def test_add_buffer_uses_cpu_slot(self):
        cpu_mgr = self._make_cpu_manager()
        cache = self._make_cache(cpu_mgr)
        key = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        val = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        _, node = cache.insert(key, val)
        cpu_slot = cpu_mgr.alloc(1)
        cache.add_buffer_idx_to_node(node, cpu_slot[0].item(), is_hotspot=False)
        assert node.buffer_idx == cpu_slot[0].item()
        assert node.buffer_idx < cpu_mgr.size

    def test_evict_buffer_frees_cpu_slot(self):
        cpu_mgr = self._make_cpu_manager(size=4)
        cache = self._make_cache(cpu_mgr)
        slots_used = []
        for i in range(4):
            key = torch.tensor([i * 10 + j for j in range(4)], dtype=torch.int64)
            val = torch.tensor([i * 10 + j + 100 for j in range(4)], dtype=torch.int64)
            _, node = cache.insert(key, val)
            slot = cpu_mgr.alloc(1)
            cache.add_buffer_idx_to_node(node, slot[0].item())
            slots_used.append(slot[0].item())
        assert cpu_mgr.can_use_mem_size == 0
        cache.free_radix_cache_to_get_enough_buffer(1)
        assert cpu_mgr.can_use_mem_size >= 1
