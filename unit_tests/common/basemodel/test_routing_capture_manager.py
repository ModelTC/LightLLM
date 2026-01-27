import pytest
import torch
import numpy as np


def test_moe_layer_counter():
    """Counter increments and resets correctly."""
    from lightllm.common.basemodel.routing_manager import (
        reset_moe_layer_counter,
        get_next_moe_layer_index,
        get_moe_layer_count,
    )

    reset_moe_layer_counter()
    assert get_moe_layer_count() == 0

    assert get_next_moe_layer_index() == 0
    assert get_next_moe_layer_index() == 1
    assert get_next_moe_layer_index() == 2
    assert get_moe_layer_count() == 3

    reset_moe_layer_counter()
    assert get_moe_layer_count() == 0
    assert get_next_moe_layer_index() == 0


class TestRoutingCaptureManager:
    """Tests for the redesigned RoutingCaptureManager."""

    def test_capture_explicit_layer_index(self):
        """Capture stores data at explicit moe_layer_index."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=4,
            topk=8,
            num_experts=64,
            batch_max_tokens=128,
            kv_cache_size=1024,
            enable_overlap=False,
        )

        # Capture at layer 2 (not sequential)
        topk_ids = torch.randint(0, 64, (10, 8), device="cuda")
        manager.capture(moe_layer_index=2, topk_ids=topk_ids)

        # Verify data is at layer 2, not layer 0
        assert torch.equal(manager.gpu_buffer[0, 2, :10, :], topk_ids.to(manager.dtype))

    def test_double_buffer_overlap_mode(self):
        """Double buffer prevents race condition in overlap mode."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=2,
            topk=4,
            num_experts=32,
            batch_max_tokens=64,
            kv_cache_size=256,
            enable_overlap=True,
        )

        # Should have 2 buffer slots
        assert manager.num_slots == 2
        assert manager.gpu_buffer.shape[0] == 2

        # Capture to slot 0 (microbatch_index=0)
        ids_0 = torch.ones((5, 4), dtype=torch.int64, device="cuda")
        manager.capture(moe_layer_index=0, topk_ids=ids_0, microbatch_index=0)

        # Capture to slot 1 (microbatch_index=1)
        ids_1 = torch.ones((5, 4), dtype=torch.int64, device="cuda") * 2
        manager.capture(moe_layer_index=0, topk_ids=ids_1, microbatch_index=1)

        # Both slots have different data
        assert manager.gpu_buffer[0, 0, 0, 0].item() == 1
        assert manager.gpu_buffer[1, 0, 0, 0].item() == 2

    def test_flush_and_extract(self):
        """Flush transfers data to CPU, extract retrieves by mem_index."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=2,
            topk=4,
            num_experts=32,
            batch_max_tokens=64,
            kv_cache_size=256,
            enable_overlap=False,
        )

        # Capture some data (microbatch_index defaults to 0)
        topk_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
        manager.capture(moe_layer_index=0, topk_ids=topk_ids)
        manager.capture(moe_layer_index=1, topk_ids=topk_ids + 10)

        # Flush to mem_indexes 10 and 11
        mem_indexes = torch.tensor([10, 11], device="cuda")
        manager.flush_to_cpu_async(mem_indexes, microbatch_index=0)

        # Extract
        result = manager.extract_for_request(mem_indexes.cpu())

        assert result.shape == (2, 2, 4)  # [layers, tokens, topk]
        assert result[0, 0, 0] == 1
        assert result[1, 0, 0] == 11

    def test_dtype_selection(self):
        """Uses int8 for <=127 experts, int16 otherwise."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        # Small expert count -> int8
        manager_small = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=64,
            batch_max_tokens=32,
            kv_cache_size=128,
            enable_overlap=False,
        )
        assert manager_small.dtype == torch.int8

        # Large expert count -> int16
        manager_large = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=256,
            batch_max_tokens=32,
            kv_cache_size=128,
            enable_overlap=False,
        )
        assert manager_large.dtype == torch.int16
