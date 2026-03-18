import torch
import numpy as np


class TestRoutingCaptureManager:
    def test_capture_and_extract_basic(self):
        """Test the core pipeline: capture → flush_to_kv_buffer → extract_from_gpu."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=4,
            topk=8,
            num_experts=64,
            kv_cache_size=1024,
            max_capture_tokens=64,
        )

        # Simulate a batch of 10 tokens at KV-cache positions [100..109]
        mem_indexes = torch.arange(100, 110, device="cuda")

        # Capture routing for each MoE layer (writes to capture buffer)
        for layer_idx in range(4):
            topk_ids = torch.randint(0, 64, (10, 8), device="cuda")
            manager.capture(moe_layer_index=layer_idx, topk_ids=topk_ids, microbatch_index=0)

        # Flush from capture buffer to KV-indexed gpu_kv_buffer
        manager.flush_to_kv_buffer(mem_indexes, num_tokens=10, microbatch_index=0)

        # Extract for those same KV-cache positions
        result = manager.extract_from_gpu(mem_indexes)
        assert result.shape == (4, 10, 8)
        assert result.dtype == np.int8

    def test_capture_writes_to_correct_kv_positions(self):
        """Verify that captured data lands in the right KV-cache positions after flush."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=2,
            topk=4,
            num_experts=32,
            kv_cache_size=256,
            max_capture_tokens=16,
        )

        # Use non-contiguous mem_indexes to simulate real KV-cache
        mem_indexes = torch.tensor([10, 50, 200], device="cuda")

        topk_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], device="cuda")
        manager.capture(moe_layer_index=0, topk_ids=topk_ids, microbatch_index=0)

        topk_ids_layer1 = topk_ids + 20
        manager.capture(moe_layer_index=1, topk_ids=topk_ids_layer1, microbatch_index=0)

        # Flush to KV positions
        manager.flush_to_kv_buffer(mem_indexes, num_tokens=3, microbatch_index=0)

        # Extract and verify
        result = manager.extract_from_gpu(mem_indexes)
        assert result.shape == (2, 3, 4)
        np.testing.assert_array_equal(result[0], topk_ids.cpu().numpy())
        np.testing.assert_array_equal(result[1], topk_ids_layer1.cpu().numpy())

    def test_microbatch_isolation(self):
        """Two microbatches writing to different KV positions don't interfere."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=4,
            num_experts=32,
            kv_cache_size=256,
            max_capture_tokens=16,
        )

        # Microbatch 0: positions [10, 11]
        mem0 = torch.tensor([10, 11], device="cuda")
        ids_0 = torch.ones((2, 4), dtype=torch.int64, device="cuda")
        manager.capture(moe_layer_index=0, topk_ids=ids_0, microbatch_index=0)

        # Microbatch 1: positions [20, 21]
        mem1 = torch.tensor([20, 21], device="cuda")
        ids_1 = torch.ones((2, 4), dtype=torch.int64, device="cuda") * 2
        manager.capture(moe_layer_index=0, topk_ids=ids_1, microbatch_index=1)

        # Flush each microbatch to different KV positions
        manager.flush_to_kv_buffer(mem0, num_tokens=2, microbatch_index=0)
        manager.flush_to_kv_buffer(mem1, num_tokens=2, microbatch_index=1)

        # Extract microbatch 0
        result0 = manager.extract_from_gpu(mem0)
        assert result0.shape == (1, 2, 4)
        assert result0[0, 0, 0] == 1

        # Extract microbatch 1
        result1 = manager.extract_from_gpu(mem1)
        assert result1[0, 0, 0] == 2

    def test_dtype_selection_int8(self):
        """Models with ≤127 experts use int8."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=64,
            kv_cache_size=128,
            max_capture_tokens=16,
        )
        assert manager.dtype == torch.int8
        assert manager.np_dtype == np.int8
        assert manager.dtype_id == 1

    def test_dtype_selection_int16(self):
        """Models with >127 experts use int16."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=256,
            kv_cache_size=128,
            max_capture_tokens=16,
        )
        assert manager.dtype == torch.int16
        assert manager.np_dtype == np.int16
        assert manager.dtype_id == 2

    def test_extract_preserves_values(self):
        """Extracted values exactly match what was captured, no dtype truncation."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=4,
            num_experts=64,
            kv_cache_size=64,
            max_capture_tokens=16,
        )

        mem_indexes = torch.tensor([0, 1, 2], device="cuda")

        topk_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 63, 1], [0, 5, 127, 3]], device="cuda")
        manager.capture(moe_layer_index=0, topk_ids=topk_ids, microbatch_index=0)

        # Flush then extract
        manager.flush_to_kv_buffer(mem_indexes, num_tokens=3, microbatch_index=0)
        result = manager.extract_from_gpu(mem_indexes)
        expected = topk_ids.cpu().numpy().astype(np.int8)
        np.testing.assert_array_equal(result[0], expected)

    def test_gpu_kv_buffer_shape(self):
        """Buffer shape is (num_moe_layers, kv_cache_size, topk)."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        # 127 experts fits in int8 (max value 127)
        manager = RoutingCaptureManager(
            num_moe_layers=48,
            topk=8,
            num_experts=127,
            kv_cache_size=2048,
            max_capture_tokens=256,
        )
        assert manager.gpu_kv_buffer.shape == (48, 2048, 8)
        assert manager.gpu_kv_buffer.dtype == torch.int8
        assert manager.gpu_kv_buffer.device.type == "cuda"

        # 128 experts requires int16
        manager2 = RoutingCaptureManager(
            num_moe_layers=48,
            topk=8,
            num_experts=128,
            kv_cache_size=2048,
            max_capture_tokens=256,
        )
        assert manager2.gpu_kv_buffer.dtype == torch.int16

    def test_partial_token_capture(self):
        """capture() only writes num_tokens rows to the buffer."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=32,
            kv_cache_size=128,
            max_capture_tokens=16,
        )

        # Capture only 3 tokens, flush to 5 KV positions (first 3 get data)
        mem_indexes = torch.tensor([10, 11, 12, 13, 14], device="cuda")

        topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], device="cuda")  # only 3 tokens
        manager.capture(moe_layer_index=0, topk_ids=topk_ids, microbatch_index=0)

        # Flush only the 3 captured tokens
        manager.flush_to_kv_buffer(mem_indexes[:3], num_tokens=3, microbatch_index=0)

        # Positions 10-12 should have data, 13-14 should be zeros (from init)
        result_written = manager.extract_from_gpu(mem_indexes[:3])
        np.testing.assert_array_equal(result_written[0], topk_ids.cpu().numpy().astype(np.int8))

        result_unwritten = manager.extract_from_gpu(mem_indexes[3:])
        np.testing.assert_array_equal(result_unwritten[0], np.zeros((2, 2), dtype=np.int8))

    def test_capture_buffer_shape(self):
        """Capture buffer has correct shape (max_tokens, num_moe_layers, topk)."""
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=4,
            topk=8,
            num_experts=64,
            kv_cache_size=1024,
            max_capture_tokens=256,
        )
        assert manager._capture_buffer[0].shape == (256, 4, 8)
        assert manager._capture_buffer[1].shape == (256, 4, 8)
        assert manager._capture_buffer[0].dtype == torch.int8
