import torch
import numpy as np


class TestRoutingCaptureManager:
    def test_capture_explicit_layer_index(self):
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=4,
            topk=8,
            num_experts=64,
            batch_max_tokens=128,
            kv_cache_size=1024,
            enable_overlap=False,
        )

        topk_ids = torch.randint(0, 64, (10, 8), device="cuda")
        manager.capture(moe_layer_index=2, topk_ids=topk_ids)

        assert torch.equal(manager.gpu_buffer[0, 2, :10, :], topk_ids.to(manager.dtype))

    def test_double_buffer_overlap_mode(self):
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=2,
            topk=4,
            num_experts=32,
            batch_max_tokens=64,
            kv_cache_size=256,
            enable_overlap=True,
        )

        assert manager.num_slots == 2
        assert manager.gpu_buffer.shape[0] == 2

        ids_0 = torch.ones((5, 4), dtype=torch.int64, device="cuda")
        manager.capture(moe_layer_index=0, topk_ids=ids_0, microbatch_index=0)

        ids_1 = torch.ones((5, 4), dtype=torch.int64, device="cuda") * 2
        manager.capture(moe_layer_index=0, topk_ids=ids_1, microbatch_index=1)

        assert manager.gpu_buffer[0, 0, 0, 0].item() == 1
        assert manager.gpu_buffer[1, 0, 0, 0].item() == 2

    def test_flush_and_extract(self):
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager = RoutingCaptureManager(
            num_moe_layers=2,
            topk=4,
            num_experts=32,
            batch_max_tokens=64,
            kv_cache_size=256,
            enable_overlap=False,
        )

        topk_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
        manager.capture(moe_layer_index=0, topk_ids=topk_ids)
        manager.capture(moe_layer_index=1, topk_ids=topk_ids + 10)

        mem_indexes = torch.tensor([10, 11], device="cuda")
        manager.flush_to_cpu_async(mem_indexes, microbatch_index=0)

        result = manager.extract_for_request(mem_indexes.cpu())

        assert result.shape == (2, 2, 4)
        assert result[0, 0, 0] == 1
        assert result[1, 0, 0] == 11

    def test_dtype_selection(self):
        from lightllm.common.basemodel.routing_manager import RoutingCaptureManager

        manager_small = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=64,
            batch_max_tokens=32,
            kv_cache_size=128,
            enable_overlap=False,
        )
        assert manager_small.dtype == torch.int8

        manager_large = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=256,
            batch_max_tokens=32,
            kv_cache_size=128,
            enable_overlap=False,
        )
        assert manager_large.dtype == torch.int16
