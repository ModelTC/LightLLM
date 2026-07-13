from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel import routing_manager as _routing_mgr
from lightllm.common.basemodel.routing_manager import RoutingCaptureManager


def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for routing capture.")


def test_make_routing_capture_callback_uses_global_manager(monkeypatch):
    calls = []

    class _Manager:
        def make_capture_callback(self, layer_num, mem_indexes):
            calls.append((layer_num, mem_indexes))
            return "callback"

    mem_indexes = object()
    infer_state = SimpleNamespace(mem_index=mem_indexes)
    monkeypatch.setattr(_routing_mgr, "g_routing_capture_manager", _Manager())

    assert _routing_mgr.make_routing_capture_callback(infer_state, 7) == "callback"
    assert calls == [(7, mem_indexes)]


class TestRoutingCaptureManager:
    def test_capture_and_extract_basic(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=4,
            topk=8,
            num_experts=64,
            kv_cache_size=1024,
            max_capture_tokens=64,
        )
        mem_indexes = torch.arange(100, 110, device="cuda")
        expected = np.zeros((10, 4, 8), dtype=np.uint8)

        for layer_idx in range(4):
            topk_ids = torch.randint(0, 64, (10, 8), device="cuda")
            manager.make_capture_callback(layer_idx, mem_indexes)(topk_ids)
            expected[:, layer_idx, :] = topk_ids.cpu().numpy().astype(np.uint8)

        result = manager.extract_routing_data(mem_indexes)
        assert result.shape == (10, 4, 8)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, expected)

    def test_capture_writes_to_correct_kv_positions(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=2,
            topk=4,
            num_experts=32,
            kv_cache_size=256,
            max_capture_tokens=16,
        )
        mem_indexes = torch.tensor([10, 50, 200], device="cuda")
        topk_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], device="cuda")
        topk_ids_layer1 = topk_ids + 20

        manager.make_capture_callback(0, mem_indexes)(topk_ids)
        manager.make_capture_callback(1, mem_indexes)(topk_ids_layer1)

        result = manager.extract_routing_data(mem_indexes)
        np.testing.assert_array_equal(result[:, 0, :], topk_ids.cpu().numpy().astype(np.uint8))
        np.testing.assert_array_equal(result[:, 1, :], topk_ids_layer1.cpu().numpy().astype(np.uint8))

    def test_capture_maps_transformer_layer_num_to_routing_slot(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=2,
            topk=2,
            num_experts=32,
            kv_cache_size=256,
            max_capture_tokens=16,
            layer_num_to_moe_index={3: 0, 7: 1},
        )
        mem_indexes = torch.tensor([10, 11], device="cuda")
        ids_layer3 = torch.tensor([[1, 2], [3, 4]], device="cuda")
        ids_layer7 = torch.tensor([[5, 6], [7, 8]], device="cuda")

        manager.make_capture_callback(3, mem_indexes)(ids_layer3)
        manager.make_capture_callback(7, mem_indexes)(ids_layer7)
        assert manager.make_capture_callback(4, mem_indexes) is None

        result = manager.extract_routing_data(mem_indexes)
        np.testing.assert_array_equal(result[:, 0, :], ids_layer3.cpu().numpy().astype(np.uint8))
        np.testing.assert_array_equal(result[:, 1, :], ids_layer7.cpu().numpy().astype(np.uint8))

    def test_capture_rejects_unexpected_topk_width(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=32,
            kv_cache_size=256,
            max_capture_tokens=16,
        )
        mem_indexes = torch.tensor([10, 11], device="cuda")
        topk_ids_with_shared_expert = torch.tensor([[1, 2, 32], [3, 4, 32]], device="cuda")

        with pytest.raises(AssertionError):
            manager.make_capture_callback(0, mem_indexes)(topk_ids_with_shared_expert)

    def test_cuda_graph_replay_uses_copied_mem_indexes(self):
        _skip_without_cuda()

        class _NoopDecodeState:
            def copy_for_decode_cuda_graph(self, other):
                return

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=32,
            kv_cache_size=256,
            max_capture_tokens=16,
        )
        graph_infer_state = InferStateInfo()
        graph_infer_state.decode_att_state = _NoopDecodeState()
        graph_infer_state.mem_index = torch.tensor([10, 11], device="cuda")
        capture_callback = manager.make_capture_callback(0, graph_infer_state.mem_index)
        topk_ids = torch.tensor([[1, 2], [3, 4]], device="cuda")

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            capture_callback(topk_ids)

        graph.replay()
        result = manager.extract_routing_data(graph_infer_state.mem_index)
        np.testing.assert_array_equal(result[:, 0, :], topk_ids.cpu().numpy().astype(np.uint8))

        new_infer_state = InferStateInfo()
        new_infer_state.decode_att_state = _NoopDecodeState()
        new_infer_state.mem_index = torch.tensor([20, 21], device="cuda")
        new_topk_ids = torch.tensor([[5, 6], [7, 8]], device="cuda")
        graph_infer_state.copy_for_cuda_graph(new_infer_state)
        topk_ids.copy_(new_topk_ids)

        graph.replay()

        result = manager.extract_routing_data(new_infer_state.mem_index)
        np.testing.assert_array_equal(result[:, 0, :], new_topk_ids.cpu().numpy().astype(np.uint8))

    def test_microbatch_isolation(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=4,
            num_experts=32,
            kv_cache_size=256,
            max_capture_tokens=16,
        )
        mem0 = torch.tensor([10, 11], device="cuda")
        mem1 = torch.tensor([20, 21], device="cuda")
        ids_0 = torch.ones((2, 4), dtype=torch.int64, device="cuda")
        ids_1 = torch.ones((2, 4), dtype=torch.int64, device="cuda") * 2

        capture0 = manager.make_capture_callback(0, mem0)
        capture1 = manager.make_capture_callback(0, mem1)
        capture0(ids_0)
        capture1(ids_1)

        result0 = manager.extract_routing_data(mem0)
        result1 = manager.extract_routing_data(mem1)
        assert result0[0, 0, 0] == 1
        assert result1[0, 0, 0] == 2

    def test_dtype_selection_uint8(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=256,
            kv_cache_size=128,
            max_capture_tokens=16,
        )
        assert manager.dtype == torch.uint8
        assert manager.np_dtype == np.uint8
        assert manager.dtype_id == 1

    def test_dtype_selection_int16(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=257,
            kv_cache_size=128,
            max_capture_tokens=16,
        )
        assert manager.dtype == torch.int16
        assert manager.np_dtype == np.int16
        assert manager.dtype_id == 2

    def test_extract_preserves_uint8_values(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=4,
            num_experts=256,
            kv_cache_size=64,
            max_capture_tokens=16,
        )
        mem_indexes = torch.tensor([0, 1, 2], device="cuda")
        topk_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 63, 1], [0, 5, 255, 3]], device="cuda")

        manager.make_capture_callback(0, mem_indexes)(topk_ids)

        result = manager.extract_routing_data(mem_indexes)
        np.testing.assert_array_equal(result[:, 0, :], topk_ids.cpu().numpy().astype(np.uint8))

    def test_routing_buffer_and_pointer_shape(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=48,
            topk=8,
            num_experts=256,
            kv_cache_size=2048,
            max_capture_tokens=256,
        )
        assert manager.routing_buffer.shape == (2048, 48, 8)
        assert manager.routing_buffer.dtype == torch.uint8
        assert manager.routing_buffer.device.type == "cpu"
        assert manager.routing_buffer.is_pinned()
        assert manager.routing_buffer_ptr.shape == (1,)
        assert manager.routing_buffer_ptr.dtype == torch.uint64
        assert manager.routing_buffer_ptr.device.type == "cuda"

    def test_partial_token_capture(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=1,
            topk=2,
            num_experts=32,
            kv_cache_size=128,
            max_capture_tokens=16,
        )
        mem_indexes = torch.tensor([10, 11, 12, 13, 14], device="cuda")
        topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], device="cuda")

        manager.make_capture_callback(0, mem_indexes[:3])(topk_ids)

        result_written = manager.extract_routing_data(mem_indexes[:3])
        np.testing.assert_array_equal(result_written[:, 0, :], topk_ids.cpu().numpy().astype(np.uint8))

        result_unwritten = manager.extract_routing_data(mem_indexes[3:])
        np.testing.assert_array_equal(result_unwritten[:, 0, :], np.zeros((2, 2), dtype=np.uint8))

    def test_capture_does_not_allocate_capture_buffer(self):
        _skip_without_cuda()

        manager = RoutingCaptureManager(
            num_moe_layers=4,
            topk=8,
            num_experts=64,
            kv_cache_size=1024,
            max_capture_tokens=256,
        )
        assert not hasattr(manager, "_capture_buffer")
