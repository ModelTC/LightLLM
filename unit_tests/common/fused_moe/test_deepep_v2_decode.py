from types import SimpleNamespace

import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as moe
from lightllm.distributed import communication_op
from lightllm.server.core.objs.start_args_type import StartArgs


def test_v2_dispatch_keeps_collective_capacity_and_compute_stream_ownership(monkeypatch):
    calls = []
    native_handle = SimpleNamespace(psum_num_recv_tokens_per_expert=object())
    recv_weights = object()
    quantized, received, repacked, counts = object(), object(), object(), object()

    def dispatch(x, **kwargs):
        assert x is quantized
        assert kwargs["num_max_tokens_per_rank"] == 384
        assert kwargs["do_cpu_sync"] is False and kwargs["do_expand"] is True
        assert kwargs["async_with_compute_stream"] is False
        assert kwargs["allocate_on_comm_stream"] is False
        calls.append("dispatch")
        return received, None, recv_weights, native_handle, None

    def repack(x, psum, max_m):
        assert calls == ["dispatch"]
        assert x is received and psum is native_handle.psum_num_recv_tokens_per_expert
        assert max_m == 6144
        return repacked, counts

    monkeypatch.setattr(moe.dist_group_manager, "ep_buffer", SimpleNamespace(dispatch=dispatch, num_ranks=16))
    monkeypatch.setattr(moe, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 384)
    monkeypatch.setattr(moe, "per_token_group_quant_fp8", lambda x, block: quantized)
    monkeypatch.setattr(moe, "ep_expanded_to_masked", repack)
    x = torch.empty(3, 5120)
    result = moe.deepep_v2_decode_dispatch(x, object(), object(), 384)
    assert result == (repacked, counts, (native_handle, recv_weights))


def test_v2_combine_passes_router_weights_to_repack_once(monkeypatch):
    calls = []
    native_handle = SimpleNamespace(psum_num_recv_tokens_per_expert=object())
    weights, gemm, expanded, output = object(), object(), object(), object()

    def repack(x, psum, recv_weights):
        assert x is gemm and recv_weights is weights
        assert psum is native_handle.psum_num_recv_tokens_per_expert
        calls.append("weighted_repack")
        return expanded

    def combine(x, handle, **kwargs):
        assert x is expanded and handle is native_handle
        assert kwargs["topk_weights"] is None
        assert kwargs["async_with_compute_stream"] is False
        assert kwargs["allocate_on_comm_stream"] is False
        calls.append("combine")
        return output, None, None

    monkeypatch.setattr(moe, "ep_masked_to_expanded", repack)
    monkeypatch.setattr(moe.dist_group_manager, "ep_buffer", SimpleNamespace(combine=combine))
    actual, hook = moe.deepep_v2_decode_combine(gemm, (native_handle, weights))
    assert actual is output and calls == ["weighted_repack", "combine"]
    hook()
    assert calls == ["weighted_repack", "combine"]


def test_v2_initialization_does_not_allocate_legacy_rdma_buffer(monkeypatch):
    args = StartArgs()
    args.enable_ep_moe = True
    args.ep_moe_backend = "deepep_v2"
    args.run_mode = "normal"
    args.data_type = "bfloat16"
    allocations = []

    class ElasticBuffer:
        def __init__(self, *args, **kwargs):
            allocations.append(kwargs)

        def get_theoretical_num_sms(self, *args):
            return 14

    class LegacyBuffer:
        @staticmethod
        def get_low_latency_rdma_size_hint(*args):
            return 4096

        def __init__(self, *args, **kwargs):
            raise AssertionError("v2 must not initialize the NVSHMEM legacy buffer")

    real_empty = torch.empty
    monkeypatch.setattr(communication_op, "get_env_start_args", lambda: args)
    monkeypatch.setattr(communication_op, "HAS_DEEPEP", True)
    monkeypatch.setattr(
        communication_op, "deep_ep", SimpleNamespace(ElasticBuffer=ElasticBuffer, Buffer=LegacyBuffer), raising=False
    )
    monkeypatch.setattr(communication_op, "get_global_world_size", lambda: 16)
    monkeypatch.setattr(communication_op, "get_deepep_num_max_dispatch_tokens_per_rank_prefill", lambda: 8192)
    monkeypatch.setattr(communication_op, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 384)
    monkeypatch.setattr(communication_op, "get_deep_ep_prefill_moe_workspace_size", lambda **kwargs: 4096)
    monkeypatch.setattr(communication_op, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(communication_op.dist, "new_group", lambda *args: object())
    monkeypatch.setattr(communication_op.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(communication_op.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(communication_op.torch, "empty", lambda shape, dtype, device: real_empty(shape, dtype=dtype))
    manager = communication_op.DistributeGroupManager()
    manager.groups = [object()]
    monkeypatch.setattr(manager, "_set_num_sms_for_deep_gemm", lambda *args: None)
    manager.new_deepep_group(384, 5120, {"fp8w8a8-b128-deepgemm"}, 6, 2304)
    assert len(allocations) == 1
    assert manager.ep_decode_use_v2
    assert manager.ep_low_latency_buffer is None
    assert manager.ep_prefill_moe_workspace.numel() == 4096
