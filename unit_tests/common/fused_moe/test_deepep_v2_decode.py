from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as moe
from lightllm.distributed import communication_op
from lightllm.server.core.objs.start_args_type import StartArgs


def test_v2_dispatch_keeps_collective_capacity_and_compute_stream_ownership(monkeypatch):
    calls = []
    native_handle = SimpleNamespace(psum_num_recv_tokens_per_expert=object())
    recv_weights = object()
    quantized, received = object(), object()

    def dispatch(x, **kwargs):
        assert x is quantized
        assert kwargs["num_max_tokens_per_rank"] == 384
        assert kwargs["do_cpu_sync"] is False and kwargs["do_expand"] is True
        assert kwargs["do_handle_copy"] is False
        assert kwargs["async_with_compute_stream"] is False
        assert kwargs["allocate_on_comm_stream"] is False
        assert kwargs["do_zero_padding"] and kwargs["use_tma_aligned_col_major_sf"]
        assert kwargs["expert_alignment"] == 128
        calls.append("dispatch")
        return received, None, recv_weights, native_handle, None

    monkeypatch.setattr(moe.dist_group_manager, "ep_buffer", SimpleNamespace(dispatch=dispatch, num_ranks=16))
    monkeypatch.setattr(moe, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 384)
    monkeypatch.setattr(moe, "per_token_group_quant_fp8", lambda x, block: quantized)
    monkeypatch.setattr(
        moe, "deep_gemm", SimpleNamespace(get_mk_alignment_for_contiguous_layout=lambda: 128), raising=False
    )
    x = torch.empty(3, 5120)
    result = moe.deepep_v2_decode_dispatch(x, object(), object(), 384)
    assert calls == ["dispatch"]
    assert result == (received, native_handle.psum_num_recv_tokens_per_expert, (native_handle, recv_weights))


def test_v2_combine_uses_pre_reduced_output_and_compact_metadata(monkeypatch):
    calls = []
    original_metadata, compact_metadata = object(), object()
    native_handle = SimpleNamespace(
        recv_src_metadata=original_metadata, psum_num_recv_tokens_per_scaleup_rank=torch.tensor([3, 5])
    )
    weights, gemm, dense, output = object(), object(), object(), object()

    def reduce(x, recv_weights, metadata, count):
        assert x is gemm and recv_weights is weights
        assert metadata is original_metadata and count.tolist() == [5]
        calls.append("reduce")
        return dense, compact_metadata

    def combine(x, handle, **kwargs):
        assert x is dense and handle is native_handle
        assert handle.recv_src_metadata is compact_metadata
        assert kwargs["topk_weights"] is None
        assert kwargs["async_with_compute_stream"] is False
        assert kwargs["allocate_on_comm_stream"] is False
        calls.append("combine")
        return output, None, None

    monkeypatch.setattr(moe, "ep_reduce_decode_output", reduce)
    monkeypatch.setattr(moe.dist_group_manager, "ep_buffer", SimpleNamespace(combine=combine))
    actual, hook = moe.deepep_v2_decode_combine(gemm, (native_handle, weights))
    assert actual is output and calls == ["reduce", "combine"]
    hook()
    assert calls == ["reduce", "combine"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2_local_reduction_preserves_source_metadata():
    outputs = torch.tensor([[1.0] * 8, [2.0] * 8, [4.0] * 8], device="cuda", dtype=torch.bfloat16)
    weights = torch.tensor([0.5, 0.25, 0.0], device="cuda")
    metadata = torch.tensor([[7, 9, 0, 1], [3, 4, -1, -1]], device="cuda", dtype=torch.int32)
    valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    dense, compact = moe.ep_reduce_decode_output(outputs, weights, metadata, valid)

    assert dense.tolist() == [[1.0] * 8, [0.0] * 8]
    assert compact.tolist() == [[7, 9, 0, -1], [3, 4, 1, -1]]
    assert metadata.tolist() == [[7, 9, 0, 1], [3, 4, -1, -1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2_psum_activation_matches_v41_clamped_masked_path():
    rows = torch.randn((2, 3, 256), device="cuda", dtype=torch.bfloat16) * 8
    counts = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    ends = torch.tensor([3, 10], device="cuda", dtype=torch.int32)
    expanded = torch.zeros((16, 256), device="cuda", dtype=rows.dtype)
    expanded[:3] = rows[0]
    expanded[8:10] = rows[1, :2]
    masked_q = torch.empty((2, 3, 128), device="cuda", dtype=torch.float8_e4m3fn)
    masked_scale = torch.empty((2, 3, 1), device="cuda", dtype=torch.float32)
    expanded_q = torch.zeros((16, 128), device="cuda", dtype=torch.float8_e4m3fn)
    expanded_scale = torch.zeros((16, 1), device="cuda", dtype=torch.float32)

    moe.silu_and_mul_masked_post_quant_fwd(rows, masked_q, masked_scale, 128, counts, limit=7.0)
    moe.silu_and_mul_masked_post_quant_fwd(
        expanded, expanded_q, expanded_scale, 128, ends, limit=7.0, expert_alignment=8
    )

    torch.testing.assert_close(expanded_q[:3].float(), masked_q[0].float(), rtol=0, atol=0)
    torch.testing.assert_close(expanded_q[8:10].float(), masked_q[1, :2].float(), rtol=0, atol=0)
    torch.testing.assert_close(expanded_scale[:3], masked_scale[0], rtol=0, atol=0)
    torch.testing.assert_close(expanded_scale[8:10], masked_scale[1, :2], rtol=0, atol=0)
    assert torch.count_nonzero(expanded_q[3:8].float()).item() == 0
    assert torch.count_nonzero(expanded_scale[3:8]).item() == 0


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
