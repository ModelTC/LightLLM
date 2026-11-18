"""Transport selection and scratch storage must match actual execution paths."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.distributed import communication_op as comm
from lightllm.utils import device_utils


@pytest.mark.parametrize(
    "tp,node,nvlink,expected",
    [
        (8, 4, True, False),
        (6, 8, True, False),
        (4, 8, True, True),
        (2, 4, True, True),
        (4, 4, False, False),
        (1, 4, True, False),
    ],
)
def test_custom_allreduce_requires_whole_node_local_groups(monkeypatch, tp, node, nvlink, expected):
    group = comm.CustomProcessGroup.__new__(comm.CustomProcessGroup)
    group.dp_world_size = tp
    monkeypatch.setattr(comm, "get_node_world_size", lambda: node)
    monkeypatch.setattr(comm, "has_nvlink", lambda: nvlink)
    assert group._support_custom_allreduce() is expected


@pytest.fixture
def buffers(monkeypatch):
    args = SimpleNamespace(
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        enable_prefill_cudagraph=False,
        enable_rl=False,
        eplb_num_redundant_experts_per_rank=2,
        data_type="bfloat16",
        run_mode="prefill",
        disable_cudagraph=True,
    )
    monkeypatch.setattr(comm, "get_env_start_args", lambda: args)
    monkeypatch.setattr(comm, "get_global_world_size", lambda: 8)
    monkeypatch.setattr(comm, "get_deepep_num_max_dispatch_tokens_per_rank", lambda: 1024)
    monkeypatch.setattr(comm, "get_device_sm_count", lambda: 132)
    monkeypatch.setattr(comm, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(comm, "HAS_DEEPEP", True)
    monkeypatch.setattr(comm.dist, "new_group", Mock(return_value=object()))
    monkeypatch.setattr(comm.dist, "barrier", Mock())
    monkeypatch.setattr(comm.torch.cuda, "current_device", lambda: 0)
    legacy_storage = torch.empty(1536, dtype=torch.uint8)
    legacy = Mock()
    legacy.get_local_buffer_tensor.return_value = legacy_storage
    legacy_class = Mock(return_value=legacy)
    legacy_class.get_low_latency_rdma_size_hint.return_value = 1281
    elastic = Mock()
    elastic.get_theoretical_num_sms.return_value = 24
    monkeypatch.setattr(
        comm, "deep_ep", SimpleNamespace(Buffer=legacy_class, ElasticBuffer=Mock(return_value=elastic)), raising=False
    )
    monkeypatch.setattr(comm.DistributeGroupManager, "_set_num_sms_for_deep_gemm", Mock())
    manager = comm.DistributeGroupManager()
    manager.groups = [object(), object()]
    return args, manager, legacy_class, legacy, legacy_storage


def initialize(manager):
    manager.new_deepep_group(256, 7168, {"fp8w8a8-b128-deepgemm"}, 8, 2048)


@pytest.mark.parametrize(
    "mode,disable_graph,eplb,experts",
    [
        ("prefill", True, True, 272),
        ("prefill", False, True, 272),
        ("normal", True, True, 272),
        ("normal", False, True, 272),
        ("decode", True, True, 256),
        ("decode", False, True, 256),
        ("prefill", True, False, 256),
        ("prefill", False, False, 256),
        ("normal", True, False, 256),
        ("normal", False, False, 256),
        ("decode", True, False, 256),
        ("decode", False, False, 256),
    ],
)
def test_eplb_preserves_transport_and_uses_physical_expert_count(buffers, mode, disable_graph, eplb, experts):
    args, manager, constructor, legacy, storage = buffers
    args.run_mode, args.disable_cudagraph = mode, disable_graph
    args.enable_prefill_eplb = eplb
    initialize(manager)
    constructor.assert_not_called()
    assert manager.ll_num_experts == experts
    manager.ep_buffer.get_theoretical_num_sms.assert_called_once_with(experts, 8)
    manager._set_num_sms_for_deep_gemm.assert_called_once_with(24)
    assert manager.ep_non_overlap_num_sms == 78
    assert not hasattr(manager, "ep_low_latency_buffer")
    manager.clear_deepep_buffer()
    legacy.clean_low_latency_buffer.assert_not_called()


def test_disabling_ep_clears_buffers(buffers):
    args, manager, constructor, _, _ = buffers
    initialize(manager)
    args.enable_ep_moe = False
    initialize(manager)
    assert not hasattr(manager, "ep_low_latency_buffer")
    assert manager.ep_buffer is None
    constructor.assert_not_called()
