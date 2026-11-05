"""统一 decode 路由入口的环境选择、参数转发与 CUDA Graph 集成测试。"""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import decode_eplb_topk_ids as routing
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
    eplb_hash_targets_topk_ids,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_minimal import eplb_minimal_topk_ids
from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_rank_weighted import (
    eplb_rank_weighted_topk_ids,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    build_initial_local_expert_ids,
    build_logical_to_rank_physical_map,
    build_logical_to_physical_map,
    build_minimal_logical_to_physical_map,
)
from lightllm.utils.envs_utils import get_eplb_decode_routing_mode


@pytest.fixture(autouse=True)
def reset_routing_environment(monkeypatch):
    # 每个测试独立读取环境，避免进程级配置缓存污染其他算法的测试。
    monkeypatch.delenv("LIGHTLLM_EPLB_DECODE_ROUTING_MODE", raising=False)
    get_eplb_decode_routing_mode.cache_clear()
    yield
    get_eplb_decode_routing_mode.cache_clear()


@pytest.mark.parametrize("mode", [None, "default_targets", "hash_targets"])
@pytest.mark.parametrize("record_statistics", [False, True])
@pytest.mark.parametrize("node_world_size", [1, 2, 4])
@pytest.mark.parametrize("world_size", [4, 8, 16])
@pytest.mark.parametrize("threshold_offset", [-1, 0, 1])
def test_entrypoint_selects_algorithm_and_forwards_all_arguments(
    monkeypatch, mode, record_statistics, node_world_size, world_size, threshold_offset
):
    if mode is not None:
        monkeypatch.setenv("LIGHTLLM_EPLB_DECODE_ROUTING_MODE", mode)
    tokens = 512 // world_size + threshold_offset
    logical = torch.tensor([[0, 2, 4]], dtype=torch.int32).repeat(tokens, 1)
    physical = torch.tensor([[0, 4, 8]], dtype=torch.int32).repeat(tokens, 1)
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.zeros((32, 3 + world_size * 4), dtype=torch.int32),
        logical_to_rank_physical_map=torch.full((32, world_size), -1, dtype=torch.int32),
        minimal_logical_to_physical_map=torch.arange(32, dtype=torch.int32),
        current_rank=0,
        num_physical_experts_per_rank=4,
        node_world_size=node_world_size,
        route_counter=torch.zeros((32, 32), dtype=torch.int64) if record_statistics else None,
        update_route_counter=record_statistics,
        route_recording_enabled=torch.ones((1,), dtype=torch.int32) if record_statistics else None,
    )
    calls = []

    def hash_targets(**actual):
        calls.append(("hash_targets", actual))
        return physical

    def default_targets(**actual):
        calls.append(("default_targets", actual))
        return physical

    def minimal_targets(**actual):
        calls.append(("minimal", actual))
        return physical

    monkeypatch.setattr(routing, "eplb_hash_targets_topk_ids", hash_targets)
    monkeypatch.setattr(routing, "eplb_rank_weighted_topk_ids", default_targets)
    monkeypatch.setattr(routing, "eplb_minimal_topk_ids", minimal_targets)
    assert routing.eplb_repair_decode_topk_ids(**kwargs) is physical
    assert len(calls) == 1
    selected_mode = mode or "default_targets"
    if selected_mode == "default_targets" and tokens * world_size < 512:
        selected_mode = "minimal"
    assert calls[0][0] == selected_mode
    # 最小映射只需要逻辑 ID、唯一槽位表和统计参数；大批量加权路由及 hash 参数不变。
    forwarded = dict(kwargs)
    if selected_mode == "minimal":
        for name in (
            "logical_to_physical_map",
            "logical_to_rank_physical_map",
            "current_rank",
            "num_physical_experts_per_rank",
            "node_world_size",
        ):
            forwarded.pop(name)
    else:
        forwarded.pop("minimal_logical_to_physical_map")
    if selected_mode == "default_targets":
        forwarded.pop("logical_to_physical_map")
        forwarded.pop("num_physical_experts_per_rank")
    assert calls[0][1].keys() == forwarded.keys()
    assert all(calls[0][1][key] is value for key, value in forwarded.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for decode routing")
@pytest.mark.parametrize("mode", [None, "default_targets"])
def test_weighted_entrypoint_defaults_to_inverse_replica_coverage(monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv("LIGHTLLM_EPLB_DECODE_ROUTING_MODE", mode)
    # 原始覆盖数会将 expert 0 分配到 rank 1/3；倒数加权应与只有一个副本的 expert 3 共址到 rank 2。
    placement = [[4, 5, 6, 7, 8], [0, 1, 2, 4, 9], [0, 3, 5, 6, 7], [0, 1, 2, 8, 9], [1, 2, 4, 5, 9]]
    output = routing.eplb_repair_decode_topk_ids(
        logical_topk_ids=torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32, device="cuda").repeat(104, 1),
        logical_to_physical_map=torch.tensor(
            build_logical_to_physical_map(placement, 10, current_rank=0, node_world_size=5),
            dtype=torch.int32,
            device="cuda",
        ),
        logical_to_rank_physical_map=torch.tensor(
            build_logical_to_rank_physical_map(placement, 10), dtype=torch.int32, device="cuda"
        ),
        minimal_logical_to_physical_map=torch.tensor(
            build_minimal_logical_to_physical_map(placement, 10), dtype=torch.int32, device="cuda"
        ),
        current_rank=0,
        num_physical_experts_per_rank=5,
        node_world_size=5,
    )
    assert output[0, 0].item() == 10
    assert output[0, 3].item() == 11
    assert output[0, 4].item() == 0


@pytest.mark.parametrize("mode", [None, "default_targets", "hash_targets"])
@pytest.mark.parametrize("run_mode", ["prefill", "decode"])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("node_world_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for decode routing")
def test_entrypoint_graph_matches_algorithm_and_updates_layout(
    monkeypatch, mode, run_mode, id_dtype, node_world_size, num_tokens
):
    if mode is not None:
        monkeypatch.setenv("LIGHTLLM_EPLB_DECODE_ROUTING_MODE", mode)
    selected_mode = mode or "default_targets"
    # 两种 run mode 都复用 prefill 表，额外构造按 rank 查询表。
    placement = [[0, 1, 6, 7], [2, 3, 0, 1], [4, 5, 0, 1], [6, 7, 2, 4]]
    default = build_logical_to_physical_map(placement, 8, current_rank=0, node_world_size=node_world_size)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(placement, 8)
    logical = torch.tensor([[0, 2, 4]], dtype=id_dtype, device="cuda").repeat(num_tokens, 1)
    default_map = torch.tensor(default, dtype=torch.int32, device="cuda")
    logical_to_rank_physical_map = torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda")
    minimal_map = torch.tensor(build_minimal_logical_to_physical_map(placement, 8), dtype=torch.int32, device="cuda")
    addresses = (default_map.data_ptr(), logical_to_rank_physical_map.data_ptr(), minimal_map.data_ptr())
    counter = torch.zeros((8, 8), dtype=torch.int64, device="cuda")
    recording_enabled = torch.ones((1,), dtype=torch.int32, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=default_map,
        logical_to_rank_physical_map=logical_to_rank_physical_map,
        minimal_logical_to_physical_map=minimal_map,
        current_rank=0,
        num_physical_experts_per_rank=4,
        route_counter=counter,
        update_route_counter=True,
        route_recording_enabled=recording_enabled,
    )
    expected_function = eplb_hash_targets_topk_ids if selected_mode == "hash_targets" else eplb_rank_weighted_topk_ids
    expected_kwargs = dict(kwargs)
    expected_kwargs["node_world_size"] = node_world_size
    if selected_mode == "default_targets" and num_tokens * 4 < 512:
        expected_function = eplb_minimal_topk_ids
        for name in (
            "logical_to_physical_map",
            "logical_to_rank_physical_map",
            "current_rank",
            "num_physical_experts_per_rank",
            "node_world_size",
        ):
            expected_kwargs.pop(name)
    else:
        expected_kwargs.pop("minimal_logical_to_physical_map")
    if expected_function is eplb_rank_weighted_topk_ids:
        expected_kwargs.pop("logical_to_physical_map")
        expected_kwargs.pop("num_physical_experts_per_rank")
    kwargs["node_world_size"] = node_world_size
    expected = expected_function(**expected_kwargs)
    assert torch.equal(routing.eplb_repair_decode_topk_ids(**kwargs), expected)
    assert torch.equal(logical, torch.tensor([[0, 2, 4]], dtype=id_dtype, device="cuda").repeat(num_tokens, 1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = routing.eplb_repair_decode_topk_ids(**kwargs)
    counter.zero_()
    graph.replay()
    assert torch.equal(output, expected)
    assert output.dtype == logical.dtype and output.shape == logical.shape
    selected = torch.nn.functional.one_hot(logical.cpu().to(torch.long), num_classes=8).sum(dim=1)
    expected_counter = torch.triu(input=selected.T @ selected)
    assert torch.equal(counter.cpu(), expected_counter)

    # 只改 rank 3 的冗余槽，不改各卡主槽，因此两种 run mode 都合法。
    updated = [[0, 1, 6, 7], [2, 3, 0, 1], [4, 5, 0, 1], [6, 7, 3, 5]]
    default = build_logical_to_physical_map(updated, 8, current_rank=0, node_world_size=node_world_size)
    logical_to_rank_physical_map_cpu = build_logical_to_rank_physical_map(updated, 8)
    default_map.copy_(torch.tensor(default, dtype=torch.int32, device="cuda"))
    logical_to_rank_physical_map.copy_(torch.tensor(logical_to_rank_physical_map_cpu, dtype=torch.int32, device="cuda"))
    minimal_map.copy_(torch.tensor(build_minimal_logical_to_physical_map(updated, 8), dtype=torch.int32, device="cuda"))
    recording_enabled.zero_()
    expected_updated = expected_function(**expected_kwargs)
    graph.replay()
    assert torch.equal(output, expected_updated)
    assert torch.equal(counter.cpu(), expected_counter)
    physical_to_logical = torch.tensor(updated, dtype=torch.int64, device="cuda").flatten()
    assert torch.equal(physical_to_logical[output.to(torch.long)], logical.to(torch.long))
    assert (default_map.data_ptr(), logical_to_rank_physical_map.data_ptr(), minimal_map.data_ptr()) == addresses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for decode routing")
@pytest.mark.parametrize("mode", ["default_targets", "hash_targets"])
def test_entrypoint_accepts_topk_above_32(monkeypatch, mode):
    monkeypatch.setenv("LIGHTLLM_EPLB_DECODE_ROUTING_MODE", mode)
    logical = torch.arange(33, dtype=torch.int32, device="cuda").reshape(1, 33)
    metadata = build_logical_to_physical_map([list(range(64))], 64, current_rank=0, node_world_size=1)
    output = routing.eplb_repair_decode_topk_ids(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.tensor(metadata, dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.arange(64, dtype=torch.int32, device="cuda").reshape(64, 1),
        minimal_logical_to_physical_map=torch.arange(64, dtype=torch.int32, device="cuda"),
        current_rank=0,
        num_physical_experts_per_rank=64,
    )
    assert torch.equal(output, logical)


def test_weighted_entrypoint_rejects_empty_inputs(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_EPLB_DECODE_ROUTING_MODE", "default_targets")
    with pytest.raises(AssertionError, match="logical_topk_ids must be non-empty"):
        routing.eplb_repair_decode_topk_ids(
            logical_topk_ids=torch.empty((0, 8), dtype=torch.int32),
            logical_to_physical_map=torch.zeros((8, 19), dtype=torch.int32),
            logical_to_rank_physical_map=torch.zeros((8, 4), dtype=torch.int32),
            minimal_logical_to_physical_map=torch.arange(8, dtype=torch.int32),
            current_rank=0,
            num_physical_experts_per_rank=4,
            node_world_size=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("source_rank", [0, 1])
def test_small_batch_graph_uses_minimal_map_and_sees_inplace_updates(dtype, source_rank):
    placement = [[2, 0, 1], [3, 1, 2]]
    logical = torch.tensor([[0, 1, 2, 3]], dtype=dtype, device="cuda")
    minimal_map = torch.tensor(build_minimal_logical_to_physical_map(placement, 4), dtype=torch.int32, device="cuda")
    candidate_map = torch.tensor(
        build_logical_to_physical_map(placement, 4, source_rank, 2), dtype=torch.int32, device="cuda"
    )
    rank_map = torch.tensor(build_logical_to_rank_physical_map(placement, 4), dtype=torch.int32, device="cuda")
    counter = torch.zeros((4, 4), dtype=torch.int64, device="cuda")
    recording_enabled = torch.ones(1, dtype=torch.int32, device="cuda")
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=candidate_map,
        logical_to_rank_physical_map=rank_map,
        minimal_logical_to_physical_map=minimal_map,
        current_rank=source_rank,
        num_physical_experts_per_rank=3,
        node_world_size=2,
        route_counter=counter,
        update_route_counter=True,
        route_recording_enabled=recording_enabled,
    )
    addresses = (candidate_map.data_ptr(), rank_map.data_ptr(), minimal_map.data_ptr())
    # 两个源 rank 都使用同一最小映射；本卡有其他副本时也不改用本卡槽位。
    assert routing.eplb_repair_decode_topk_ids(**kwargs).tolist() == [[1, 2, 5, 3]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = routing.eplb_repair_decode_topk_ids(**kwargs)
    counter.zero_()
    graph.replay()
    assert output.tolist() == [[1, 2, 5, 3]]
    expected_counter = torch.triu(torch.ones((4, 4), dtype=torch.int64))
    assert torch.equal(counter.cpu(), expected_counter)

    # 更新布局后只覆盖表内容，已捕获的查表 kernel 必须读取新的物理槽。
    updated = [[1, 0, 3], [2, 3, 1]]
    candidate_map.copy_(
        torch.tensor(build_logical_to_physical_map(updated, 4, source_rank, 2), dtype=torch.int32, device="cuda")
    )
    rank_map.copy_(torch.tensor(build_logical_to_rank_physical_map(updated, 4), dtype=torch.int32, device="cuda"))
    minimal_map.copy_(torch.tensor(build_minimal_logical_to_physical_map(updated, 4), dtype=torch.int32, device="cuda"))
    recording_enabled.zero_()
    graph.replay()
    assert output.tolist() == [[1, 0, 3, 4]]
    assert torch.equal(counter.cpu(), expected_counter)
    assert (candidate_map.data_ptr(), rank_map.data_ptr(), minimal_map.data_ptr()) == addresses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("world_size", [8, 16])
@pytest.mark.parametrize("threshold_offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_gpu_entrypoint_switches_at_exact_global_token_threshold(world_size, threshold_offset, dtype):
    tokens = 512 // world_size + threshold_offset
    current_rank = world_size - 1
    node_world_size = world_size // 2
    placement = build_initial_local_expert_ids(64, world_size, 4)
    logical = (torch.arange(tokens * 8, dtype=dtype, device="cuda") % 64).reshape(tokens, 8)
    minimal_map = torch.tensor(build_minimal_logical_to_physical_map(placement, 64), dtype=torch.int32, device="cuda")
    rank_map = torch.tensor(build_logical_to_rank_physical_map(placement, 64), dtype=torch.int32, device="cuda")
    candidate_map = torch.tensor(
        build_logical_to_physical_map(placement, 64, current_rank, node_world_size), dtype=torch.int32, device="cuda"
    )
    output = routing.eplb_repair_decode_topk_ids(
        logical_topk_ids=logical,
        logical_to_physical_map=candidate_map,
        logical_to_rank_physical_map=rank_map,
        minimal_logical_to_physical_map=minimal_map,
        current_rank=current_rank,
        num_physical_experts_per_rank=len(placement[0]),
        node_world_size=node_world_size,
    )
    if tokens * world_size < 512:
        expected = minimal_map[logical.to(torch.long)].to(dtype)
        assert output[0, 0].item() == minimal_map[0].item()
    else:
        expected = eplb_rank_weighted_topk_ids(
            logical_topk_ids=logical,
            logical_to_rank_physical_map=rank_map,
            current_rank=current_rank,
            node_world_size=node_world_size,
        )
        assert output[0, 0].item() == rank_map[0, current_rank].item()
    assert minimal_map[0].item() != rank_map[0, current_rank].item()
    assert torch.equal(output, expected)
    physical_to_logical = torch.tensor(placement, dtype=dtype, device="cuda").flatten()
    assert torch.equal(physical_to_logical[output.to(torch.long)], logical)
